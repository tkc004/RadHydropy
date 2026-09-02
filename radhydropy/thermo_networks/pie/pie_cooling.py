"""Non-RT photoionization-equilibrium cooling with a fixed UV background."""

import numpy as np

from radhydropy.thermo_networks.base import ThermochemistryNetwork
from radhydropy.thermo_networks.cie.cie_cooling import _state, _update_temperature
from radhydropy.thermo_networks.hydrogen import _rotational_specific_energy_code
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.units import from_unit_value, to_unit_value


class PIEUVBGCoolingNetwork(ThermochemistryNetwork):
    """Apply a tabulated HM12 total heating/cooling source without RT.

    The table is evaluated in temperature, hydrogen density, redshift, and
    metallicity.  It already contains the H/He plus metal volumetric rates,
    so no local photon field or separate H/He photoheating term is added.
    """

    name = "pie_uvbg_cooling"
    scalar_fields = ()

    def enabled(self, fluid, par):
        return bool(
            getattr(par, "metal_pie_enabled", False)
            and getattr(par, "metal_pie_table", None) is not None
        )

    def radiation_enabled(self, fluid, par):
        return False

    def radiation_evolution_enabled(self, fluid, par):
        return False

    def advect_ionization_fraction(self, *args, **kwargs):
        return None

    def source_state(self, mesh, fluid, par):
        table = getattr(par, "metal_pie_table", None)
        if table is None:
            raise ValueError("pie_uvbg_cooling requires metal_pie_table")
        if table.component != "hydrogen+helium+metals":
            raise ValueError(
                "pie_uvbg_cooling requires a total H/He+metals PIE table; "
                "use hydrogen_helium for a metal-only PIE table"
            )
        state = _state(mesh, fluid, par)
        state.update(
            par=par,
            metallicity=float(getattr(par, "metallicity", 1.0)),
            redshift=float(getattr(par, "metal_pie_redshift", 0.0)),
            hydrogen_mass_fraction=float(
                getattr(par, "hydrogen_mass_fraction", 1.0)
            ),
            cooling_safety_factor=float(
                getattr(par, "cooling_safety_factor", 0.1)
            ),
            compton_cmb_enabled=bool(
                getattr(par, "compton_cmb_enabled", False)
            ),
            compton_cmb_redshift=float(
                getattr(par, "compton_cmb_redshift", 0.0)
            ),
            cmb_temperature_0_K=float(
                to_unit_value(getattr(par, "cmb_temperature_0", 2.7255), "K")
            ),
        )
        return state

    def ionization_fraction_rate(self, state, ngamma):
        return np.zeros_like(state["temperature_K"])

    def thermal_rate(self, state, ngamma):
        nH = state["rho_g_cm3"] * state["hydrogen_mass_fraction"] / PROTON_MASS_CGS
        heating, cooling = state["par"].metal_pie_table.rates(
            state["temperature_K"],
            nH,
            metallicity=state["metallicity"],
            redshift=state["redshift"],
        )
        max_heating_density = getattr(
            state["par"], "metal_pie_photoheating_max_density_cm3", 50.0
        )
        if max_heating_density is not None:
            heating = np.where(
                nH > float(max_heating_density), 0.0, heating
            )
        return heating - cooling

    def get_timestep(self, state, ngamma, remaining_s, dtmax_s):
        rate = self.thermal_rate(state, ngamma)
        thermal_density = state["specific_energy_erg_g"] * state["rho_g_cm3"]
        cooling_time = thermal_density / np.maximum(np.abs(rate), 1.0e-99)
        candidate = np.min(state["cooling_safety_factor"] * cooling_time)
        return min(float(remaining_s), float(dtmax_s), float(candidate)), rate

    def update_temperature_from_energy(self, state):
        _update_temperature(state)

    def ionization_fraction_implicit_update(self, state, ngamma, dt_s):
        return None

    def apply_state(self, state, fluid, par):
        return None

    def get_source_timestep_fast(self, mesh, fluid, par, remaining):
        state = self.source_state(mesh, fluid, par)
        code = state["code"]
        remaining_s = (
            float(to_unit_value(remaining, code.time_unit))
            * state["source_scale_factor"]**2
        )
        # The thermal update is implicit, so the cooling time no longer has
        # to limit the hydro/source timestep.  The hydro CFL step is still
        # passed in as ``remaining``.
        rate = self.thermal_rate(state, None)
        return from_unit_value(
            remaining_s / state["source_scale_factor"]**2,
            code.time_unit,
        ), rate

    @staticmethod
    def _energy_at_temperature(state, temperature_K):
        return (
            BOLTZMANN_CONSTANT_CGS
            * np.asarray(temperature_K, dtype=float)
            / ((state["gamma"] - 1.0) * state["mu"] * PROTON_MASS_CGS)
        )

    def _implicit_energy_step(self, state, old_energy, dt_s, floor_K):
        """Solve one backward-Euler thermal step with vectorized bisection."""
        table = state["par"].metal_pie_table
        lower_K = max(float(floor_K), 10.0 ** float(table.log_temperature[0]))
        upper_K = 10.0 ** float(table.log_temperature[-1])
        lower = np.full_like(old_energy, lower_K, dtype=float)
        upper = np.full_like(old_energy, upper_K, dtype=float)
        rho = np.maximum(state["rho_g_cm3"], 1.0e-99)

        def residual(temperature):
            trial = dict(state)
            trial["temperature_K"] = temperature
            rate = self.thermal_rate(trial, None)
            return self._energy_at_temperature(trial, temperature) - old_energy - dt_s * rate / rho

        f_lower = residual(lower)
        f_upper = residual(upper)
        bracketed = (f_lower <= 0.0) & (f_upper >= 0.0)

        # If the requested step would cross the temperature floor, clamping
        # is the physically intended result.  Other unbracketed cells are
        # handed to the explicit fallback by marking them unsuccessful.
        old_temperature = state["temperature_K"]
        old_rate = self.thermal_rate(state, None)
        explicit_energy = old_energy + dt_s * old_rate / rho
        floor_energy = self._energy_at_temperature(state, lower)
        # The HM12 table does not represent temperatures below its lower
        # tabulated range.  Cold initial conditions (for example a 2.7 K
        # cosmological seed) must be placed directly on the configured floor;
        # sending them through the bracket/retry loop can create an enormous
        # number of futile source substeps.
        # At exactly the floor, the gas must still be allowed to heat.  Using
        # ``<=`` here permanently pinned any cell that touched 100 K, even
        # when HM12 photoheating was positive and its PIE equilibrium was
        # near 10^4 K.
        below_floor = old_temperature < lower_K
        bracketed &= ~below_floor
        trial_energy = np.where(
            below_floor,
            floor_energy,
            np.where(
                bracketed,
                old_energy,
                np.where(explicit_energy <= floor_energy, floor_energy, old_energy),
            ),
        )
        successful = below_floor | bracketed | (explicit_energy <= floor_energy)
        if np.any(bracketed):
            # Bisection is deliberately used instead of an unconstrained
            # Newton iteration because the tabulated net rate is not
            # necessarily monotonic in temperature.
            for _ in range(int(getattr(state["par"], "pie_uvbg_implicit_max_iterations", 64))):
                middle = 0.5 * (lower + upper)
                f_middle = residual(middle)
                go_right = f_middle < 0.0
                lower = np.where(bracketed & go_right, middle, lower)
                upper = np.where(bracketed & ~go_right, middle, upper)
            root = 0.5 * (lower + upper)
            root_energy = self._energy_at_temperature(state, root)
            trial_energy = np.where(bracketed, root_energy, trial_energy)
        trial_energy = np.maximum(trial_energy, floor_energy)
        return trial_energy, successful

    def _implicit_converged_step(self, state, old_energy, dt_s, floor_K):
        """Compare a full implicit step with two implicit half steps."""
        full_energy, full_ok = self._implicit_energy_step(
            state, old_energy, dt_s, floor_K
        )
        half_energy, first_ok = self._implicit_energy_step(
            state, old_energy, 0.5 * dt_s, floor_K
        )
        second_state = dict(state)
        second_state["specific_energy_erg_g"] = half_energy
        _update_temperature(second_state)
        half_energy, second_ok = self._implicit_energy_step(
            second_state, half_energy, 0.5 * dt_s, floor_K
        )
        relative_difference = np.abs(half_energy - full_energy) / np.maximum(
            np.abs(half_energy), self._energy_at_temperature(state, floor_K)
        )
        tolerance = float(getattr(state["par"], "pie_uvbg_implicit_tolerance", 1.0e-3))
        converged = (
            full_ok
            & first_ok
            & second_ok
            & np.isfinite(relative_difference)
            & (relative_difference <= tolerance)
        )
        return half_energy, converged

    def _explicit_fallback_step(self, state, old_energy, remaining_s, floor_K):
        """Advance one chunk with the existing cooling-time subcycling."""
        state["specific_energy_erg_g"] = old_energy.copy()
        _update_temperature(state)
        dt_s, rate = self.get_timestep(state, None, remaining_s, remaining_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = remaining_s
        dt_s = min(dt_s, remaining_s)
        state["specific_energy_erg_g"] += (
            rate / np.maximum(state["rho_g_cm3"], 1.0e-99) * dt_s
        )
        minimum_energy = self._energy_at_temperature(state, floor_K)
        state["specific_energy_erg_g"] = np.maximum(
            state["specific_energy_erg_g"], minimum_energy
        )
        return state["specific_energy_erg_g"].copy(), dt_s

    def apply_fast(self, dt, mesh, fluid, par):
        state = self.source_state(mesh, fluid, par)
        code = state["code"]
        remaining_s = (
            float(to_unit_value(dt, code.time_unit))
            * state["source_scale_factor"]**2
        )
        floor_K = float(
            to_unit_value(getattr(par, "cooling_temperature_floor", 1.0), "K")
        )
        source_steps = 0
        retries = int(getattr(par, "pie_uvbg_implicit_max_retries", 8))
        while remaining_s > 0.0:
            _update_temperature(state)
            old_energy = state["specific_energy_erg_g"].copy()
            dt_s = remaining_s
            accepted = False
            for _ in range(retries + 1):
                if getattr(par, "pie_uvbg_implicit_step_doubling", True):
                    new_energy, converged = self._implicit_converged_step(
                        state, old_energy, dt_s, floor_K
                    )
                else:
                    new_energy, converged = self._implicit_energy_step(
                        state, old_energy, dt_s, floor_K
                    )
                if np.all(converged):
                    state["specific_energy_erg_g"] = new_energy
                    accepted = True
                    break
                dt_s *= 0.5
            if not accepted:
                # Do not enter an unbounded explicit cooling-time loop here.
                # During unresolved central collapse the tabulated cooling
                # time can become arbitrarily short, making one hydro step
                # effectively hang.  The backward-Euler solve is bounded and
                # temperature-floor limited, so accept one full-step
                # implicit update after the retry budget is exhausted.
                new_energy, _ = self._implicit_energy_step(
                    state, old_energy, remaining_s, floor_K
                )
                state["specific_energy_erg_g"] = new_energy
                dt_s = remaining_s
            remaining_s -= dt_s
            source_steps += 1

        _update_temperature(state)
        interior = state["interior"]
        internal_super = (
            state["specific_energy_erg_g"]
            * state["source_temperature_factor"]
        )
        total_super = internal_super + 0.5 * state["velocity_supercomoving_cm_s"]**2
        fluid.Energy_code[interior] = from_unit_value(
            state["mass_g"] * total_super,
            code.energy_unit,
        )
        rotational_code = _rotational_specific_energy_code(mesh, fluid, par)
        fluid.Energy_code[interior] += from_unit_value(
            np.asarray(fluid.Mass_code[interior], dtype=float) * rotational_code,
            code.energy_unit,
        )
        fluid.temp_code[interior] = from_unit_value(
            state["temperature_K"] * state["source_temperature_factor"],
            code.temperature_unit,
        )
        if hasattr(fluid.eos, "pressure"):
            fluid.pre_code[interior] = fluid.eos.pressure(
                fluid.rho_code[interior],
                fluid.temp_code[interior],
                fluid.mu[interior],
            )
        else:
            internal_code = internal_super / code.velocity_in_cgs**2
            fluid.pre_code[interior] = (
                (state["gamma"] - 1.0)
                * np.asarray(fluid.rho_code[interior], dtype=float)
                * internal_code
            )
        return source_steps
