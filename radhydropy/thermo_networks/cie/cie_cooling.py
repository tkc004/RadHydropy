"""Collisional-ionization-equilibrium radiative cooling network."""

from pathlib import Path

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.thermo_networks.cie.cie_tables import CIETable
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.units import _code_units, from_unit_value, to_unit_value
from radhydropy.thermo_networks.base import ThermochemistryNetwork
from radhydropy.thermo_networks.compton import cmb_compton_rate
from radhydropy.thermo_networks.hydrogen import _fast_source_scaling


_TABLE_CACHE = {}


def _default_table_paths():
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[4] / "CHIANTI_11.0.2_database",
        module_path.parents[3] / "CHIANTI_11.0.2_database",
    )
    database = next((path for path in candidates if path.is_dir()), candidates[0])
    return (
        database / "cooling_tables" / "chianti_cie_ion_fractions.h5",
        database / "cooling_tables" / "chianti_cooling_table.h5",
        database / "abundance" / "sun_photospheric_2015_scott.abund",
    )


def _get_table(par):
    defaults = _default_table_paths()
    paths = tuple(
        Path(value if value is not None else default).expanduser().resolve()
        for name, default in zip(
            ("cie_ion_fraction_table", "cie_cooling_table", "cie_abundance_file"),
            defaults,
            strict=True,
        )
        for value in (getattr(par, name, None),)
    )
    if paths not in _TABLE_CACHE:
        _TABLE_CACHE[paths] = CIETable(*paths)
    return _TABLE_CACHE[paths]


def _state(mesh, fluid, par):
    code = _code_units(par)
    if code is None:
        raise ValueError("CIE cooling requires par.CodeUnits")
    interior = slice(par.noghost, par.noghost + par.nogrid)
    gamma = getattr(getattr(fluid, "eos", None), "gamma", getattr(par, "gamma", 5.0 / 3.0))
    scaling = _fast_source_scaling(fluid, par, gamma)
    rho_super = to_unit_value(fluid.rho[interior], code.density_unit)
    rho = rho_super / scaling["density_factor"]
    velocity_super = to_unit_value(fluid.vel[interior], code.velocity_unit)
    velocity = velocity_super / scaling["velocity_factor"]
    volume_code = np.asarray(mesh.vol[interior], dtype=float)
    volume = to_unit_value(volume_code, code.volume_unit) * scaling["density_factor"]
    if hasattr(fluid, "Mass"):
        mass = to_unit_value(fluid.Mass[interior], code.mass_unit)
    else:
        mass = rho_super * volume_code * code.mass_in_cgs
    total_energy_super = to_unit_value(fluid.Energy[interior], code.energy_unit)
    specific_total_super = total_energy_super / np.maximum(mass, 1.0e-99)
    specific_internal = np.maximum(
        specific_total_super - 0.5 * velocity_super**2,
        0.0,
    ) / scaling["temperature_factor"]
    thermal_energy = specific_internal * mass
    mu = np.asarray(fluid.mu[interior], dtype=float)
    return {
        "interior": interior,
        "rho_g_cm3": rho,
        "volume_cm3": volume,
        "velocity_cm_s": velocity,
        "thermal_energy_erg": thermal_energy,
        "specific_energy_erg_g": specific_internal,
        "temperature_K": to_unit_value(fluid.temp[interior], code.temperature_unit) / scaling["temperature_factor"],
        "gamma": gamma,
        "mu": mu,
        "code": code,
        "mass_g": mass,
        "velocity_supercomoving_cm_s": velocity_super,
        "source_scale_factor": scaling["scale_factor"],
        "source_temperature_factor": scaling["temperature_factor"],
    }


def _update_temperature(state):
    state["temperature_K"] = np.maximum(
        state["specific_energy_erg_g"]
        * (state["gamma"] - 1.0)
        * state["mu"]
        * PROTON_MASS_CGS
        / BOLTZMANN_CONSTANT_CGS,
        1.0,
    )


class CIECoolingNetwork(ThermochemistryNetwork):
    name = "cie_cooling"
    scalar_fields = ()

    def enabled(self, fluid, par):
        return bool(getattr(par, "cie_cooling", False))

    def radiation_enabled(self, fluid, par):
        return False

    def radiation_evolution_enabled(self, fluid, par):
        return False

    def advect_ionization_fraction(self, *args, **kwargs):
        return None

    def source_state(self, mesh, fluid, par):
        state = _state(mesh, fluid, par)
        state.update(
            par=par,
            metallicity=float(getattr(par, "metallicity", 1.0)),
            hydrogen_mass_fraction=float(getattr(par, "hydrogen_mass_fraction", 1.0)),
            cooling_safety_factor=float(getattr(par, "cooling_safety_factor", 0.1)),
            compton_cmb_enabled=bool(getattr(par, "compton_cmb_enabled", False)),
            compton_cmb_redshift=float(getattr(par, "compton_cmb_redshift", 0.0)),
            cmb_temperature_0_K=float(to_unit_value(getattr(par, "cmb_temperature_0", 2.7255), unyt.K)),
        )
        return state

    def ionization_fraction_rate(self, state, ngamma):
        return np.zeros_like(state["temperature_K"])

    def thermal_rate(self, state, ngamma):
        table = _get_table(state["par"])
        metallicity = state["metallicity"]
        nH = state["rho_g_cm3"] * state["hydrogen_mass_fraction"] / PROTON_MASS_CGS
        ne = nH * table.electron_fraction(state["temperature_K"], metallicity)
        Lambda = table.cooling_coefficient(state["temperature_K"], ne, metallicity)
        return -ne * nH * Lambda + cmb_compton_rate(
            state["temperature_K"],
            ne,
            enabled=state.get("compton_cmb_enabled", False),
            redshift=state.get("compton_cmb_redshift", 0.0),
            cmb_temperature_0_K=state.get("cmb_temperature_0_K", 2.7255),
        )

    def get_timestep(self, state, ngamma, remaining_s, dtmax_s):
        rate = self.thermal_rate(state, ngamma)
        thermal_density = state["specific_energy_erg_g"] * state["rho_g_cm3"]
        cooling_time = np.divide(
            thermal_density,
            np.maximum(np.abs(rate), 1.0e-99),
        )
        safety = float(state["cooling_safety_factor"])
        candidate = np.min(safety * cooling_time)
        return min(float(remaining_s), float(dtmax_s), candidate), rate

    def update_temperature_from_energy(self, state):
        _update_temperature(state)

    def ionization_fraction_implicit_update(self, state, ngamma, dt_s):
        return None

    def apply_state(self, state, fluid, par):
        return None

    def get_source_timestep_fast(self, mesh, fluid, par, remaining):
        state = _state(mesh, fluid, par)
        state.update(
            par=par,
            metallicity=float(getattr(par, "metallicity", 1.0)),
            hydrogen_mass_fraction=float(getattr(par, "hydrogen_mass_fraction", 1.0)),
            cooling_safety_factor=float(getattr(par, "cooling_safety_factor", 0.1)),
            compton_cmb_enabled=bool(getattr(par, "compton_cmb_enabled", False)),
            compton_cmb_redshift=float(getattr(par, "compton_cmb_redshift", 0.0)),
            cmb_temperature_0_K=float(to_unit_value(getattr(par, "cmb_temperature_0", 2.7255), unyt.K)),
        )
        code = state["code"]
        remaining_s = (
            to_unit_value(remaining, code.time_unit)
            * state["source_scale_factor"]**2
        )
        dt_s, rate = self.get_timestep(state, None, remaining_s, remaining_s)
        return from_unit_value(
            dt_s / state["source_scale_factor"]**2,
            code.time_unit,
        ), rate

    def apply_fast(self, dt, mesh, fluid, par):
        state = _state(mesh, fluid, par)
        state.update(
            par=par,
            metallicity=float(getattr(par, "metallicity", 1.0)),
            hydrogen_mass_fraction=float(getattr(par, "hydrogen_mass_fraction", 1.0)),
            cooling_safety_factor=float(getattr(par, "cooling_safety_factor", 0.1)),
            compton_cmb_enabled=bool(getattr(par, "compton_cmb_enabled", False)),
            compton_cmb_redshift=float(getattr(par, "compton_cmb_redshift", 0.0)),
            cmb_temperature_0_K=float(to_unit_value(getattr(par, "cmb_temperature_0", 2.7255), unyt.K)),
        )
        code = state["code"]
        remaining_s = (
            float(to_unit_value(dt, code.time_unit))
            * state["source_scale_factor"]**2
        )
        source_steps = 0
        floor = getattr(par, "cooling_temperature_floor", 1.0)
        floor_K = float(to_unit_value(floor, unyt.K))
        while remaining_s > 0.0:
            _update_temperature(state)
            dt_s, rate = self.get_timestep(state, None, remaining_s, remaining_s)
            if not np.isfinite(dt_s) or dt_s <= 0.0:
                dt_s = remaining_s
            dt_s = min(dt_s, remaining_s)
            state["specific_energy_erg_g"] += (
                rate / np.maximum(state["rho_g_cm3"], 1.0e-99) * dt_s
            )
            minimum_energy = (
                BOLTZMANN_CONSTANT_CGS * floor_K
                / ((state["gamma"] - 1.0) * state["mu"] * PROTON_MASS_CGS)
            )
            state["specific_energy_erg_g"] = np.maximum(
                state["specific_energy_erg_g"], minimum_energy
            )
            remaining_s -= dt_s
            source_steps += 1

        _update_temperature(state)
        interior = state["interior"]
        internal_super = state["specific_energy_erg_g"] * state["source_temperature_factor"]
        total_super = internal_super + 0.5 * state["velocity_supercomoving_cm_s"]**2
        fluid.Energy[interior] = from_unit_value(
            state["mass_g"] * total_super,
            code.energy_unit,
        )
        fluid.temp[interior] = from_unit_value(
            state["temperature_K"] * state["source_temperature_factor"],
            code.temperature_unit,
        )
        if hasattr(fluid.eos, "pressure"):
            fluid.pre[interior] = fluid.eos.pressure(
                fluid.rho[interior],
                fluid.temp[interior],
                fluid.mu[interior],
            )
        else:
            # Lightweight test doubles may only expose gamma.
            internal_code = internal_super / code.velocity_in_cgs**2
            fluid.pre[interior] = (
                (state["gamma"] - 1.0)
                * np.asarray(fluid.rho[interior], dtype=float)
                * internal_code
            )
        return source_steps
