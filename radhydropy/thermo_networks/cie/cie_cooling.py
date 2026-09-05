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
from radhydropy.thermo_networks.hydrogen import (
    _fast_source_scaling,
    _rotational_specific_energy_code,
)
from radhydropy.diagnostics import thermochemistry_active_mask
from radhydropy.state_boundaries import cgs_source_state_from_code


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
        raise ValueError("CIE cooling requires configured code units")
    ghost_cells = int(par.mesh.ghost_cells)
    grid_cells = int(par.mesh.grid_cells)
    interior = slice(ghost_cells, ghost_cells + grid_cells)
    gamma = getattr(getattr(fluid, "eos", None), "gamma", 5.0 / 3.0)
    scaling = _fast_source_scaling(fluid, par, gamma)
    runtime = fluid.code_state
    primitive_cgs = cgs_source_state_from_code(
        code_units=code,
        fluid=runtime,
        boundary_code=mesh.boundary[interior.start : interior.stop + 1],
        volume_code=mesh.vol[interior],
    )
    rho_super = primitive_cgs.rho_cgs_g_cm3[interior]
    rho = rho_super / scaling["density_factor"]
    velocity_super = primitive_cgs.velocity_cgs_cm_s[interior]
    velocity = velocity_super / scaling["velocity_factor"]
    volume_code = np.asarray(mesh.vol[interior], dtype=float)
    volume = primitive_cgs.volume_cgs_cm3 * scaling["density_factor"]
    if runtime.Mass_code is not None:
        mass = runtime.Mass_code[interior] * code.unit_conversion["mass_g"]
    else:
        mass = rho_super * volume_code * code.mass_in_cgs
    specific_total_super = primitive_cgs.specific_energy_cgs_erg_g[interior]
    rotational_specific_code = _rotational_specific_energy_code(mesh, fluid, par)
    rotational_specific_super = (
        rotational_specific_code * code.unit_conversion['velocity_cgs_cm_s']**2
    )
    specific_internal = np.maximum(
        specific_total_super - 0.5 * velocity_super**2
        - rotational_specific_super,
        0.0,
    ) / scaling["temperature_factor"]
    thermal_energy = specific_internal * mass
    mu = np.asarray(fluid.mu[interior], dtype=float)
    return {
        "interior": interior,
        "rho_cgs_g_cm3": rho,
        "active": thermochemistry_active_mask(
            rho, par, scaling["density_factor"]
        ),
        "volume_cgs_cm3": volume,
        "velocity_cgs_cm_s": velocity,
        "thermal_energy_cgs_erg": thermal_energy,
        "specific_energy_cgs_erg_g": specific_internal,
        "specific_rotational_energy_cgs_erg_g": (
            rotational_specific_super / scaling["temperature_factor"]
        ),
        "temperature_cgs_K": (
            primitive_cgs.temperature_cgs_K[interior]
            / scaling["temperature_factor"]
        ),
        "gamma": gamma,
        "mu": mu,
        "code": code,
        "mass_g": mass,
        "velocity_supercomoving_cgs_cm_s": velocity_super,
        "source_scale_factor": scaling["scale_factor"],
        "source_temperature_factor": scaling["temperature_factor"],
    }


def _update_temperature(state):
    state["temperature_cgs_K"] = np.maximum(
        state["specific_energy_cgs_erg_g"]
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
            cmb_temperature_0_cgs_K=float(to_unit_value(getattr(par, "cmb_temperature_0", 2.7255), unyt.K)),
        )
        return state

    def ionization_fraction_rate(self, state, ngamma_cgs_cm3):
        return np.zeros_like(state["temperature_cgs_K"])

    def thermal_rate(self, state, ngamma_cgs_cm3):
        table = _get_table(state["par"])
        metallicity = state["metallicity"]
        nH = state["rho_cgs_g_cm3"] * state["hydrogen_mass_fraction"] / PROTON_MASS_CGS
        ne = nH * table.electron_fraction(state["temperature_cgs_K"], metallicity)
        Lambda = table.cooling_coefficient(state["temperature_cgs_K"], ne, metallicity)
        return -ne * nH * Lambda + cmb_compton_rate(
            state["temperature_cgs_K"],
            ne,
            enabled=state.get("compton_cmb_enabled", False),
            redshift=state.get("compton_cmb_redshift", 0.0),
            cmb_temperature_0_cgs_K=state.get("cmb_temperature_0_cgs_K", 2.7255),
        )

    def get_timestep(self, state, ngamma_cgs_cm3, remaining_s, dtmax_s):
        rate = self.thermal_rate(state, ngamma_cgs_cm3)
        thermal_density = state["specific_energy_cgs_erg_g"] * state["rho_cgs_g_cm3"]
        cooling_time = np.divide(
            thermal_density,
            np.maximum(np.abs(rate), 1.0e-99),
        )
        cooling_time = np.where(state["active"], cooling_time, np.inf)
        safety = float(state["cooling_safety_factor"])
        candidate = np.min(safety * cooling_time)
        return min(float(remaining_s), float(dtmax_s), candidate), rate

    def update_temperature_from_energy(self, state):
        _update_temperature(state)

    def ionization_fraction_implicit_update(self, state, ngamma_cgs_cm3, dt_s):
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
            cmb_temperature_0_cgs_K=float(to_unit_value(getattr(par, "cmb_temperature_0", 2.7255), unyt.K)),
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
            cmb_temperature_0_cgs_K=float(to_unit_value(getattr(par, "cmb_temperature_0", 2.7255), unyt.K)),
        )
        code = state["code"]
        remaining_s = (
            float(to_unit_value(dt, code.time_unit))
            * state["source_scale_factor"]**2
        )
        source_steps = 0
        active = np.asarray(state["active"], dtype=bool)
        floor = getattr(par, "cooling_temperature_floor", 1.0)
        floor_cgs_K = float(to_unit_value(floor, unyt.K))
        while remaining_s > 0.0:
            _update_temperature(state)
            dt_s, rate = self.get_timestep(state, None, remaining_s, remaining_s)
            if not np.isfinite(dt_s) or dt_s <= 0.0:
                dt_s = remaining_s
            dt_s = min(dt_s, remaining_s)
            energy = np.asarray(state["specific_energy_cgs_erg_g"], dtype=float).copy()
            energy[active] += (
                rate[active] / np.maximum(state["rho_cgs_g_cm3"][active], 1.0e-99)
                * dt_s
            )
            state["specific_energy_cgs_erg_g"] = energy
            minimum_energy = (
                BOLTZMANN_CONSTANT_CGS * floor_cgs_K
                / ((state["gamma"] - 1.0) * state["mu"] * PROTON_MASS_CGS)
            )
            state["specific_energy_cgs_erg_g"][active] = np.maximum(
                state["specific_energy_cgs_erg_g"][active], minimum_energy[active]
            )
            remaining_s -= dt_s
            source_steps += 1

        _update_temperature(state)
        interior = state["interior"]
        internal_super = state["specific_energy_cgs_erg_g"] * state["source_temperature_factor"]
        total_super = internal_super + 0.5 * state["velocity_supercomoving_cgs_cm_s"]**2
        updated_energy = from_unit_value(
            state["mass_g"] * total_super, code.energy_unit
        )
        rotational_code = _rotational_specific_energy_code(mesh, fluid, par)
        mass_code = (
            np.asarray(fluid.Mass_code[interior], dtype=float)
            if hasattr(fluid, "Mass_code") else
            np.asarray(fluid.rho_code[interior], dtype=float)
            * np.asarray(mesh.vol[interior], dtype=float)
        )
        updated_energy = updated_energy + from_unit_value(
            mass_code * rotational_code,
            code.energy_unit,
        )
        updated_temperature = from_unit_value(
            state["temperature_cgs_K"] * state["source_temperature_factor"],
            code.temperature_unit,
        )
        energy_target = fluid.Energy_code[interior].copy()
        temperature_target = fluid.temp_code[interior].copy()
        energy_target[active] = updated_energy[active]
        temperature_target[active] = updated_temperature[active]
        fluid.Energy_code[interior] = energy_target
        fluid.temp_code[interior] = temperature_target
        if hasattr(fluid.eos, "pressure"):
            fluid.pre_code[interior] = fluid.eos.pressure(
                fluid.rho_code[interior],
                fluid.temp_code[interior],
                fluid.mu[interior],
            )
        else:
            # Lightweight test doubles may only expose gamma.
            internal_code = internal_super / code.velocity_in_cgs**2
            fluid.pre_code[interior] = (
                (state["gamma"] - 1.0)
                * np.asarray(fluid.rho_code[interior], dtype=float)
                * internal_code
            )
        return source_steps
