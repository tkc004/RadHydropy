# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Initial conditions and diagnostics for high-Mach advection."""

import numpy as np
from basic_hydro_utils import make_initial_condition

from radhydropy.eos import EOS


def build_initial_condition(config):
    initial, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n = int(initial["grid_cells"])
    size_proper_code = initial["box_size_proper"].to(units.length_unit).value
    boundary_proper_code = np.linspace(0.0, size_proper_code, n + 1)
    x_proper_code = 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    left = x_proper_code < 0.5 * size_proper_code
    rho_default = initial.get("rho_proper")
    rho_left_proper_code = (
        initial.get("rho_left_proper", rho_default)
        .to(
            units.density_unit,
        )
        .value
    )
    rho_right_proper_code = (
        initial.get("rho_right_proper", rho_default)
        .to(
            units.density_unit,
        )
        .value
    )
    rho_proper_code = np.where(left, rho_left_proper_code, rho_right_proper_code)
    vel_proper_code = np.full(
        n,
        initial["vel_proper"].to(units.velocity_unit).value,
    )
    mu = np.full(n, initial["mean_molecular_weight"])
    if "temperature_left_proper" in initial or "temperature_right_proper" in initial:
        temperature_default = initial.get("temperature_proper")
        temperature_left_proper_code = (
            initial.get(
                "temperature_left_proper",
                temperature_default,
            )
            .to(units.temperature_unit)
            .value
        )
        temperature_right_proper_code = (
            initial.get(
                "temperature_right_proper",
                temperature_default,
            )
            .to(units.temperature_unit)
            .value
        )
        temp_proper_code = np.where(
            left,
            temperature_left_proper_code,
            temperature_right_proper_code,
        )
    elif "pressure_initial_proper" in initial:
        pressure_proper_code = (
            initial["pressure_initial_proper"]
            .to(
                units.pressure_unit,
            )
            .value
        )
        temp_proper_code = np.full(n, pressure_proper_code / rho_left_proper_code)
    else:
        temp_proper_code = np.full(
            n,
            initial["temperature_proper"].to(units.temperature_unit).value,
        )
    area_proper_code = (
        np.ones(n)
        * par["mesh"]["area_proper"]
        .to(
            units.area_unit,
        )
        .value
    )
    return make_initial_condition(
        config,
        boundary_proper_code=boundary_proper_code,
        rho_proper_code=rho_proper_code,
        vel_proper_code=vel_proper_code,
        temp_proper_code=temp_proper_code,
        mu_dimensionless=mu,
        area_proper_code=area_proper_code,
        allow_vacuum=True,
    )


def _physical(state):
    first = int(state.par.mesh.ghost_cells)
    last = first + int(state.par.mesh.grid_cells)
    b = np.asarray(state.mesh.boundary_proper_code)
    return first, last, 0.5 * (b[:-1] + b[1:])


def energy_components(state):
    first, last, _ = _physical(state)
    rho_proper_code = np.asarray(state.fluid.rho_proper_code)
    vel_proper_code = np.asarray(state.fluid.vel_proper_code)
    temp_proper_code = np.asarray(state.fluid.temp_proper_code)
    mu = np.asarray(state.fluid.mu)
    eos = state.fluid.eos or EOS(
        "polytropic",
        float(state.par.hydrodynamics.gamma),
        state.par.units.CodeUnits,
    )
    pressure_proper_code = np.asarray(eos.pressure(rho_proper_code, temp_proper_code, mu))
    volume_proper_code = np.asarray(state.mesh.volume_proper_code)
    kinetic_energy_proper_code = 0.5 * rho_proper_code * vel_proper_code**2 * volume_proper_code
    thermal_energy_proper_code = pressure_proper_code / (eos.gamma - 1) * volume_proper_code
    return {
        "total_energy_proper_code": float(
            np.sum((kinetic_energy_proper_code + thermal_energy_proper_code)[first:last]),
        ),
        "kinetic_energy_proper_code": float(np.sum(kinetic_energy_proper_code[first:last])),
        "thermal_energy_proper_code": float(np.sum(thermal_energy_proper_code[first:last])),
    }


def entropy_profile(state):
    first, last, x_proper_code = _physical(state)
    rho_proper_code = np.asarray(state.fluid.rho_proper_code)
    temp_proper_code = np.asarray(state.fluid.temp_proper_code)
    gamma = float(state.par.hydrodynamics.gamma)
    return x_proper_code[first:last], temp_proper_code[first:last] / rho_proper_code[
        first:last
    ] ** (gamma - 1)


def primitive_profiles(state):
    first, last, x_proper_code = _physical(state)
    return (
        x_proper_code[first:last],
        np.asarray(state.fluid.rho_proper_code)[first:last],
        np.asarray(state.fluid.temp_proper_code)[first:last],
    )
