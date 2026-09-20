# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Radiation source orchestration for the hydrodynamics solver."""  # noqa: CPY001

import numpy as np

from radhydropy.constants import SPEED_OF_LIGHT_CGS
from radhydropy.units import _code_units, code_unit_scales


def apply_radiation_pressure(solver, dt, mesh, fluid, par, source_result):
    """Apply photon momentum deposition to the conserved gas state."""
    if not getattr(par, "radiation_pressure", False):
        return 0
    if not source_result or source_result.get("absorbed_photon_rate") is None:
        return 0

    code_units = _code_units(par)
    scales = code_unit_scales(code_units)
    interior = solver._interior_slice(par)
    absorbed = np.asarray(source_result["absorbed_photon_rate"], dtype=float)
    energies = np.asarray(source_result["photon_energy_cgs_erg"], dtype=float)
    if absorbed.ndim == 1:
        absorbed = absorbed[None, :]
    if energies.ndim == 0:
        energies = energies[None]
    if absorbed.shape[0] != energies.size:
        raise ValueError("absorbed photon groups and photon energies disagree")
    grid_cells = int(par.mesh.grid_cells)
    if absorbed.shape[1] != grid_cells:
        raise ValueError("absorbed photon rate must contain physical cells only")

    density_runtime_code, velocity_runtime_code, _, _ = solver._active_primitive_arrays(fluid, par)
    rho_cgs = np.asarray(density_runtime_code[interior], dtype=float) * scales["density_cgs_g_cm3"]
    momentum_rate_density = (
        float(source_result.get("direction", 1))
        * np.sum(absorbed * energies[:, None], axis=0)
        / SPEED_OF_LIGHT_CGS
    )
    efficiency = float(getattr(par, "radiation_pressure_efficiency", 1.0))
    valid = rho_cgs > 0.0
    if not np.any(valid):
        return 0
    acceleration_cgs = np.zeros_like(momentum_rate_density)
    acceleration_cgs[valid] = efficiency * momentum_rate_density[valid] / rho_cgs[valid]
    acceleration = acceleration_cgs / scales["acceleration_cgs_cm_s2"]
    volume = np.asarray(
        solver._geometry_state(mesh, par).volume_runtime_code[interior],
        dtype=float,
    )
    momentum = fluid.Mom_code[interior]
    energy = fluid.Energy_code[interior]
    density_runtime_code = density_runtime_code[interior]
    velocity = velocity_runtime_code[interior]
    momentum[valid] += density_runtime_code[valid] * acceleration[valid] * volume[valid] * dt
    energy[valid] += (
        density_runtime_code[valid] * velocity[valid] * acceleration[valid] * volume[valid] * dt
    )
    return 1
