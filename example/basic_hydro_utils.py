# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Shared canonical IC helpers for the basic hydro examples."""

from typing import Any, cast

import numpy as np

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import PROPER_RUNTIME_FIELDS, MeshGeometryState


def _validate_active_proper_state(
    sim: Any,
    first: Any,
    last: Any,
    *,
    allow_vacuum: Any = False,
) -> Any:
    """Validate the active proper-code primitive and conserved state."""
    rho_proper_code = np.asarray(sim.fluid.rho_proper_code[first:last], dtype=float)
    vel_proper_code = np.asarray(sim.fluid.vel_proper_code[first:last], dtype=float)
    pre_proper_code = np.asarray(sim.fluid.pre_proper_code[first:last], dtype=float)
    temp_proper_code = np.asarray(sim.fluid.temp_proper_code[first:last], dtype=float)
    mu_dimensionless = np.asarray(sim.fluid.mu[first:last], dtype=float)
    volume_proper_code = np.asarray(sim.mesh.volume_proper_code[first:last], dtype=float)

    _validate_primitives(
        sim,
        rho_proper_code,
        vel_proper_code,
        pre_proper_code,
        temp_proper_code,
        mu_dimensionless,
        volume_proper_code,
        allow_vacuum,
    )

    expected_mass_code = rho_proper_code * volume_proper_code
    expected_mom_code = expected_mass_code * vel_proper_code
    expected_energy_code = (
        sim.fluid.eos.total_energy_density(
            rho_proper_code,
            vel_proper_code,
            pre_proper_code,
        )
        * volume_proper_code
    )
    mass_code = np.asarray(sim.fluid.Mass_code[first:last], dtype=float)
    mom_code = np.asarray(sim.fluid.Mom_code[first:last], dtype=float)
    energy_code = np.asarray(sim.fluid.Energy_code[first:last], dtype=float)
    _validate_conserved(
        mass_code,
        mom_code,
        energy_code,
        expected_mass_code,
        expected_mom_code,
        expected_energy_code,
    )


def _validate_finite(fields: Any) -> Any:
    for field_name, field_values in fields:
        if not np.all(np.isfinite(field_values)):
            raise ValueError(f"active {field_name} contains non-finite values")


def _validate_primitives(
    sim: Any,
    rho_proper_code: Any,
    vel_proper_code: Any,
    pre_proper_code: Any,
    temp_proper_code: Any,
    mu_dimensionless: Any,
    volume_proper_code: Any,
    allow_vacuum: Any,
) -> Any:
    _validate_finite(
        (
            ("rho_proper_code", rho_proper_code),
            ("vel_proper_code", vel_proper_code),
            ("pre_proper_code", pre_proper_code),
            ("temp_proper_code", temp_proper_code),
            ("mu_dimensionless", mu_dimensionless),
            ("volume_proper_code", volume_proper_code),
        ),
    )
    density_limit = rho_proper_code < 0.0 if allow_vacuum else rho_proper_code <= 0.0
    density_message = (
        "active rho_proper_code must be non-negative"
        if allow_vacuum
        else "active rho_proper_code must be strictly positive"
    )
    for _values, limit, message in (
        (rho_proper_code, density_limit, density_message),
        (temp_proper_code, temp_proper_code < 0.0, "active temp_proper_code must be non-negative"),
        (pre_proper_code, pre_proper_code < 0.0, "active pre_proper_code must be non-negative"),
        (
            volume_proper_code,
            volume_proper_code <= 0.0,
            "active volume_proper_code must be strictly positive",
        ),
    ):
        if np.any(limit):
            raise ValueError(message)
    expected_pre_proper_code = np.asarray(
        sim.fluid.eos.pressure(rho_proper_code, temp_proper_code, mu_dimensionless),
        dtype=float,
    )
    if not np.allclose(
        pre_proper_code,
        expected_pre_proper_code,
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise ValueError("active proper-code pressure is inconsistent with rho/temp/mu")


def _validate_conserved(
    mass_code: Any,
    momentum_code: Any,
    energy_code: Any,
    expected_mass_code: Any,
    expected_momentum_code: Any,
    expected_energy_code: Any,
) -> Any:
    _validate_finite(
        (("Mass_code", mass_code), ("Mom_code", momentum_code), ("Energy_code", energy_code)),
    )
    for actual, expected, message in (
        (mass_code, expected_mass_code, "active Mass_code is inconsistent with rho/volume"),
        (
            momentum_code,
            expected_momentum_code,
            "active Mom_code is inconsistent with rho/vel/volume",
        ),
        (
            energy_code,
            expected_energy_code,
            "active Energy_code is inconsistent with rho/vel/pre/volume",
        ),
    ):
        if not np.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError(message)


def make_initial_condition(
    config: Any,
    *,
    boundary_proper_code: Any,
    rho_proper_code: Any,
    vel_proper_code: Any,
    temp_proper_code: Any,
    mu_dimensionless: Any,
    area_proper_code: Any = None,
    allow_vacuum: Any = False,
) -> Any:
    """Build a proper-code IC from a complete nested example configuration.

    The profile arrays are already in the configured proper-code system.  The
    helper owns ghost-cell setup and returns a runner built from the same
    components that are serialized to HDF5.
    """
    sim = Rsim(config["par"])
    grid_cells = int(config["par"]["mesh"]["grid_cells"])
    sim.par.mesh.grid_cells = grid_cells
    boundary_proper_code = np.asarray(boundary_proper_code, dtype=float)
    rho_proper_code = np.asarray(rho_proper_code, dtype=float)
    vel_proper_code = np.asarray(vel_proper_code, dtype=float)
    temp_proper_code = np.asarray(temp_proper_code, dtype=float)
    mu_dimensionless = np.asarray(mu_dimensionless, dtype=float)
    if boundary_proper_code.size != grid_cells + 1:
        raise ValueError("proper-code boundary must contain grid_cells + 1 values")
    for field_name, field_values in (
        ("rho_proper_code", rho_proper_code),
        ("vel_proper_code", vel_proper_code),
        ("temp_proper_code", temp_proper_code),
        ("mu_dimensionless", mu_dimensionless),
    ):
        if field_values.size != grid_cells:
            raise ValueError(f"{field_name} must contain one value per physical cell")
        if not np.all(np.isfinite(field_values)):
            raise ValueError(f"{field_name} contains non-finite values")
    sim.mesh.boundary_proper_code = as_named_array(boundary_proper_code)
    sim.fluid.rho_proper_code = as_named_array(rho_proper_code)
    sim.fluid.vel_proper_code = as_named_array(vel_proper_code)
    sim.fluid.temp_proper_code = as_named_array(temp_proper_code)
    sim.fluid.mu = as_named_array(mu_dimensionless)
    sim.SetMesh()
    if area_proper_code is not None:
        area_proper_code = np.asarray(area_proper_code, dtype=float)
        if area_proper_code.size != grid_cells:
            raise ValueError(
                "custom area must contain one proper-code value per physical cell",
            )
        ghost_cells = int(sim.par.mesh.ghost_cells)
        area_with_ghosts = np.concatenate(
            (
                np.full(ghost_cells, area_proper_code[0]),
                area_proper_code,
                np.full(ghost_cells, area_proper_code[-1]),
            ),
        )
        sim.mesh.area_proper_code = as_named_array(area_with_ghosts)
        sim.mesh.volume_proper_code = as_named_array(
            area_with_ghosts * np.asarray(sim.mesh.width_proper_code, dtype=float),
        )
        sim.mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            x_proper_code=sim.mesh.x_proper_code,
            boundary_proper_code=sim.mesh.boundary_proper_code,
            width_proper_code=sim.mesh.width_proper_code,
            area_proper_code=sim.mesh.area_proper_code,
            volume_proper_code=sim.mesh.volume_proper_code,
        )
    sim.fluid.SetUpFluid(sim.par, sim.mesh)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    first = int(sim.par.mesh.ghost_cells)
    last = first + grid_cells
    _validate_active_proper_state(
        sim,
        first,
        last,
        allow_vacuum=allow_vacuum,
    )
    sim.mesh.boundary_proper_code = as_named_array(sim.mesh.boundary_proper_code[first : last + 1])
    for field in (
        "rho_proper_code",
        "vel_proper_code",
        "pre_proper_code",
        "temp_proper_code",
        "mu",
        "Mass_code",
        "Mom_code",
        "AngularMomentum_code",
        "Energy_code",
        "InternalEnergy_code",
    ):
        if hasattr(sim.fluid, field):
            setattr(sim.fluid, field, as_named_array(getattr(sim.fluid, field)[first:last]))
    sim.par.mesh.ghost_cells = 0
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=sim.mesh.x_proper_code[first:last],
        boundary_proper_code=sim.mesh.boundary_proper_code,
        width_proper_code=sim.mesh.width_proper_code[first:last],
        area_proper_code=sim.mesh.area_proper_code[first:last],
        volume_proper_code=sim.mesh.volume_proper_code[first:last],
    )
    sim.fluid.refresh_runtime_state()
    return Rsim.FromComponents(sim.par, sim.mesh, sim.fluid, sim.solver)


def finalize_initial_condition(sim: Any, grid_cells: Any, extra_fields: Any = ()) -> Any:
    """Remove setup ghosts before serializing a canonical proper-code IC."""
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(grid_cells)
    sim.mesh.boundary_proper_code = as_named_array(
        sim.mesh.boundary_proper_code[first : last + 1],
    )
    fields = (
        "rho_proper_code",
        "vel_proper_code",
        "temp_proper_code",
        "pre_proper_code",
        "mu",
        "Mass_code",
        "Mom_code",
        "AngularMomentum_code",
        "Energy_code",
        "InternalEnergy_code",
        *extra_fields,
    )
    for field in fields:
        if hasattr(sim.fluid, field):
            values = getattr(sim.fluid, field)
            if values is not None and np.ndim(values) >= 1 and len(values) >= last:
                setattr(sim.fluid, field, as_named_array(values[first:last]))
    sim.par.mesh.ghost_cells = 0
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=sim.mesh.x_proper_code[first:last],
        boundary_proper_code=sim.mesh.boundary_proper_code,
        width_proper_code=sim.mesh.width_proper_code[first:last],
        area_proper_code=sim.mesh.area_proper_code[first:last],
        volume_proper_code=sim.mesh.volume_proper_code[first:last],
    )
    sim.fluid.refresh_runtime_state()
    return sim


def physical_snapshot(config: Any, filename: Any) -> Any:
    sim = Rsim(config["par"])
    import radhydropy.io as rio  # noqa: PLC0415

    rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(cast("Any", sim.par.mesh.grid_cells))
    boundary_proper_code = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    return sim, boundary_proper_code[first : last + 1], slice(first, last)
