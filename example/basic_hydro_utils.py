"""Shared canonical IC helpers for the basic hydro examples."""

import numpy as np

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS


def make_initial_condition(
    config,
    *,
    boundary_proper_code,
    rho_proper_code,
    vel_proper_code,
    temp_proper_code,
    mu_dimensionless,
    area_proper_code=None,
):
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
                "custom area must contain one proper-code value per physical cell"
            )
        ghost_cells = int(sim.par.mesh.ghost_cells)
        area_with_ghosts = np.concatenate(
            (
                np.full(ghost_cells, area_proper_code[0]),
                area_proper_code,
                np.full(ghost_cells, area_proper_code[-1]),
            )
        )
        sim.mesh.area_proper_code = as_named_array(area_with_ghosts)
        sim.mesh.volume_proper_code = as_named_array(
            area_with_ghosts * np.asarray(sim.mesh.width_proper_code, dtype=float)
        )
        sim.mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            coordinate=sim.mesh.x_proper_code,
            boundary=sim.mesh.boundary_proper_code,
            width=sim.mesh.width_proper_code,
            area=sim.mesh.area_proper_code,
            volume=sim.mesh.volume_proper_code,
        )
    sim.fluid.SetUpFluid(sim.par, sim.mesh)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    first = int(sim.par.mesh.ghost_cells)
    last = first + grid_cells
    sim.mesh.boundary_proper_code = as_named_array(sim.mesh.boundary_proper_code[first:last + 1])
    for field in (
        "rho_proper_code", "vel_proper_code", "temp_proper_code", "mu",
        "Mass_code", "Mom_code", "AngularMomentum_code", "Energy_code",
        "InternalEnergy_code",
    ):
        if hasattr(sim.fluid, field):
            setattr(sim.fluid, field, as_named_array(getattr(sim.fluid, field)[first:last]))
    sim.par.mesh.ghost_cells = 0
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=sim.mesh.x_proper_code[first:last],
        boundary=sim.mesh.boundary_proper_code,
        width=sim.mesh.width_proper_code[first:last],
        area=sim.mesh.area_proper_code[first:last],
        volume=sim.mesh.volume_proper_code[first:last],
    )
    sim.fluid._refresh_runtime_state()
    return Rsim.FromComponents(sim.par, sim.mesh, sim.fluid, sim.solver)


def finalize_initial_condition(sim, grid_cells, extra_fields=()):
    """Remove setup ghosts before serializing a canonical proper-code IC."""
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(grid_cells)
    sim.mesh.boundary_proper_code = as_named_array(
        sim.mesh.boundary_proper_code[first:last + 1]
    )
    fields = (
        "rho_proper_code", "vel_proper_code", "temp_proper_code", "pre_proper_code", "mu",
        "Mass_code", "Mom_code", "AngularMomentum_code", "Energy_code",
        "InternalEnergy_code", *extra_fields,
    )
    for field in fields:
        if hasattr(sim.fluid, field):
            values = getattr(sim.fluid, field)
            if values is not None and np.ndim(values) >= 1 and len(values) >= last:
                setattr(sim.fluid, field, as_named_array(values[first:last]))
    sim.par.mesh.ghost_cells = 0
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=sim.mesh.x_proper_code[first:last],
        boundary=sim.mesh.boundary_proper_code,
        width=sim.mesh.width_proper_code[first:last],
        area=sim.mesh.area_proper_code[first:last],
        volume=sim.mesh.volume_proper_code[first:last],
    )
    sim.fluid._refresh_runtime_state()
    return sim


def physical_snapshot(config, filename):
    sim = Rsim(config["par"])
    import radhydropy.io as rio
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    boundary = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    return sim, boundary[first:last + 1], slice(first, last)
