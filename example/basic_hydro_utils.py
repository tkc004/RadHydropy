"""Shared canonical IC helpers for the basic hydro examples."""

import numpy as np

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS


def make_initial_condition(config, boundary, rho, velocity, temperature, mu, area=None):
    """Build a physical-cell IC through the current runtime startup contract."""
    sim = Rsim(config["par"])
    grid_cells = int(config["initial_condition"].get(
        "grid_cells", config["par"]["mesh"]["grid_cells"]
    ))
    sim.par.mesh.grid_cells = grid_cells
    boundary = np.asarray(boundary, dtype=float)
    sim.mesh.boundary_proper_code = as_named_array(boundary)
    sim.fluid.rho_proper_code = as_named_array(np.asarray(rho, dtype=float))
    sim.fluid.vel_proper_code = as_named_array(np.asarray(velocity, dtype=float))
    sim.fluid.temp_proper_code = as_named_array(np.asarray(temperature, dtype=float))
    sim.fluid.mu = as_named_array(np.asarray(mu, dtype=float))
    sim.SetMesh()
    if area is not None:
        area_proper_code = np.asarray(area, dtype=float)
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
    return sim


def physical_snapshot(config, filename):
    sim = Rsim(config["par"])
    import radhydropy.io as rio
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    boundary = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    return sim, boundary[first:last + 1], slice(first, last)
