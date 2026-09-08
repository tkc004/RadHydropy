"""Initial condition and plotting helpers for outflow into vacuum."""

import numpy as np
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value


def analytic_density_profile(radius, time_proper_code, config, cell_faces=None):
    """Cold spherical outflow profile, sampled as cell averages when given."""
    radius = np.asarray(radius, dtype=float)
    initial = config['initial_condition']
    boundary = config['par']['boundary']
    injection_radius = float(initial['radius_injection_proper'])
    density_outflow = float(boundary['rho_outflow_proper'])
    velocity_outflow = float(boundary['vel_outflow_proper'])
    front = injection_radius + velocity_outflow * float(time_proper_code)
    profile = np.full_like(radius, np.nan, dtype=float)
    if cell_faces is None:
        inside = (radius >= injection_radius) & (radius <= front)
        profile[inside] = density_outflow * (injection_radius / radius[inside])**2
        return profile, front
    faces = np.asarray(cell_faces, dtype=float)
    left = np.maximum(faces[:-1], injection_radius)
    right = np.minimum(faces[1:], front)
    inside = right > left
    volume_factor = faces[1:]**3 - faces[:-1]**3
    profile[inside] = (
        3.0 * density_outflow * injection_radius**2
        * (right[inside] - left[inside]) / volume_factor[inside]
    )
    return profile, front


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    sim = Rsim(config['par'])
    grid_cells = int(initial['grid_cells'])
    sim.par.mesh.grid_cells = grid_cells
    injection_radius_code = quantity_to_value(initial['radius_injection_proper'], code_units.length_unit)
    box_size_code = quantity_to_value(initial['box_size_proper'], code_units.length_unit)
    sim.mesh.boundary_proper_code = as_named_array(np.linspace(
        injection_radius_code, injection_radius_code + box_size_code, grid_cells + 1
    ))
    sim.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells))
    sim.fluid.temp_proper_code = as_named_array(np.zeros(grid_cells))
    sim.fluid.rho_proper_code = as_named_array(np.zeros(grid_cells))
    sim.fluid.mu = as_named_array(np.full(grid_cells, initial['mean_molecular_weight']))
    sim.SetMesh()
    sim.fluid.SetUpFluid(sim.par, sim.mesh)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    first = int(sim.par.mesh.ghost_cells)
    last = first + grid_cells
    sim.mesh.boundary_proper_code = as_named_array(sim.mesh.boundary_proper_code[first:last + 1])
    for field in ('rho_proper_code', 'vel_proper_code', 'temp_proper_code', 'mu', 'Energy_code', 'InternalEnergy_code'):
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
    sim.fluid._refresh_runtime_state()
    return sim
