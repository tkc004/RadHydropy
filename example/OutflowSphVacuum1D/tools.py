"""Initial condition and plotting helpers for outflow into vacuum."""

from types import SimpleNamespace

import numpy as np


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def analytic_density_profile(radius, time, icparams, runparams, cell_faces=None):
    """Cold spherical outflow profile, sampled as cell averages when given."""
    radius = np.asarray(radius, dtype=float)
    injection_radius = float(icparams['injection_radius'])
    boundary = runparams['boundary']
    density_outflow = float(boundary['outflow_density'])
    velocity_outflow = float(boundary['outflow_velocity'])
    front = injection_radius + velocity_outflow * float(time)
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
    icparams = config['initial_condition']
    code_units = config['_code_units']
    sim = SimpleNamespace()
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    sim.par.unit_system = code_units.unit_system
    grid_cells = int(icparams['grid_cells'])
    box_size = icparams['box_size'] * np.ones(1)
    sim.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
    sim.par.simulation = SimpleNamespace(
        coordinate_system=icparams['coordinate_system'],
        box_size=box_size,
        current_time=icparams['current_time'] * np.ones(1),
    )
    sim.mesh.boundary = np.linspace(
        float(icparams['injection_radius']),
        float(icparams['injection_radius'] + icparams['box_size']),
        grid_cells + 1,
    )
    sim.fluid.vel_code = icparams['initial_velocity'] * np.ones(grid_cells)
    sim.fluid.temp_code = icparams['initial_temperature'] * np.ones(grid_cells)
    sim.fluid.rho_code = icparams['initial_density'] * np.ones(grid_cells)
    sim.fluid.mu = icparams['mean_molecular_weight'] * np.ones(grid_cells)


    return sim
