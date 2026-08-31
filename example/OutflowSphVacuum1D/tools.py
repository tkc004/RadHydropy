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


class Simwrap:
    def __init__(self, icparams, code_units):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.unit_system = code_units.unit_system
        grid_cells = int(icparams['grid_cells'])
        box_size = icparams['box_size'] * np.ones(1)
        self.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordinate_system'],
            box_size=box_size,
            current_time=icparams['current_time'] * np.ones(1),
        )
        self.mesh.boundary = np.linspace(
            float(icparams['injection_radius']),
            float(icparams['injection_radius'] + icparams['box_size']),
            grid_cells + 1,
        )
        self.fluid.vel = icparams['initial_velocity'] * np.ones(grid_cells)
        self.fluid.temp = icparams['initial_temperature'] * np.ones(grid_cells)
        self.fluid.rho = icparams['initial_density'] * np.ones(grid_cells)
        self.fluid.mu = icparams['mean_molecular_weight'] * np.ones(grid_cells)
