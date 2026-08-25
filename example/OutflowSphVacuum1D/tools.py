"""Initial condition and plotting helpers for outflow into vacuum."""

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
    injection_radius = float(icparams['rinj'])
    density_outflow = float(runparams['rho_outflow'])
    velocity_outflow = float(runparams['vel_outflow'])
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
        self.par.CodeUnits = code_units
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(icparams['nogrid'])
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)
        self.mesh.boundary = np.linspace(
            float(icparams['rinj']),
            float(icparams['rinj'] + icparams['boxsize']),
            self.par.nogrid + 1,
        )
        self.fluid.vel = icparams['vini'] * np.ones(self.par.nogrid)
        self.fluid.temp = icparams['tempini'] * np.ones(self.par.nogrid)
        self.fluid.rho = icparams['rhoini'] * np.ones(self.par.nogrid)
        self.fluid.mu = icparams['muini'] * np.ones(self.par.nogrid)
