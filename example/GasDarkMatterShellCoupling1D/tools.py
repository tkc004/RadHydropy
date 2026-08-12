"""Initial-condition and dark-matter helpers for the coupled example."""

import numpy as np
import unyt

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams, code_units):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        self.par.nogrid = int(icparams['nogrid'])
        self.par.coordsys = 'spherical'
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']
        self.mesh.boundary = np.linspace(
            icparams['rmin'], icparams['rmax'], self.par.nogrid + 1
        )
        self.mesh.coordinate = 0.75 * (
            self.mesh.boundary[1:]**4 - self.mesh.boundary[:-1]**4
        ) / (self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = 4.0 * np.pi / 3.0 * (
            self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3
        )
        self.fluid.rho = np.ones(self.par.nogrid) * icparams['gas_density']
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['gas_temperature']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['mu']
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s


def make_dark_matter(icparams, code_units):
    count = int(icparams['dark_matter_shells'])
    radius = np.linspace(0.05, 0.95, count)
    velocity = np.asarray(radius) * float(icparams['dark_matter_velocity_scale'])
    angular_momentum = np.full(count, float(icparams['dark_matter_angular_momentum']))
    return DarkMatterShells(
        radius=radius,
        velocity=velocity,
        mass=np.full(count, icparams['dark_matter_mass'] / count),
        angular_momentum=angular_momentum,
        softening=icparams['dark_matter_softening'],
        code_units=code_units,
    )


def load_units(runparams):
    return CodeUnits.from_mapping(runparams['CodeUnits'])
