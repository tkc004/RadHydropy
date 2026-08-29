"""Initial-condition helper for the passive gas angular-momentum example."""

import numpy as np
import unyt


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)

        dx = self.par.boxsize[0] / self.par.nogrid
        self.mesh.boundary = np.linspace(
            0.0 * self.par.boxsize[0],
            self.par.boxsize[0],
            self.par.nogrid + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])
        phase = 2.0 * np.pi * coordinate / self.par.boxsize[0]

        self.fluid.rho = np.ones(self.par.nogrid) * icparams['rhoini']
        self.fluid.vel = np.ones(self.par.nogrid) * icparams['vini']
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['tempini']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
        if icparams.get('include_angular_momentum', True):
            self.fluid.specific_angular_momentum = (
                icparams['angular_momentum_offset']
                + icparams['angular_momentum_amplitude'] * np.sin(phase)
            )
