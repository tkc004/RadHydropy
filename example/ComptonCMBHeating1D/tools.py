"""Small IC helper for the fixed-density CMB Compton example."""

import numpy as np
import unyt


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
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(icparams['nogrid'])
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']

        self.mesh.boundary = np.linspace(
            0.0,
            1.0,
            self.par.nogrid + 1,
        ) * icparams['boxsize']
        self.fluid.rho = (
            np.ones(self.par.nogrid)
            * icparams['nHini']
            * unyt.mp
        ).to(unyt.g / unyt.cm**3)
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['tempini']
        self.fluid.xHI = np.ones(self.par.nogrid) * icparams['xHIini']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
