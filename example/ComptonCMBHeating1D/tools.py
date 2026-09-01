"""Small IC helper for the fixed-density CMB Compton example."""

import numpy as np
import unyt
from types import SimpleNamespace


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams, simulation, mesh, code_units):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(mesh['grid_cells'])
        self.par.coordsys = simulation['coordinate_system']
        self.par.boxsize = np.ones(1) * icparams['box_size']
        self.par.time = np.ones(1) * icparams['current_time']
        self.par.simulation = SimpleNamespace(
            current_time=icparams['current_time'], box_size=icparams['box_size'],
            coordinate_system=simulation['coordinate_system'])
        self.par.mesh = SimpleNamespace(grid_cells=self.par.nogrid, ghost_cells=0)

        self.mesh.boundary = np.linspace(
            0.0,
            1.0,
            self.par.nogrid + 1,
        ) * icparams['box_size']
        self.fluid.rho = (
            np.ones(self.par.nogrid)
            * icparams['hydrogen_density']
            * unyt.mp
        ).to(unyt.g / unyt.cm**3)
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['initial_temperature']
        self.fluid.xHI = np.ones(self.par.nogrid) * icparams['xHI']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['mean_molecular_weight']
