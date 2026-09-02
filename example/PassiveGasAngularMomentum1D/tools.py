"""Initial-condition helper for the passive gas angular-momentum example."""

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
    def __init__(self, icparams, code_units=None, grid_cells=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(grid_cells)
        self.par.coordsys = 'cartesian'
        self.par.boxsize = icparams['box_size'] * np.ones(1)
        self.par.time = icparams['current_time'] * np.ones(1)
        self.par.simulation = SimpleNamespace(
            current_time=self.par.time,
            box_size=self.par.boxsize,
            coordinate_system='cartesian',
        )
        self.par.mesh = SimpleNamespace(grid_cells=self.par.nogrid, ghost_cells=0)

        dx = self.par.boxsize[0] / self.par.nogrid
        self.mesh.boundary = np.linspace(
            0.0 * self.par.boxsize[0],
            self.par.boxsize[0],
            self.par.nogrid + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])
        phase = 2.0 * np.pi * coordinate / self.par.boxsize[0]

        self.fluid.rho_code = np.ones(self.par.nogrid) * icparams['initial_density']
        self.fluid.vel_code = np.ones(self.par.nogrid) * icparams['velocity']
        self.fluid.temp_code = np.ones(self.par.nogrid) * icparams['temperature']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['mean_molecular_weight']
        if icparams.get('include_angular_momentum', True):
            self.fluid.specific_angular_momentum_code = (
                icparams['angular_momentum_offset']
                + icparams['angular_momentum_amplitude'] * np.sin(phase)
            )
