"""Initial-condition helper for the isochoric PIE parcel benchmark."""

from types import SimpleNamespace

import numpy as np


class Simwrap:
    """Build a uniform, zero-velocity spherical parcel IC."""

    def __init__(self, icparams, code_units, hydrogen_density_cm3):
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.par.CodeUnits = code_units
        self.par.unit_system = code_units.unit_system
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordsys'],
            current_time=icparams['time'],
            box_size=icparams['boxsize'],
        )
        self.par.nogrid = int(icparams['nogrid'])
        self.par.mesh = SimpleNamespace(grid_cells=self.par.nogrid, ghost_cells=2)
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)
        dx = self.par.boxsize[0] / self.par.nogrid
        self.mesh.boundary = np.linspace(
            dx, self.par.boxsize[0] + dx, self.par.nogrid + 1
        )
        self.fluid.vel = np.zeros(self.par.nogrid) * icparams['vini']
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['tempini']
        hydrogen_mass_fraction = float(icparams['hydrogen_mass_fraction'])
        proton_mass_g = float(icparams['proton_mass_g'])
        rho = hydrogen_density_cm3 * proton_mass_g / hydrogen_mass_fraction
        self.fluid.rho = np.ones(self.par.nogrid) * rho
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
