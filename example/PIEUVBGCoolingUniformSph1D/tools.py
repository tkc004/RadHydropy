"""Helpers for the uniform HM12 PIE cooling example."""

from types import SimpleNamespace

import numpy as np


class Simwrap:
    """Build a uniform spherical IC object for ``writehdf5``."""

    def __init__(self, config, code_units, hydrogen_density_cgs_cm3):
        icparams = config['initial_condition']
        thermochemistry = config['par']['thermochemistry']
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        grid_cells = int(config['par']['mesh']['grid_cells'])
        boxsize = icparams["boxsize"] * np.ones(1)
        self.par.time = icparams["time"] * np.ones(1)
        self.par.simulation = SimpleNamespace(
            current_time=icparams["time"], box_size=icparams["boxsize"],
            coordinate_system=icparams["coordsys"],
        )
        self.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)

        dx = boxsize[0] / grid_cells
        self.mesh.boundary = np.linspace(
            dx, boxsize[0] + dx, grid_cells + 1
        )
        self.fluid.vel_code = np.zeros(grid_cells) * icparams["vini"]
        self.fluid.temp_code = np.ones(grid_cells) * icparams["tempini"]
        hydrogen_mass_fraction = float(icparams["hydrogen_mass_fraction"])
        proton_mass_g = float(icparams["proton_mass_g"])
        rho = hydrogen_density_cgs_cm3 * proton_mass_g / hydrogen_mass_fraction
        self.fluid.rho_code = np.ones(grid_cells) * rho
        self.fluid.mu = np.ones(grid_cells) * icparams["muini"]
