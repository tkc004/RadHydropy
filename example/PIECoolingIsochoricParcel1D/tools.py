"""Initial-condition helper for the isochoric PIE parcel benchmark."""

from types import SimpleNamespace

import numpy as np
import unyt


class Simwrap:
    """Build a uniform, zero-velocity spherical parcel IC."""

    def __init__(self, icparams, code_units, hydrogen_density_cgs_cm3,
                 hydrogen_mass_fraction, temperature):
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordinate_system'],
            current_time=icparams['current_time'],
            box_size=icparams['box_size'],
        )
        grid_cells = int(icparams['grid_cells'])
        boxsize = icparams['box_size'] * np.ones(1)
        self.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=2)
        self.par.time = icparams['current_time'] * np.ones(1)
        dx = boxsize[0] / grid_cells
        self.mesh.boundary = np.linspace(
            dx, boxsize[0] + dx, grid_cells + 1
        )
        self.fluid.vel_code = np.zeros(grid_cells) * (0.0 * unyt.cm / unyt.s)
        self.fluid.temp_code = np.ones(grid_cells) * temperature
        proton_mass_g = 1.67262192369e-24
        rho = hydrogen_density_cgs_cm3 * proton_mass_g / hydrogen_mass_fraction
        self.fluid.rho_code = np.ones(grid_cells) * rho
        self.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']
