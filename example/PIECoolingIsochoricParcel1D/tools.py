"""Initial-condition helper for the isochoric PIE parcel benchmark."""

from types import SimpleNamespace

import numpy as np
import unyt


def build_initial_condition(config):
    initial = config['initial_condition']
    thermochemistry = config['par']['thermochemistry']
    code_units = config['_code_units']
    density_cgs_cm3 = initial['hydrogen_density_cgs_cm3']
    temperature_unyt = initial['temperature_unyt']
    hydrogen_mass_fraction = float(thermochemistry['hydrogen_mass_fraction'])
    result = SimpleNamespace()
    result.par = SimpleNamespace()
    result.mesh = SimpleNamespace()
    result.fluid = SimpleNamespace()
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.simulation = SimpleNamespace(
        coordinate_system=initial['coordinate_system'],
        current_time=initial['current_time'],
        box_size=initial['box_size'],
    )
    grid_cells = int(config['par']['mesh']['grid_cells'])
    boxsize = initial['box_size'] * np.ones(1)
    result.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=2)
    result.par.time = initial['current_time'] * np.ones(1)
    dx = boxsize[0] / grid_cells
    result.mesh.boundary = np.linspace(dx, boxsize[0] + dx, grid_cells + 1)
    result.fluid.vel_code = np.zeros(grid_cells) * (0.0 * unyt.cm / unyt.s)
    result.fluid.temp_code = np.ones(grid_cells) * temperature_unyt
    rho_cgs_g_cm3 = density_cgs_cm3 * unyt.mp.to_value(unyt.g) / hydrogen_mass_fraction
    result.fluid.rho_code = np.ones(grid_cells) * rho_cgs_g_cm3
    result.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']
    return result
