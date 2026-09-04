"""Helpers for the uniform HM12 PIE cooling example."""

from types import SimpleNamespace

import numpy as np


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    grid_cells = int(par['mesh']['grid_cells'])
    result = SimpleNamespace()
    result.par = SimpleNamespace(
        units=SimpleNamespace(CodeUnits=config['_code_units']),
        time=initial['time'] * np.ones(1),
        simulation=SimpleNamespace(
            current_time=initial['time'], box_size=initial['boxsize'],
            coordinate_system=initial['coordsys'],
        ),
        mesh=SimpleNamespace(grid_cells=grid_cells, ghost_cells=0),
    )
    result.mesh = SimpleNamespace()
    result.fluid = SimpleNamespace()
    boxsize = initial['boxsize']
    dx = boxsize / grid_cells
    result.mesh.boundary = np.linspace(dx, boxsize + dx, grid_cells + 1)
    result.fluid.vel_code = np.zeros(grid_cells) * initial['vini']
    result.fluid.temp_code = np.ones(grid_cells) * initial['tempini']
    rho = config['initial_condition']['hydrogen_density_cgs_cm3'] * float(initial['proton_mass_g']) / float(initial['hydrogen_mass_fraction'])
    result.fluid.rho_code = np.ones(grid_cells) * rho
    result.fluid.mu = np.ones(grid_cells) * initial['muini']
    return result
