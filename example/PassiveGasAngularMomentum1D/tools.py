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


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    result = SimpleNamespace(par=Par(), mesh=Mesh(), fluid=Fluid())
    result.par.CodeUnits = code_units
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.unit_system = code_units.unit_system
    result.par.nogrid = grid_cells
    result.par.coordsys = 'cartesian'
    result.par.boxsize = initial['box_size'] * np.ones(1)
    result.par.time = initial['current_time'] * np.ones(1)
    result.par.simulation = SimpleNamespace(current_time=result.par.time, box_size=result.par.boxsize, coordinate_system='cartesian')
    result.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
    result.mesh.boundary = np.linspace(0.0, result.par.boxsize[0], grid_cells + 1)
    coordinate = 0.5 * (result.mesh.boundary[1:] + result.mesh.boundary[:-1])
    phase = 2.0 * np.pi * coordinate / result.par.boxsize[0]
    result.fluid.rho_code = np.ones(grid_cells) * initial['initial_density']
    result.fluid.vel_code = np.ones(grid_cells) * initial['velocity']
    result.fluid.temp_code = np.ones(grid_cells) * initial['temperature']
    result.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']
    if initial.get('include_angular_momentum', True):
        result.fluid.specific_angular_momentum_code = initial['angular_momentum_offset'] + initial['angular_momentum_amplitude'] * np.sin(phase)
    return result
