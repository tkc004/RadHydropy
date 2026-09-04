"""Small IC helper for the fixed-density CMB Compton example."""

import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.units import CodeUnits


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    initial = config['initial_condition']
    runtime = config['par']
    simulation = runtime['simulation']
    mesh = runtime['mesh']
    code_units = config['_code_units']
    result = SimpleNamespace()
    result.par = Par()
    result.mesh = Mesh()
    result.fluid = Fluid()
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.simulation = SimpleNamespace(
        current_time=initial['current_time'], box_size=initial['box_size'],
        coordinate_system=simulation['coordinate_system'],
    )
    result.par.mesh = SimpleNamespace(grid_cells=int(mesh['grid_cells']), ghost_cells=0)
    result.par.hydrodynamics = SimpleNamespace(gamma=float(runtime.get('hydrodynamics', {}).get('gamma', 5.0 / 3.0)))
    result.par.coordinate_frame = 'physical'
    result.par.velocity_representation = 'physical'
    result.par.density_representation = 'physical'
    result.par.temperature_representation = 'physical'
    result.mesh.boundary = np.linspace(0.0, 1.0, result.par.mesh.grid_cells + 1) * initial['box_size']
    result.fluid.rho_code = np.ones(result.par.mesh.grid_cells) * initial['hydrogen_density'] * unyt.mp
    result.fluid.vel_code = np.zeros(result.par.mesh.grid_cells) * unyt.cm / unyt.s
    result.fluid.temp_code = np.ones(result.par.mesh.grid_cells) * initial['initial_temperature']
    result.fluid.xHI = np.ones(result.par.mesh.grid_cells) * initial['xHI']
    result.fluid.mu = np.ones(result.par.mesh.grid_cells) * initial['mean_molecular_weight']
    return result
