"""Helper utilities for the cartesian advection example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
from radhydropy.units import CodeUnits


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    sim = SimpleNamespace()
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    if code_units is not None:
        sim.par.unit_system = code_units.unit_system

    grid_cells = config['par']['mesh']['grid_cells']
    box_size = initial['box_size'] * np.ones(1)
    sim.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
    sim.par.simulation = SimpleNamespace(
        coordinate_system=config['par']['simulation']['coordinate_system'],
        current_time=initial['current_time'] * np.ones(1),
        box_size=box_size,
    )

    dx = box_size[0] / grid_cells
    sim.mesh.boundary = np.linspace(
        0.0 * box_size[0], box_size[0] + dx, grid_cells + 1,
    )
    coordinate = 0.5 * (sim.mesh.boundary[1:] + sim.mesh.boundary[:-1])

    rho = np.ones(grid_cells) * initial['initial_density']
    sim.fluid.vel_code = np.ones(grid_cells) * initial['initial_velocity']
    sim.fluid.temp_code = np.ones(grid_cells) * initial['initial_temperature']
    rho[
        np.logical_or(
            coordinate < 0.25 * box_size[0],
            coordinate > 0.75 * box_size[0],
        )
    ] *= 0.5
    sim.fluid.rho_code = rho
    sim.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']


    return sim

def ReadandPlot(outfilename, config, **kwargs):
    initial = config['initial_condition']
    run = config['par']
    rout = build_initial_condition(config)
    code_units_obj = config['_code_units']
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    time = rout.par.simulation.current_time * code_units_obj.time_unit
    x = np.linspace(
        0.0 * initial['box_size'],
        initial['box_size'],
        initial['grid_cells'],
    )
    rho = np.ones(initial['grid_cells']) * initial['initial_density']
    x1 = 0.25 * initial['box_size'] + time * initial['initial_velocity']
    x2 = 0.75 * initial['box_size'] + time * initial['initial_velocity']
    if x1 > initial['box_size']:
        x1 -= initial['box_size']
    if x2 > initial['box_size']:
        x2 -= initial['box_size']
    if x2 > x1:
        rho[np.logical_or(x < x1, x > x2)] *= 0.5
    if x1 > x2:
        rho[np.logical_and(x > x1, x < x2)] *= 0.5
    plt.plot(x, rho, color=kwargs['color'], ls='solid')
    rplot1d(rout, showfig=0, **kwargs)





