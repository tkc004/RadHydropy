"""Helper utilities for the cartesian outflow example."""

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
    icparams = config['initial_condition']
    code_units = config['_code_units']
    sim = SimpleNamespace()
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    if code_units is not None:
        sim.par.unit_system = code_units.unit_system

    grid_cells = icparams['grid_cells']
    box_size = icparams['box_size'] * np.ones(1)
    sim.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
    sim.par.simulation = SimpleNamespace(
        coordinate_system=icparams['coordinate_system'],
        current_time=icparams['current_time'] * np.ones(1),
        box_size=box_size,
    )

    sim.mesh.boundary = np.linspace(
        0.0 * box_size[0], box_size[0], grid_cells + 1,
    )
    sim.fluid.vel_code = icparams['initial_velocity'] * np.ones(grid_cells)
    sim.fluid.temp_code = icparams['initial_temperature'] * np.ones(grid_cells)
    sim.fluid.rho_code = icparams['initial_density'] * np.ones(grid_cells)
    sim.fluid.mu = icparams['mean_molecular_weight'] * np.ones(grid_cells)


    return sim

def ReadandPlot(outfilename, config, **kwargs):
    icparams = config['initial_condition']
    runparams = config['par']
    rout = build_initial_condition(config)
    code_units_obj = config['_code_units']
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout, yquan='rho_code', showhalf=0, showfig=0, **kwargs)
    plt.axvline(
        x=(rout.par.simulation.current_time * code_units_obj.time_unit)
        * runparams['boundary']['outflow_velocity'],
        color=kwargs['color'],
        ls='dashed',
    )





