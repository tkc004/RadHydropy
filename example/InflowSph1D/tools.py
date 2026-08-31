"""Helper utilities for the spherical inflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
from radhydropy.units import CodeUnits
import inflow_sph_analytic as ia


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        if code_units is not None:
            self.par.unit_system = code_units.unit_system

        grid_cells = icparams['grid_cells']
        box_size = icparams['box_size'] * np.ones(1)
        self.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
        self.par.simulation = SimpleNamespace(coordinate_system=icparams['coordinate_system'], current_time=icparams['current_time'] * np.ones(1), box_size=box_size)

        dx = box_size[0] / grid_cells
        self.mesh.boundary = np.linspace(
            -0.5 * dx,
            box_size[0] + 0.5 * dx,
            grid_cells + 1,
        )
        self.fluid.vel = icparams['velocity'] * np.ones(grid_cells)
        self.fluid.temp = icparams['temperature'] * np.ones(grid_cells)
        self.fluid.rho = icparams['initial_density'] * np.ones(grid_cells)
        self.fluid.mu = icparams['mean_molecular_weight'] * np.ones(grid_cells)


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    time = rout.par.simulation.current_time * code_units_obj.time_unit
    radius = rout.mesh.boundary[:-1] * code_units_obj.length_unit
    rplot1d(rout, yquan='rho', showhalf=0, showfig=0, **kwargs)
    plt.ylim(ymax=10.1)
    plt.axvline(
        x=ia.front_position(
            icparams['box_size'],
            time,
            runparams['boundary']['inflow_velocity'],
        ),
        color=kwargs['color'],
        ls='dashed',
    )
    rhoana = ia.density_profile(
        radius,
        runparams['boundary']['inflow_density'],
        icparams['box_size'],
    )
    plt.plot(rout.mesh.boundary[:-1], rhoana, ls='dashed', color='k')
