"""Helper utilities for the cartesian advection example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
import radhydropy.utils as ru
from radhydropy.units import CodeUnits


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

        if ru.CheckParamDimen(icparams) != True:
            raise Exception('%s unit not correctly set in params' % ru.CheckParamDimen(icparams))

        grid_cells = icparams['grid_cells']
        box_size = icparams['box_size'] * np.ones(1)
        self.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordinate_system'],
            current_time=icparams['current_time'] * np.ones(1),
            box_size=box_size,
        )

        dx = box_size[0] / grid_cells
        self.mesh.boundary = np.linspace(
            0.0 * box_size[0], box_size[0] + dx, grid_cells + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])

        rho = np.ones(grid_cells) * icparams['initial_density']
        self.fluid.vel_code = np.ones(grid_cells) * icparams['initial_velocity']
        self.fluid.temp_code = np.ones(grid_cells) * icparams['initial_temperature']
        rho[
            np.logical_or(
                coordinate < 0.25 * box_size[0],
                coordinate > 0.75 * box_size[0],
            )
        ] *= 0.5
        self.fluid.rho_code = rho_code
        self.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    time = rout.par.simulation.current_time * code_units_obj.time_unit
    x = np.linspace(
        0.0 * icparams['box_size'],
        icparams['box_size'],
        icparams['grid_cells'],
    )
    rho = np.ones(icparams['grid_cells']) * icparams['initial_density']
    x1 = 0.25 * icparams['box_size'] + time * icparams['initial_velocity']
    x2 = 0.75 * icparams['box_size'] + time * icparams['initial_velocity']
    if x1 > icparams['box_size']:
        x1 -= icparams['box_size']
    if x2 > icparams['box_size']:
        x2 -= icparams['box_size']
    if x2 > x1:
        rho[np.logical_or(x < x1, x > x2)] *= 0.5
    if x1 > x2:
        rho[np.logical_and(x > x1, x < x2)] *= 0.5
    plt.plot(x, rho, color=kwargs['color'], ls='solid')
    rplot1d(rout, showfig=0, **kwargs)
