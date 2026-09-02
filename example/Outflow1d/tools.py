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
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordinate_system'],
            current_time=icparams['current_time'] * np.ones(1),
            box_size=box_size,
        )

        self.mesh.boundary = np.linspace(
            0.0 * box_size[0], box_size[0], grid_cells + 1,
        )
        self.fluid.vel_code = icparams['initial_velocity'] * np.ones(grid_cells)
        self.fluid.temp_code = icparams['initial_temperature'] * np.ones(grid_cells)
        self.fluid.rho_code = icparams['initial_density'] * np.ones(grid_cells)
        self.fluid.mu = icparams['mean_molecular_weight'] * np.ones(grid_cells)


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
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
