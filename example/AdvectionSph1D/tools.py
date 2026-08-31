"""Helper utilities for the spherical advection example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
from radhydropy.units import CodeUnits
import advection_sph_analytic as asa


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

        dx = box_size[0] / grid_cells
        self.mesh.boundary = np.linspace(
            dx,
            box_size[0] + dx,
            grid_cells + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])

        self.fluid.vel = icparams['initial_velocity'] * np.ones(grid_cells)
        self.fluid.temp = icparams['initial_temperature'] * np.ones(grid_cells)
        rho = icparams['initial_density'] * np.ones(grid_cells)
        rho[
            np.logical_or(
                coordinate < 0.25 * box_size[0],
                coordinate > 0.75 * box_size[0],
            )
        ] *= 0.01
        self.fluid.rho = rho
        self.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    time = rout.par.simulation.current_time * code_units_obj.time_unit
    radius = rout.mesh.boundary[:-1] * code_units_obj.length_unit
    rplot1d(rout, yquan='rho', showfig=0, **kwargs)
    rout.mesh.vol = np.absolute(
        rout.mesh.boundary[1:]**3 - rout.mesh.boundary[:-1]**3
    ) * 4.0 * np.pi / 3.0
    mtot = np.sum(rout.fluid.rho * rout.mesh.vol)
    print('mtot', mtot)
    x = rout.mesh.boundary[:-1] * code_units_obj.length_unit
    rho = asa.top_hat_density_profile(
        radius,
        time,
        icparams['initial_velocity'],
        icparams['box_size'],
        icparams['initial_density'],
    )
    plt.plot(x, rho, color=kwargs['color'], ls='solid')
