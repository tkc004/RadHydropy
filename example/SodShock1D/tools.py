"""Helper utilities for the Sod shock-tube example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
import radhydropy.utils as ru
from radhydropy.units import CodeUnits
from sodshock_analytic import shocktubecal, shocktubeanalyticgraph


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
            current_time=np.array([0.0]) * icparams['current_time'],
            box_size=box_size,
        )

        dx = box_size[0] / grid_cells
        self.mesh.boundary = np.linspace(
            -dx,
            box_size[0] + dx,
            grid_cells + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])

        rho = np.ones(grid_cells) * icparams['initial_density']
        self.fluid.vel_code = np.ones(grid_cells) * icparams['initial_velocity']
        indexlow = np.logical_and(
            coordinate > 0.25 * box_size[0],
            coordinate < 0.75 * box_size[0],
        )
        rho[indexlow] *= icparams['density_ratio']
        self.fluid.rho_code = rho
        temp = np.ones(grid_cells) * icparams['initial_temperature']
        temp[indexlow] *= icparams['temperature_ratio']
        self.fluid.temp_code = temp
        self.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']


def getAnalyticSolution(icparams, runparams, rout):
    code_units_obj = getattr(rout.par.units, 'CodeUnits', None)
    if code_units_obj is None:
        code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    time = rout.par.simulation.current_time
    if not hasattr(time, 'in_cgs'):
        time = time * code_units_obj.time_unit
    boundary = rout.mesh.boundary
    if not hasattr(boundary, 'in_cgs'):
        boundary = np.asarray(boundary, dtype=float) * code_units_obj.length_unit
    p5 = ru.CalPressure(icparams['initial_density'], icparams['initial_temperature'], icparams['mean_molecular_weight'])
    p1 = ru.CalPressure(
        icparams['initial_density'] * icparams['density_ratio'],
        icparams['initial_temperature'] * icparams['temperature_ratio'],
        icparams['mean_molecular_weight'],
    )
    p5 = np.array(p5.in_cgs())
    p1 = np.array(p1.in_cgs())
    rho5 = np.array(icparams['initial_density'].in_cgs())
    rho1 = np.array((icparams['initial_density'] * icparams['density_ratio']).in_cgs())

    rho2, rho3, p2, v2, vt, vs, Mach = shocktubecal(
        runparams['hydrodynamics']['gamma'],
        rho1,
        rho5,
        p1,
        p5,
    )
    rho_ana, p_ana, v_ana = shocktubeanalyticgraph(
        runparams['hydrodynamics']['gamma'],
        rho1,
        rho2,
        rho3,
        rho5,
        p1,
        p2,
        p5,
        v2,
        vt,
        vs,
        np.array(time.in_cgs()),
        np.array(boundary.in_cgs()),
        np.array(0.25 * icparams['box_size']),
    )
    return rho_ana, p_ana, v_ana


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout, yquan='rho_code', showfig=0, showhalf=1, **kwargs)
    rho_ana, p_ana, v_ana = getAnalyticSolution(icparams, runparams, rout)
    plt.plot(rout.mesh.boundary, rho_ana)
