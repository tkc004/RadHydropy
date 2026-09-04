"""Helper utilities for the cartesian Sedov-Taylor example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
import radhydropy.utils as ru
from radhydropy.units import CodeUnits
import SedovTaylor_analytic as sa


def set_plot_style():
    plotparams = {
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'font.size': 24,
        'legend.fontsize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'xtick.top': True,
        'ytick.right': True,
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.figsize': (30.45, 6.5),
        'figure.subplot.left': 0.2,
        'figure.subplot.right': 0.9,
        'figure.subplot.bottom': 0.2,
        'figure.subplot.top': 0.85,
        'figure.subplot.wspace': 0.2,
        'figure.subplot.hspace': 0.2,
        'lines.markersize': 6,
        'lines.linewidth': 3.0,
    }
    rcParams.update(plotparams)


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    icparams = config['initial_condition']
    runparams = config['par']
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

    dx = box_size[0] / grid_cells
    sim.mesh.boundary = np.linspace(
        -0.5 * dx,
        box_size[0] + dx,
        grid_cells + 1,
    )
    sim.mesh.coordinate = 0.5 * (
        sim.mesh.boundary[:-1] + sim.mesh.boundary[1:]
    )
    sim.fluid.vel_code = np.zeros(grid_cells) * unyt.cm / unyt.s
    sim.fluid.rho_code = icparams['initial_density'] * np.ones(grid_cells)
    sim.mesh.area = runparams['mesh']['area'] * np.ones(grid_cells)
    sim.mesh.vol = sim.mesh.area * (
        sim.mesh.boundary[1:] - sim.mesh.boundary[:-1]
    )
    sim.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']
    sim.fluid.temp_code = np.ones(grid_cells) * 0.0 * unyt.K
    icut = 1
    pre = icparams['explosion_energy'] / np.sum(sim.mesh.vol[icut]) * (
        runparams['hydrodynamics']['gamma'] - 1.0
    )
    sim.fluid.temp_code[icut] = ru.CalTemperature(
        sim.fluid.rho_code[icut],
        pre,
        sim.fluid.mu[icut],
    )


    return sim

def ReadandPlot(outfilename, config, **kwargs):
    icparams = config['initial_condition']
    runparams = config['par']
    rout = build_initial_condition(config)
    code_units_obj = config['_code_units']
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rout.fluid.pre_code = ru.CalPressure(rout.fluid.rho_code, rout.fluid.temp_code, rout.fluid.mu)
    nu = 1
    g = runparams['hydrodynamics']['gamma']
    w = 0.0
    if runparams['boundary']['condition'] == 'Periodic' or runparams['boundary']['condition'] == 'Open':
        E0 = icparams['explosion_energy']
    else:
        E0 = icparams['explosion_energy'] * 2.0
    rho1d0 = icparams['initial_density'] * runparams['mesh']['area']
    A0 = rho1d0
    t = rout.par.simulation.current_time * code_units_obj.time_unit
    r, rho, v, p, Rs = sa.get_blastwave_solution(E0, A0, nu, g, w, t)
    r = unyt.uconcatenate((r, unyt.unyt_array([1.0, 2] * Rs)))
    rho = unyt.uconcatenate((rho, unyt.unyt_array([rho1d0, rho1d0])))
    v = unyt.uconcatenate((
        v,
        unyt.unyt_array([0.0 * unyt.cm / unyt.s, 0.0 * unyt.cm / unyt.s]),
    ))
    p = unyt.uconcatenate((p, unyt.unyt_array([0.0 * unyt.dyn, 0.0 * unyt.dyn])))
    print('r', r)
    plt.subplot(1, 3, 1)
    rplot1d(rout, yquan='pre_code', showfig=0, showhalf=1, **kwargs)
    plt.plot(r.in_cgs(), p.in_cgs(), color=kwargs['color'])
    plt.subplot(1, 3, 2)
    rplot1d(rout, yquan='vel_code', showfig=0, showhalf=1, **kwargs)
    plt.plot(r.in_cgs(), v.in_cgs(), color=kwargs['color'])
    plt.subplot(1, 3, 3)
    rplot1d(rout, yquan='rho_code', showfig=0, showhalf=1, **kwargs)
    plt.plot(r.in_cgs(), (rho / runparams['mesh']['area']).in_cgs(), color=kwargs['color'])





