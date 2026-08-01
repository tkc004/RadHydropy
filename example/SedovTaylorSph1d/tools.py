"""Helper utilities for the spherical Sedov-Taylor example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import unyt

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


class Simwrap:
    def __init__(self, icparams, runparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        if code_units is not None:
            self.par.unit_system = code_units.unit_system

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)

        self.mesh.boundary = np.linspace(
            icparams['rinj'],
            icparams['rinj'] + self.par.boxsize[0],
            self.par.nogrid + 1,
        )
        self.mesh.coordinate = 0.5 * (
            self.mesh.boundary[:-1] + self.mesh.boundary[1:]
        )
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.rho = icparams['rhoini'] * np.ones(self.par.nogrid)
        self.mesh.vol = (
            self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3
        ) * 4.0 * np.pi / 3.0
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
        self.fluid.mass = self.fluid.rho * self.mesh.vol
        self.fluid.temp = np.ones(self.par.nogrid) * 0.0 * unyt.K
        icut = np.logical_and(
            self.mesh.coordinate < icparams['rini'],
            self.mesh.coordinate >= icparams['rinj'],
        )
        pre = icparams['Eini'] / np.sum(self.mesh.vol[icut]) * (
            runparams['gamma'] - 1.0
        )
        self.fluid.temp[icut] = ru.CalTemperature(
            self.fluid.rho[icut],
            pre,
            self.fluid.mu[icut],
        )


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams, runparams)
    code_units_obj = CodeUnits.from_mapping(runparams['CodeUnits'])
    rout.par.CodeUnits = code_units_obj
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rout.fluid.pre = ru.CalPressure(rout.fluid.rho, rout.fluid.temp, rout.fluid.mu)
    nu = 3
    g = runparams['gamma']
    w = 0.0
    E0 = float(np.asarray(icparams['Eini'].to_value(unyt.erg), dtype=float))
    A0 = float(np.asarray(icparams['rhoini'].to_value(unyt.g / unyt.cm**3), dtype=float))
    t = float(np.asarray(rout.par.time, dtype=float))
    r, rho, v, p, Rs = sa.get_blastwave_solution(E0, A0, nu, g, w, t)
    r = r * unyt.cm
    rho = rho * (unyt.g / unyt.cm**3)
    v = v * (unyt.cm / unyt.s)
    p = p * (unyt.dyn / unyt.cm**2)
    r = np.concatenate((r, unyt.unyt_array(np.array([1.0, 2.0]) * Rs) * unyt.cm))
    rho = np.concatenate((
        rho,
        unyt.unyt_array([icparams['rhoini'], icparams['rhoini']]).to(unyt.g / unyt.cm**3),
    ))
    v = np.concatenate((
        v,
        unyt.unyt_array([0.0 * unyt.cm / unyt.s, 0.0 * unyt.cm / unyt.s]),
    ))
    p = np.concatenate((
        p,
        unyt.unyt_array([0.0 * unyt.dyn / unyt.cm**2, 0.0 * unyt.dyn / unyt.cm**2]),
    ))
    plt.subplot(1, 3, 1)
    rplot1d(rout, yquan='pre', showfig=0, **kwargs)
    plt.plot(r.in_cgs(), p.in_cgs(), color=kwargs['color'])
    plt.xlim([0, 4])
    plt.subplot(1, 3, 2)
    rplot1d(rout, yquan='vel', showfig=0, **kwargs)
    plt.plot(r.in_cgs(), v.in_cgs(), color=kwargs['color'])
    plt.xlim([0, 4])
    plt.subplot(1, 3, 3)
    rplot1d(rout, yquan='rho', showfig=0, **kwargs)
    plt.plot(r.in_cgs(), rho.in_cgs(), color=kwargs['color'])
    plt.xlim([0, 4])
