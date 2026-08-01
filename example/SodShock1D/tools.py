"""Helper utilities for the Sod shock-tube example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

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
    def __init__(self, icparams):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()

        if ru.CheckParamDimen(icparams) != True:
            raise Exception('%s unit not correctly set in params' % ru.CheckParamDimen(icparams))

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = np.array([0.0]) * icparams['time']

        dx = self.par.boxsize[0] / self.par.nogrid
        self.mesh.boundary = np.linspace(
            -dx,
            self.par.boxsize[0] + dx,
            self.par.nogrid + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])

        rho = np.ones(self.par.nogrid) * icparams['rhoini']
        self.fluid.vel = np.ones(self.par.nogrid) * icparams['vini']
        indexlow = np.logical_and(
            coordinate > 0.25 * self.par.boxsize[0],
            coordinate < 0.75 * self.par.boxsize[0],
        )
        rho[indexlow] *= icparams['rhoratio']
        self.fluid.rho = rho
        temp = np.ones(self.par.nogrid) * icparams['tempini']
        temp[indexlow] *= icparams['tempratio']
        self.fluid.temp = temp
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']


def getAnalyticSolution(icparams, runparams, rout):
    code_units = getattr(rout.par, 'CodeUnits', None)
    if code_units is None:
        code_units = CodeUnits.from_mapping(runparams.get('CodeUnits'))
    time = rout.par.time
    if not hasattr(time, 'in_cgs'):
        time = time * code_units.time_unit
    boundary = rout.mesh.boundary
    if not hasattr(boundary, 'in_cgs'):
        boundary = np.asarray(boundary, dtype=float) * code_units.length_unit
    p5 = ru.CalPressure(icparams['rhoini'], icparams['tempini'], icparams['muini'])
    p1 = ru.CalPressure(
        icparams['rhoini'] * icparams['rhoratio'],
        icparams['tempini'] * icparams['tempratio'],
        icparams['muini'],
    )
    p5 = np.array(p5.in_cgs())
    p1 = np.array(p1.in_cgs())
    rho5 = np.array(icparams['rhoini'].in_cgs())
    rho1 = np.array((icparams['rhoini'] * icparams['rhoratio']).in_cgs())

    rho2, rho3, p2, v2, vt, vs, Mach = shocktubecal(
        runparams['gamma'],
        rho1,
        rho5,
        p1,
        p5,
    )
    rho_ana, p_ana, v_ana = shocktubeanalyticgraph(
        runparams['gamma'],
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
        np.array(0.25 * icparams['boxsize']),
    )
    return rho_ana, p_ana, v_ana


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units = CodeUnits.from_mapping(runparams.get('CodeUnits'))
    rout.par.CodeUnits = code_units
    rout.par.unit_system = code_units.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout, yquan='rho', showfig=0, showhalf=1, **kwargs)
    rho_ana, p_ana, v_ana = getAnalyticSolution(icparams, runparams, rout)
    plt.plot(rout.mesh.boundary, rho_ana)
