"""Helper utilities for the cartesian advection example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, icparams):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()

        if ru.CheckParamDimen(icparams) != True:
            raise Exception('%s unit not correctly set in params' % ru.CheckParamDimen(icparams))

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)

        dx = self.par.boxsize[0] / self.par.nogrid
        self.mesh.boundary = np.linspace(
            0.0 * self.par.boxsize[0],
            self.par.boxsize[0] + dx,
            self.par.nogrid + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])

        rho = np.ones(self.par.nogrid) * icparams['rhoini']
        self.fluid.vel = np.ones(self.par.nogrid) * icparams['vini']
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['tempini']
        rho[
            np.logical_or(
                coordinate < 0.25 * self.par.boxsize[0],
                coordinate > 0.75 * self.par.boxsize[0],
            )
        ] *= 0.5
        self.fluid.rho = rho
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units = CodeUnits.from_mapping(runparams.get('CodeUnits'))
    rout.par.CodeUnits = code_units
    rout.par.unit_system = code_units.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    time = rout.par.time * code_units.time_unit
    x = np.linspace(
        0.0 * icparams['boxsize'],
        icparams['boxsize'],
        icparams['nogrid'],
    )
    rho = np.ones(icparams['nogrid']) * icparams['rhoini']
    x1 = 0.25 * icparams['boxsize'] + time * icparams['vini']
    x2 = 0.75 * icparams['boxsize'] + time * icparams['vini']
    if x1 > icparams['boxsize']:
        x1 -= icparams['boxsize']
    if x2 > icparams['boxsize']:
        x2 -= icparams['boxsize']
    if x2 > x1:
        rho[np.logical_or(x < x1, x > x2)] *= 0.5
    if x1 > x2:
        rho[np.logical_and(x > x1, x < x2)] *= 0.5
    plt.plot(x, rho, color=kwargs['color'], ls='solid')
    rplot1d(rout, showfig=0, **kwargs)
