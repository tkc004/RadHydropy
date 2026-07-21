"""Helper utilities for the spherical advection example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
import advection_sph_analytic as asa


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

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)

        dx = self.par.boxsize[0] / self.par.nogrid
        self.mesh.boundary = np.linspace(
            dx,
            self.par.boxsize[0] + dx,
            self.par.nogrid + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])

        self.fluid.vel = icparams['vini'] * np.ones(self.par.nogrid)
        self.fluid.temp = icparams['tempini'] * np.ones(self.par.nogrid)
        rho = icparams['rhoini'] * np.ones(self.par.nogrid)
        rho[
            np.logical_or(
                coordinate < 0.25 * self.par.boxsize[0],
                coordinate > 0.75 * self.par.boxsize[0],
            )
        ] *= 0.01
        self.fluid.rho = rho
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout, yquan='rho', showfig=0, **kwargs)
    rout.mesh.vol = np.absolute(
        rout.mesh.boundary[1:]**3 - rout.mesh.boundary[:-1]**3
    ) * 4.0 * np.pi / 3.0
    mtot = np.sum(rout.fluid.rho * rout.mesh.vol)
    print('mtot', mtot)
    x = np.linspace(
        0.0 * icparams['boxsize'],
        icparams['boxsize'],
        icparams['nogrid'],
    )
    rho = asa.top_hat_density_profile(
        x,
        rout.par.time,
        icparams['vini'],
        icparams['boxsize'],
        icparams['rhoini'],
    )
    plt.plot(x, rho, color=kwargs['color'], ls='solid')
