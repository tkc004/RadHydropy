"""Helper utilities for the spherical inflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
import inflow_sph_analytic as ia


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
            -0.5 * dx,
            self.par.boxsize[0] + 0.5 * dx,
            self.par.nogrid + 1,
        )
        self.fluid.vel = icparams['vini'] * np.ones(self.par.nogrid)
        self.fluid.temp = icparams['tempini'] * np.ones(self.par.nogrid)
        self.fluid.rho = icparams['rhoini'] * np.ones(self.par.nogrid)
        self.fluid.mu = icparams['muini'] * np.ones(self.par.nogrid)


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout, yquan='rho', showhalf=0, showfig=0, **kwargs)
    plt.ylim(ymax=10.1)
    plt.axvline(
        x=ia.front_position(
            icparams['boxsize'],
            rout.par.time,
            runparams['vel_inflow'],
        ),
        color=kwargs['color'],
        ls='dashed',
    )
    rhoana = ia.density_profile(
        rout.mesh.boundary[:-1],
        runparams['rho_inflow'],
        icparams['boxsize'],
    )
    plt.plot(rout.mesh.boundary[:-1], rhoana, ls='dashed', color='k')
