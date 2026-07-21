"""Helper utilities for the cartesian outflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radhydropy.analysis import rplot1d
import radhydropy.io as rio


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

        self.mesh.boundary = np.linspace(
            0.0 * self.par.boxsize[0],
            self.par.boxsize[0],
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
    plt.axvline(
        x=rout.par.time * runparams['vel_outflow'],
        color=kwargs['color'],
        ls='dashed',
    )
