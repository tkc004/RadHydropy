"""Helper utilities for the spherical inflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
from radhydropy.units import CodeUnits
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
    code_units = CodeUnits.from_mapping(runparams.get('CodeUnits'))
    rout.par.CodeUnits = code_units
    rout.par.unit_system = code_units.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    time = rout.par.time * code_units.time_unit
    radius = rout.mesh.boundary[:-1] * code_units.length_unit
    rplot1d(rout, yquan='rho', showhalf=0, showfig=0, **kwargs)
    plt.ylim(ymax=10.1)
    plt.axvline(
        x=ia.front_position(
            icparams['boxsize'],
            time,
            runparams['vel_inflow'],
        ),
        color=kwargs['color'],
        ls='dashed',
    )
    rhoana = ia.density_profile(
        radius,
        runparams['rho_inflow'],
        icparams['boxsize'],
    )
    plt.plot(rout.mesh.boundary[:-1], rhoana, ls='dashed', color='k')
