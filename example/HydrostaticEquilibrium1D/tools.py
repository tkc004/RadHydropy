"""Helper utilities for the hydrostatic-equilibrium check example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.analysis import rplot1d
import radhydropy.io as rio


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def sound_speed_squared(temp, mu):
    """Return the isothermal sound speed squared."""
    return (unyt.kb * temp / (mu * unyt.mp)).to(unyt.cm**2 / unyt.s**2)


def hydrostatic_density_profile(coordinate, rho_ref, temp, mu, gravity_strength):
    """Return the exact isothermal hydrostatic density profile."""
    c_s2 = sound_speed_squared(temp, mu)
    scale_height = c_s2 / gravity_strength
    return rho_ref * np.exp(-coordinate / scale_height)


def constant_gravity_acceleration(gravity_strength):
    """Return a callable for a uniform downward acceleration field."""

    def _acceleration(coordinate):
        return -gravity_strength * np.ones(np.shape(coordinate))

    return _acceleration


class Simwrap:
    def __init__(self, icparams):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']

        self.mesh.boundary = np.linspace(
            0.0,
            1.0,
            self.par.nogrid + 1,
        ) * icparams['boxsize']
        self.mesh.coordinate = 0.5 * (
            self.mesh.boundary[:-1] + self.mesh.boundary[1:]
        )
        dx = self.mesh.boundary[1] - self.mesh.boundary[0]
        self.mesh.area = np.ones(self.par.nogrid) * (1.0 * unyt.cm**2)
        self.mesh.vol = self.mesh.area * dx

        self.fluid.temp = np.ones(self.par.nogrid) * icparams['tempini']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.rho = hydrostatic_density_profile(
            self.mesh.coordinate,
            icparams['rho_ref'],
            icparams['tempini'],
            icparams['muini'],
            icparams['gravity_strength'],
        )


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    """Read a snapshot and compare it with the analytic hydrostatic profile."""
    rout = Simwrap(icparams)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    color = kwargs.get('color', 'C0')
    nghost = int(runparams.get('noghost', 0))
    xall = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    if nghost > 0:
        xcoord = xall[nghost:-nghost]
        rho_num = rout.fluid.rho[nghost:-nghost]
        vel_num = rout.fluid.vel[nghost:-nghost]
    else:
        xcoord = xall
        rho_num = rout.fluid.rho
        vel_num = rout.fluid.vel
    rho_analytic = hydrostatic_density_profile(
        xcoord,
        icparams['rho_ref'],
        icparams['tempini'],
        icparams['muini'],
        icparams['gravity_strength'],
    )
    zero_velocity = np.zeros(len(xcoord)) * unyt.cm / unyt.s

    plt.subplot(1, 2, 1)
    plt.plot(xcoord.in_cgs(), rho_num.in_cgs(), **kwargs)
    plt.plot(
        xcoord.in_cgs(),
        rho_analytic.in_cgs(),
        ls='dashed',
        color=color,
    )
    plt.ylabel(r"$\rho$")

    plt.subplot(1, 2, 2)
    plt.plot(xcoord.in_cgs(), vel_num.in_cgs(), **kwargs)
    plt.plot(
        xcoord.in_cgs(),
        zero_velocity.in_cgs(),
        ls='dashed',
        color=color,
    )
    plt.ylabel(r"$v$")
