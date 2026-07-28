"""Helper utilities for the spherical hydrostatic point-mass example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.gravity import point_mass_potential
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


def spherical_cell_centers(boundary):
    """Return spherical cell centers consistent with the mesh geometry."""
    coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    vol_denom = boundary[1:]**3 - boundary[:-1]**3
    nonzero = vol_denom != 0.0
    coordinate[nonzero] = 0.75 * (
        boundary[1:][nonzero]**4 - boundary[:-1][nonzero]**4
    ) / vol_denom[nonzero]
    return coordinate


def point_mass_hydrostatic_density_profile(
    coordinate,
    rho_ref,
    temp,
    mu,
    point_mass,
    reference_radius,
):
    """Return the exact isothermal hydrostatic density profile."""
    c_s2 = sound_speed_squared(temp, mu)
    phi_ref = point_mass_potential(reference_radius, point_mass)
    phi = point_mass_potential(coordinate, point_mass)
    return rho_ref * np.exp(-(phi - phi_ref) / c_s2)


def point_mass_acceleration(point_mass, softening=0.0 * unyt.cm):
    """Return a callable for a point-mass gravitational acceleration field."""

    def _acceleration(coordinate):
        radius = np.maximum(coordinate, softening)
        return (
            -unyt.physical_constants.gravitational_constant
            * point_mass
            / radius**2
        ).to(unyt.cm / unyt.s**2)

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
            icparams['rmin'],
            icparams['rmax'],
            self.par.nogrid + 1,
        )
        self.mesh.coordinate = spherical_cell_centers(self.mesh.boundary)
        dx = self.mesh.boundary[1] - self.mesh.boundary[0]
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = (
            np.absolute(self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3)
            * 4.0
            * np.pi
            / 3.0
        )

        self.fluid.temp = np.ones(self.par.nogrid) * icparams['tempini']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.rho = point_mass_hydrostatic_density_profile(
            self.mesh.coordinate,
            icparams['rho_ref'],
            icparams['tempini'],
            icparams['muini'],
            icparams['point_mass'],
            reference_radius=self.mesh.coordinate[0],
        )


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    """Read a snapshot and compare it with the analytic hydrostatic profile."""
    rout = Simwrap(icparams)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    color = kwargs.get('color', 'C0')
    nghost = int(runparams.get('noghost', 0))
    xall = spherical_cell_centers(rout.mesh.boundary)
    if nghost > 0:
        xcoord = xall[nghost:-nghost]
        rho_num = rout.fluid.rho[nghost:-nghost]
        vel_num = rout.fluid.vel[nghost:-nghost]
    else:
        xcoord = xall
        rho_num = rout.fluid.rho
        vel_num = rout.fluid.vel
    rho_analytic = point_mass_hydrostatic_density_profile(
        xcoord,
        icparams['rho_ref'],
        icparams['tempini'],
        icparams['muini'],
        icparams['point_mass'],
        reference_radius=xcoord[0],
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
