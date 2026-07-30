"""Helper utilities for the spherical ballistic-infall example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
import radhydropy.io as rio
from radhydropy.units import code_unit_scales


ACCELERATION_UNIT = unyt.cm / unyt.s**2


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def spherical_cell_centers(boundary):
    """Return spherical cell centers consistent with the mesh geometry."""
    coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    vol_denom = boundary[1:]**3 - boundary[:-1]**3
    nonzero = vol_denom != 0.0
    coordinate[nonzero] = 0.75 * (
        boundary[1:][nonzero]**4 - boundary[:-1][nonzero]**4
    ) / vol_denom[nonzero]
    return coordinate


def point_mass_acceleration(point_mass, softening=0.0, code_units=None):
    """Return a callable for a point-mass gravitational acceleration field."""
    if code_units is not None:
        scales = code_unit_scales(code_units)
        point_mass = (
            point_mass.to_value(unyt.g)
            if hasattr(point_mass, "to_value")
            else float(point_mass) * scales["mass_g"]
        )
        softening = (
            softening.to_value(unyt.cm)
            if hasattr(softening, "to_value")
            else float(softening) * scales["length_cm"]
        )
        coord_unit = code_units.length_unit
        accel_unit = ACCELERATION_UNIT
    else:
        point_mass = (
            point_mass.to_value(unyt.g)
            if hasattr(point_mass, "to_value")
            else float(point_mass)
        )
        softening = (
            softening.to_value(unyt.cm)
            if hasattr(softening, "to_value")
            else float(softening)
        )
        coord_unit = unyt.cm
        accel_unit = ACCELERATION_UNIT

    def _acceleration(coordinate):
        radius = (
            coordinate.to_value(coord_unit)
            if hasattr(coordinate, "to_value")
            else np.asarray(coordinate, dtype=float)
        )
        if code_units is not None:
            radius = radius * scales["length_cm"]
        radius = np.maximum(radius, softening)
        return (
            -GRAVITATIONAL_CONSTANT_CGS * point_mass / radius**2
        ) * accel_unit

    return _acceleration


def ballistic_density_profile(coordinate, rho_ref):
    """Return a constant-density profile for the ballistic infall example."""
    return np.ones(np.shape(coordinate), dtype=float) * rho_ref


def ballistic_velocity_profile(coordinate, point_mass, time, softening=0.0):
    """Return the short-time free-fall velocity under a point mass."""
    acceleration = point_mass_acceleration(
        point_mass,
        softening=softening,
        code_units=None,
    )(coordinate)
    return acceleration * time


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
        self.fluid.rho = ballistic_density_profile(
            self.mesh.coordinate,
            icparams['rho_ref'],
        )


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    """Read a snapshot and compare it with the ballistic short-time profile."""
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
    rho_analytic = ballistic_density_profile(xcoord, icparams['rho_ref'])
    vel_analytic = ballistic_velocity_profile(
        xcoord,
        icparams['point_mass'],
        rout.fluid.time,
    )
    zero_velocity = np.zeros(len(xcoord)) * unyt.cm / unyt.s
    xcoord_cgs = xcoord.in_cgs()
    rho_num_cgs = rho_num.in_cgs()
    rho_analytic_cgs = rho_analytic.in_cgs()
    vel_num_cgs = vel_num.in_cgs()
    vel_analytic_cgs = vel_analytic.in_cgs()
    zero_velocity_cgs = zero_velocity.in_cgs()

    plt.subplot(1, 2, 1)
    plt.plot(xcoord_cgs, rho_num_cgs, label='numerical', **kwargs)
    plt.plot(
        xcoord_cgs,
        rho_analytic_cgs,
        ls='dashed',
        color=color,
        label='analytic',
    )
    plt.xlabel(rf"$r \; [{xcoord_cgs.units.latex_repr}]$")
    plt.ylabel(rf"$\rho \; [{rho_num_cgs.units.latex_repr}]$")
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(xcoord_cgs, vel_num_cgs, label='numerical', **kwargs)
    plt.plot(
        xcoord_cgs,
        vel_analytic_cgs,
        ls='dashed',
        color=color,
        label='free-fall',
    )
    plt.plot(
        xcoord_cgs,
        zero_velocity_cgs,
        ls='dotted',
        color='C2',
        label='zero velocity',
    )
    plt.xlabel(rf"$r \; [{xcoord_cgs.units.latex_repr}]$")
    plt.ylabel(rf"$v \; [{vel_num_cgs.units.latex_repr}]$")
    plt.legend(loc='best')
