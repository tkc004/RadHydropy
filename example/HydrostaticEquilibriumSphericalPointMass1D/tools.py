"""Helper utilities for the spherical hydrostatic point-mass example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, GRAVITATIONAL_CONSTANT_CGS, PROTON_MASS_CGS
import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_unit_scales

SPEED_SQUARED_UNIT = unyt.cm**2 / unyt.s**2
DENSITY_UNIT = unyt.g / unyt.cm**3
ACCELERATION_UNIT = unyt.cm / unyt.s**2


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def sound_speed_squared(temp, mu, code_units=None):
    """Return the isothermal sound speed squared."""
    if hasattr(temp, "to_value"):
        temp_value = float(temp.to_value(unyt.K))
    elif code_units is not None:
        temp_value = float(np.asarray(temp, dtype=float)) * code_unit_scales(code_units)["temperature_K"]
    else:
        temp_value = float(temp)
    mu_value = float(np.asarray(mu, dtype=float))
    return (
        BOLTZMANN_CONSTANT_CGS
        * temp_value
        / (mu_value * PROTON_MASS_CGS)
    ) * SPEED_SQUARED_UNIT


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
    code_units=None,
):
    """Return the exact isothermal hydrostatic density profile."""
    c_s2 = sound_speed_squared(temp, mu, code_units=code_units)
    c_s2_value = c_s2.to_value(unyt.cm**2 / unyt.s**2)
    if hasattr(coordinate, "to_value"):
        coord_value = coordinate.to_value(unyt.cm)
    elif code_units is not None:
        coord_value = np.asarray(coordinate, dtype=float) * code_unit_scales(code_units)["length_cm"]
    else:
        coord_value = np.asarray(coordinate, dtype=float)
    if hasattr(reference_radius, "to_value"):
        reference_radius_value = reference_radius.to_value(unyt.cm)
    elif code_units is not None:
        reference_radius_value = np.asarray(reference_radius, dtype=float) * code_unit_scales(code_units)["length_cm"]
    else:
        reference_radius_value = float(reference_radius)
    if hasattr(point_mass, "to_value"):
        point_mass_value = point_mass.to_value(unyt.g)
    elif code_units is not None:
        point_mass_value = np.asarray(point_mass, dtype=float) * code_unit_scales(code_units)["mass_g"]
    else:
        point_mass_value = float(point_mass)
    if hasattr(rho_ref, "to_value"):
        rho_value = rho_ref.to_value(unyt.g / unyt.cm**3)
    elif code_units is not None:
        rho_value = np.asarray(rho_ref, dtype=float) * code_unit_scales(code_units)["density_g_cm3"]
    else:
        rho_value = float(rho_ref)
    phi_ref = -GRAVITATIONAL_CONSTANT_CGS * point_mass_value / reference_radius_value
    phi = -GRAVITATIONAL_CONSTANT_CGS * point_mass_value / coord_value
    exponent = -(phi - phi_ref) / c_s2_value
    return rho_value * np.exp(exponent) * DENSITY_UNIT


def point_mass_acceleration(point_mass, softening=0.0, code_units=None):
    """Return a callable for a point-mass gravitational acceleration field."""
    if hasattr(point_mass, "to_value"):
        point_mass = point_mass.to_value(unyt.g)
    elif code_units is not None:
        point_mass = np.asarray(point_mass, dtype=float) * code_unit_scales(code_units)["mass_g"]
    else:
        point_mass = float(point_mass)
    if hasattr(softening, "to_value"):
        softening = softening.to_value(unyt.cm)
    elif code_units is not None:
        softening = np.asarray(softening, dtype=float) * code_unit_scales(code_units)["length_cm"]
    else:
        softening = float(softening)

    def _acceleration(coordinate):
        if hasattr(coordinate, "to_value"):
            radius = coordinate.to_value(unyt.cm)
        elif code_units is not None:
            radius = np.asarray(coordinate, dtype=float) * code_unit_scales(code_units)["length_cm"]
        else:
            radius = np.asarray(coordinate, dtype=float)
        radius = np.maximum(radius, softening)
        return (
            -GRAVITATIONAL_CONSTANT_CGS * point_mass / radius**2
        ) * ACCELERATION_UNIT

    return _acceleration


class Simwrap:
    def __init__(self, icparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.code_units = code_units
        self.par.CodeUnits = code_units

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
            code_units=code_units,
        )


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    """Read a snapshot and compare it with the analytic hydrostatic profile."""
    code_units = CodeUnits.from_mapping(runparams.get('CodeUnits'))
    rout = Simwrap(icparams, code_units=code_units)
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
        code_units=code_units,
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
