"""Helper utilities for the spherical hydrostatic point-mass example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, GRAVITATIONAL_CONSTANT_CGS, PROTON_MASS_CGS
import radhydropy.io as rio
from radhydropy.units import (
    CodeUnits,
    code_quantity_to_cgs,
    code_unit_scales,
    quantity_to_value,
)

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
    if hasattr(temp_code, "to_value"):
        temp_value = float(temp.to_value(unyt.K))
    elif code_units is not None:
        temp_value = float(np.asarray(temp, dtype=float)) * code_unit_scales(code_units)["temperature_cgs_K"]
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
        coord_value = np.asarray(coordinate, dtype=float) * code_unit_scales(code_units)["length_cgs_cm"]
    else:
        coord_value = np.asarray(coordinate, dtype=float)
    if hasattr(reference_radius, "to_value"):
        reference_radius_value = reference_radius.to_value(unyt.cm)
    elif code_units is not None:
        reference_radius_value = np.asarray(reference_radius, dtype=float) * code_unit_scales(code_units)["length_cgs_cm"]
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
        rho_value = np.asarray(rho_ref, dtype=float) * code_unit_scales(code_units)["density_cgs_g_cm3"]
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
        softening = np.asarray(softening, dtype=float) * code_unit_scales(code_units)["length_cgs_cm"]
    else:
        softening = float(softening)

    def _acceleration(coordinate):
        if hasattr(coordinate, "to_value"):
            radius = coordinate.to_value(unyt.cm)
        elif code_units is not None:
            radius = np.asarray(coordinate, dtype=float) * code_unit_scales(code_units)["length_cgs_cm"]
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
        self.par.CodeUnits = code_units
        self.par.units = SimpleNamespace(CodeUnits=code_units)

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']
        self.par.mesh = SimpleNamespace(grid_cells=self.par.nogrid, ghost_cells=2)
        self.par.simulation = SimpleNamespace(
            coordinate_system='spherical',
            current_time=self.par.time,
            box_size=self.par.boxsize,
        )

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

        self.fluid.temp_code = np.ones(self.par.nogrid) * icparams['tempini']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
        self.fluid.vel_code = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.rho_code = point_mass_hydrostatic_density_profile(
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
    code_units_mapping = runparams.get('CodeUnits')
    code_units_obj = CodeUnits.from_mapping(code_units_mapping) if code_units_mapping is not None else None
    rout = Simwrap(icparams, code_units=code_units_obj)
    if code_units_obj is not None:
        rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    color = kwargs.get('color', 'C0')
    nghost = int(runparams.get('noghost', 0))
    xall = spherical_cell_centers(rout.mesh.boundary)
    if nghost > 0:
        xcoord = xall[nghost:-nghost]
        rho_num = rout.fluid.rho_code[nghost:-nghost]
        vel_code_num = rout.fluid.vel_code[nghost:-nghost]
    else:
        xcoord = xall
        rho_num = rout.fluid.rho_code
        vel_code_num = rout.fluid.vel_code
    rho_analytic = point_mass_hydrostatic_density_profile(
        xcoord,
        icparams['rho_ref'],
        icparams['tempini'],
        icparams['muini'],
        icparams['point_mass'],
        reference_radius=xcoord[0],
        code_units=code_units_obj,
    )
    zero_velocity = np.zeros(len(xcoord)) * unyt.cm / unyt.s
    x_units = getattr(xcoord, 'units', code_units_obj.length_unit.units if code_units_obj is not None else unyt.cm)
    rho_units = getattr(rho_num, 'units', code_units_obj.density_unit.units if code_units_obj is not None else unyt.g / unyt.cm**3)
    vel_units = getattr(vel_num, 'units', code_units_obj.velocity_unit.units if code_units_obj is not None else unyt.cm / unyt.s)
    xplot = code_quantity_to_cgs(xcoord, code_units_obj, 'length_cgs_cm')
    rho_num_plot = code_quantity_to_cgs(rho_num, code_units_obj, 'density_cgs_g_cm3')
    rho_analytic_plot = quantity_to_value(rho_analytic, unyt.g / unyt.cm**3)
    vel_num_plot = code_quantity_to_cgs(vel_num, code_units_obj, 'velocity_cgs_cm_s')
    zero_velocity_plot = quantity_to_value(zero_velocity, unyt.cm / unyt.s)

    plt.subplot(1, 2, 1)
    plt.plot(xplot, rho_num_plot, **kwargs)
    plt.plot(
        xplot,
        rho_analytic_plot,
        ls='dashed',
        color=color,
    )
    plt.xlabel(rf"$r \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$\rho \; [{rho_units.latex_repr}]$")

    plt.subplot(1, 2, 2)
    plt.plot(xplot, vel_num_plot, **kwargs)
    plt.plot(
        xplot,
        zero_velocity_plot,
        ls='dashed',
        color=color,
    )
    plt.xlabel(rf"$r \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$v \; [{vel_units.latex_repr}]$")
