"""Helper utilities for the spherical ballistic-infall example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
import radhydropy.io as rio
from radhydropy.units import (
    CodeUnits,
    code_quantity_to_cgs,
    code_unit_scales,
    quantity_to_value,
    time_seconds,
)


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
            else float(softening) * scales["length_cgs_cm"]
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
            radius = radius * scales["length_cgs_cm"]
        radius = np.maximum(radius, softening)
        return (
            -GRAVITATIONAL_CONSTANT_CGS * point_mass / radius**2
        ) * accel_unit

    return _acceleration


def ballistic_density_profile(coordinate, rho_ref):
    """Return a constant-density profile for the ballistic infall example."""
    return np.ones(np.shape(coordinate), dtype=float) * rho_ref


def ballistic_velocity_profile(
    coordinate,
    point_mass,
    time,
    softening=0.0,
    code_units=None,
):
    """Return the short-time free-fall velocity under a point mass."""
    acceleration = point_mass_acceleration(
        point_mass,
        softening=softening,
        code_units=code_units,
    )(coordinate)
    if code_units is not None:
        time = time_seconds(time, code_units) * unyt.s
    elif hasattr(time, 'to_value'):
        time = time.to_value(unyt.s) * unyt.s
    else:
        time = float(time) * unyt.s
    return acceleration * time


def build_initial_condition(config):
    icparams = config['initial_condition']
    code_units = config['_code_units']
    sim = SimpleNamespace()
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)

    grid_cells = int(icparams['grid_cells'])
    coordinate_system = icparams['coordinate_system']
    box_size = np.ones(1) * icparams['box_size']
    sim.par.time = np.ones(1) * icparams['current_time']
    sim.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
    sim.par.simulation = SimpleNamespace(
        coordinate_system=coordinate_system,
        current_time=sim.par.time,
        box_size=box_size,
    )

    sim.mesh.boundary = np.linspace(
        icparams['inner_radius'],
        icparams['outer_radius'],
        grid_cells + 1,
    )
    sim.mesh.coordinate = spherical_cell_centers(sim.mesh.boundary)
    sim.mesh.area = 4.0 * np.pi * sim.mesh.boundary[:-1]**2
    sim.mesh.vol = (
        np.absolute(sim.mesh.boundary[1:]**3 - sim.mesh.boundary[:-1]**3)
        * 4.0
        * np.pi
        / 3.0
    )

    sim.fluid.temp_code = np.ones(grid_cells) * icparams['initial_temperature']
    sim.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']
    sim.fluid.vel_code = np.zeros(grid_cells) * unyt.cm / unyt.s
    sim.fluid.rho_code = ballistic_density_profile(
        sim.mesh.coordinate,
        icparams['reference_density'],
    )


    return sim
def ReadandPlot(outfilename, config, **kwargs):
    """Read a snapshot and compare it with the ballistic short-time profile."""
    icparams = config['initial_condition']
    runparams = config['par']
    code_units_obj = config['_code_units']
    rout = build_initial_condition(config)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    color = kwargs.get('color', 'C0')
    nghost = int(runparams.get('mesh', {}).get('ghost_cells', 0))
    xall = spherical_cell_centers(rout.mesh.boundary)
    if nghost > 0:
        xcoord = xall[nghost:-nghost]
        rho_num = rout.fluid.rho_code[nghost:-nghost]
        vel_code_num = rout.fluid.vel_code[nghost:-nghost]
    else:
        xcoord = xall
        rho_num = rout.fluid.rho_code
        vel_code_num = rout.fluid.vel_code
    rho_analytic = ballistic_density_profile(xcoord, icparams['reference_density'])
    vel_analytic = ballistic_velocity_profile(
        xcoord,
        icparams['point_mass'],
        rout.fluid.time,
        code_units=code_units_obj,
    )
    zero_velocity = np.zeros(len(xcoord)) * unyt.cm / unyt.s
    x_units = getattr(xcoord, 'units', code_units_obj.length_unit.units)
    rho_units = getattr(rho_num, 'units', code_units_obj.density_unit.units)
    vel_units = getattr(vel_code_num, 'units', code_units_obj.velocity_unit.units)
    xcoord_cgs = code_quantity_to_cgs(xcoord, code_units_obj, 'length_cgs_cm')
    rho_num_cgs = code_quantity_to_cgs(rho_num, code_units_obj, 'density_cgs_g_cm3')
    rho_analytic_cgs = quantity_to_value(rho_analytic, unyt.g / unyt.cm**3)
    vel_num_cgs = code_quantity_to_cgs(vel_code_num, code_units_obj, 'velocity_cgs_cm_s')
    vel_analytic_cgs = quantity_to_value(vel_analytic, unyt.cm / unyt.s)
    zero_velocity_cgs = quantity_to_value(zero_velocity, unyt.cm / unyt.s)

    plt.subplot(1, 2, 1)
    plt.plot(xcoord_cgs, rho_num_cgs, label='numerical', **kwargs)
    plt.plot(
        xcoord_cgs,
        rho_analytic_cgs,
        ls='dashed',
        color=color,
        label='analytic',
    )
    plt.xlabel(rf"$r \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$\rho \; [{rho_units.latex_repr}]$")
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
    plt.xlabel(rf"$r \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$v \; [{vel_units.latex_repr}]$")
    plt.legend(loc='best')
