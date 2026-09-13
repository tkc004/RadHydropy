"""Helper utilities for the spherical hydrostatic point-mass example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, GRAVITATIONAL_CONSTANT_CGS, PROTON_MASS_CGS
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import (
    CodeUnits,
    code_quantity_to_cgs,
    code_unit_scales,
    quantity_to_value,
)

SPEED_SQUARED_UNIT = unyt.cm**2 / unyt.s**2
DENSITY_UNIT = unyt.g / unyt.cm**3
ACCELERATION_UNIT = unyt.cm / unyt.s**2


def sound_speed_squared(temperature_proper_code, mu, code_unit_system=None):
    """Return the isothermal sound speed squared."""
    if hasattr(temperature_proper_code, "to_value"):
        temp_value = float(temperature_proper_code.to_value(unyt.K))
    elif code_unit_system is not None:
        temp_value = float(np.asarray(temperature_proper_code, dtype=float)) * code_unit_scales(code_unit_system)["temperature_cgs_K"]
    else:
        raise TypeError("temperature_proper_code requires a unit or code-unit system")
    mu_value = float(np.asarray(mu, dtype=float))
    return (
        BOLTZMANN_CONSTANT_CGS
        * temp_value
        / (mu_value * PROTON_MASS_CGS)
    ) * SPEED_SQUARED_UNIT


def spherical_cell_centers(boundary_proper_code):
    """Return spherical cell centers consistent with the mesh geometry."""
    x_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    vol_denom = boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3
    nonzero = vol_denom != 0.0
    x_proper_code[nonzero] = 0.75 * (
        boundary_proper_code[1:][nonzero]**4 - boundary_proper_code[:-1][nonzero]**4
    ) / vol_denom[nonzero]
    return x_proper_code


def point_mass_hydrostatic_density_profile(
    x_proper_code,
    rho_reference_proper,
    temperature_proper_code,
    mu,
    point_mass,
    reference_radius,
    code_unit_system=None,
):
    """Return the exact isothermal hydrostatic density profile."""
    c_s2 = sound_speed_squared(temperature_proper_code, mu, code_unit_system=code_unit_system)
    c_s2_value = c_s2.to_value(unyt.cm**2 / unyt.s**2)
    if hasattr(x_proper_code, "to_value"):
        coord_value = x_proper_code.to_value(unyt.cm)
    elif code_unit_system is not None:
        coord_value = np.asarray(x_proper_code, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
    else:
        coord_value = np.asarray(x_proper_code, dtype=float)
    if hasattr(reference_radius, "to_value"):
        reference_radius_value = reference_radius.to_value(unyt.cm)
    elif code_unit_system is not None:
        reference_radius_value = np.asarray(reference_radius, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
    else:
        reference_radius_value = float(reference_radius)
    if hasattr(point_mass, "to_value"):
        point_mass_value = point_mass.to_value(unyt.g)
    elif code_unit_system is not None:
        point_mass_value = np.asarray(point_mass, dtype=float) * code_unit_scales(code_unit_system)["mass_g"]
    else:
        point_mass_value = float(point_mass)
    if hasattr(rho_reference_proper, "to_value"):
        rho_value = rho_reference_proper.to_value(unyt.g / unyt.cm**3)
    elif code_unit_system is not None:
        rho_value = np.asarray(rho_reference_proper, dtype=float) * code_unit_scales(code_unit_system)["density_cgs_g_cm3"]
    else:
        rho_value = float(rho_reference_proper)
    phi_ref = -GRAVITATIONAL_CONSTANT_CGS * point_mass_value / reference_radius_value
    phi = -GRAVITATIONAL_CONSTANT_CGS * point_mass_value / coord_value
    exponent = -(phi - phi_ref) / c_s2_value
    return rho_value * np.exp(exponent) * DENSITY_UNIT


def point_mass_acceleration(
    point_mass_proper_g, softening_proper_unyt=0.0, code_unit_system=None
):
    """Return a callable for a point-mass gravitational acceleration field."""
    if hasattr(point_mass_proper_g, "to_value"):
        point_mass_proper_g = point_mass_proper_g.to_value(unyt.g)
    elif code_unit_system is not None:
        point_mass_proper_g = np.asarray(point_mass_proper_g, dtype=float) * code_unit_scales(code_unit_system)["mass_g"]
    else:
        point_mass_proper_g = float(point_mass_proper_g)
    if hasattr(softening_proper_unyt, "to_value"):
        softening_proper_cgs_cm = softening_proper_unyt.to_value(unyt.cm)
    elif code_unit_system is not None:
        softening_proper_cgs_cm = np.asarray(softening_proper_unyt, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
    else:
        softening_proper_cgs_cm = float(softening_proper_unyt)

    def _acceleration(x_proper_code):
        if hasattr(x_proper_code, "to_value"):
            radius_proper_cgs_cm = x_proper_code.to_value(unyt.cm)
        elif code_unit_system is not None:
            radius_proper_cgs_cm = np.asarray(x_proper_code, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
        else:
            radius_proper_cgs_cm = np.asarray(x_proper_code, dtype=float)
        radius_proper_cgs_cm = np.maximum(radius_proper_cgs_cm, softening_proper_cgs_cm)
        return (
            -GRAVITATIONAL_CONSTANT_CGS * point_mass_proper_g / radius_proper_cgs_cm**2
        ) * ACCELERATION_UNIT

    return _acceleration


def build_initial_condition(config):
    code_unit_system = CodeUnits.from_mapping(
        config['par']['units']['CodeUnits']
    )
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    boundary_proper_unyt = np.linspace(
        initial_condition['radius_inner_proper'],
        initial_condition['radius_outer_proper'], grid_cells + 1,
    )
    coordinate_proper_unyt = spherical_cell_centers(
        quantity_to_value(boundary_proper_unyt, code_unit_system.length_unit)
    ) * code_unit_system.length_unit
    rho_proper_unyt = point_mass_hydrostatic_density_profile(
        coordinate_proper_unyt,
        initial_condition['rho_reference_proper'],
        initial_condition['temperature_proper'],
        initial_condition['mean_molecular_weight'],
        initial_condition['point_mass'],
        reference_radius=coordinate_proper_unyt[0],
        code_unit_system=code_unit_system,
    )
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_unit_system,
        ic_config=initial_condition,
    )
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(coordinate_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_unit_system.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial_condition['temperature_proper'])
    writer.simulation.fluid.mu = np.full(grid_cells, initial_condition['mean_molecular_weight'])
    return writer
def plot_snapshot(outfilename, config, **kwargs):
    """Read a snapshot and compare it with the analytic hydrostatic profile."""
    code_units_mapping = config['par']['units']['CodeUnits']
    code_units_obj = CodeUnits.from_mapping(code_units_mapping) if code_units_mapping is not None else None
    rout = rio.loadhdf5(config, outfilename)
    color = kwargs.get('color', 'C0')
    nghost = int(config['par']['mesh']['ghost_cells'])
    boundary_proper_code = np.asarray(rout.mesh.boundary_radarray.value, dtype=float)
    xall = spherical_cell_centers(boundary_proper_code)
    if nghost > 0:
        # Mesh geometry contains physical cells; only fluid arrays carry
        # ghost cells after HDF5 reload.
        coordinate_proper_code = xall[nghost:-nghost]
        rho_values = np.asarray(rout.fluid.rho_radarray.value, dtype=float)
        vel_values = np.asarray(rout.fluid.vel_radarray.value, dtype=float)
        rho_num = rho_values[nghost:-nghost] if rho_values.size != int(config['par']['mesh']['grid_cells']) else rho_values
        vel_code_num = vel_values[nghost:-nghost] if vel_values.size != int(config['par']['mesh']['grid_cells']) else vel_values
    else:
        coordinate_proper_code = xall
        rho_num = np.asarray(rout.fluid.rho_radarray.value, dtype=float)
        vel_code_num = np.asarray(rout.fluid.vel_radarray.value, dtype=float)
    rho_analytic = point_mass_hydrostatic_density_profile(
        coordinate_proper_code,
        config['initial_condition']['rho_reference_proper'],
        config['initial_condition']['temperature_proper'],
        config['initial_condition']['mean_molecular_weight'],
        config['initial_condition']['point_mass'],
        reference_radius=coordinate_proper_code[0],
        code_unit_system=code_units_obj,
    )
    zero_velocity = np.zeros(len(coordinate_proper_code)) * unyt.cm / unyt.s
    x_units = getattr(coordinate_proper_code, 'units', code_units_obj.length_unit.units if code_units_obj is not None else unyt.cm)
    rho_units = getattr(rho_num, 'units', code_units_obj.density_unit.units if code_units_obj is not None else unyt.g / unyt.cm**3)
    vel_units = getattr(vel_code_num, 'units', code_units_obj.velocity_unit.units if code_units_obj is not None else unyt.cm / unyt.s)
    xplot = code_quantity_to_cgs(coordinate_proper_code, code_units_obj, 'length_cgs_cm')
    rho_num_plot = code_quantity_to_cgs(rho_num, code_units_obj, 'density_cgs_g_cm3')
    rho_analytic_plot = quantity_to_value(rho_analytic, unyt.g / unyt.cm**3)
    vel_num_plot = code_quantity_to_cgs(vel_code_num, code_units_obj, 'velocity_cgs_cm_s')
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
