"""Helper utilities for the hydrostatic-equilibrium check example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
import radhydropy.io as rio
from radhydropy.units import (
    CodeUnits,
    code_quantity_to_cgs,
    code_unit_scales,
    quantity_to_value,
)
from basic_hydro_utils import make_initial_condition

SPEED_SQUARED_UNIT = unyt.cm**2 / unyt.s**2
DENSITY_UNIT = unyt.g / unyt.cm**3
ACCELERATION_UNIT = unyt.cm / unyt.s**2


def _physical_value(value, unit, name):
    """Convert a physical IC quantity only at the IC/code-unit boundary."""
    if not hasattr(value, 'to_value'):
        raise TypeError('%s must be a unit-bearing physical quantity' % name)
    try:
        return value.to_value(unit)
    except (TypeError, ValueError) as error:
        raise ValueError(
            '%s must have units equivalent to %s' % (name, unit)
        ) from error


def sound_speed_squared(temperature_proper_code, mu, code_unit_system=None):
    """Return the isothermal sound speed squared."""
    if hasattr(temperature_proper_code, "to_value"):
        temp_value = float(temperature_proper_code.to_value(unyt.K))
    elif code_unit_system is not None:
        temp_value = float(np.asarray(temperature_proper_code, dtype=float)) * code_unit_scales(code_unit_system)["temperature_cgs_K"]
    else:
        raise TypeError('temperature_proper_code must be unit-bearing or paired with code_unit_system')
    mu_value = float(np.asarray(mu, dtype=float))
    return (
        BOLTZMANN_CONSTANT_CGS
        * temp_value
        / (mu_value * PROTON_MASS_CGS)
    ) * SPEED_SQUARED_UNIT


def hydrostatic_density_profile(
    coordinate_proper_code,
    rho_reference_proper_unyt,
    temperature_proper_code,
    mu,
    gravity_strength,
    code_unit_system=None,
):
    """Return the exact isothermal hydrostatic density profile."""
    c_s2 = sound_speed_squared(temperature_proper_code, mu, code_unit_system=code_unit_system)
    c_s2_value = c_s2.to_value(unyt.cm**2 / unyt.s**2)
    if hasattr(coordinate_proper_code, "to_value"):
        coord_value = coordinate_proper_code.to_value(unyt.cm)
    elif code_unit_system is not None:
        coord_value = np.asarray(coordinate_proper_code, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
    else:
        raise TypeError('coordinate_proper_code must be unit-bearing or paired with code_unit_system')
    if hasattr(rho_reference_proper_unyt, "to_value"):
        rho_value = rho_reference_proper_unyt.to_value(unyt.g / unyt.cm**3)
    elif code_unit_system is not None:
        rho_value = np.asarray(rho_reference_proper_unyt, dtype=float) * code_unit_scales(code_unit_system)["density_cgs_g_cm3"]
    else:
        raise TypeError('rho_reference_proper_unyt must be unit-bearing or paired with code_unit_system')
    if hasattr(gravity_strength, "to_value"):
        gravity_value = gravity_strength.to_value(unyt.cm / unyt.s**2)
    elif code_unit_system is not None:
        gravity_value = np.asarray(gravity_strength, dtype=float) * code_unit_scales(code_unit_system)["acceleration_cgs_cm_s2"]
    else:
        raise TypeError('gravity_strength must be unit-bearing or paired with code_unit_system')
    scale_height = c_s2_value / gravity_value
    profile = rho_value * np.exp(-np.asarray(coord_value, dtype=float) / scale_height)
    return profile * DENSITY_UNIT


def constant_gravity_acceleration(gravity_strength, code_unit_system=None):
    """Return a callable for a uniform downward acceleration field."""
    if hasattr(gravity_strength, "to_value"):
        gravity_strength = gravity_strength.to_value(unyt.cm / unyt.s**2)
    elif code_unit_system is not None:
        gravity_strength = np.asarray(gravity_strength, dtype=float) * code_unit_scales(code_unit_system)["acceleration_cgs_cm_s2"]
    else:
        gravity_strength = float(gravity_strength)
    gravity_strength = float(np.asarray(gravity_strength, dtype=float))
    scale = ACCELERATION_UNIT

    def _acceleration(coordinate):
        return -gravity_strength * np.ones(np.shape(coordinate), dtype=float) * scale

    return _acceleration


def build_initial_condition(config):
    initial_condition = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    box_size_proper_unyt = _physical_value(
        initial_condition['box_size_proper'], unyt.cm, 'box_size_proper'
    ) * unyt.cm
    boundary_proper_code = np.linspace(0.0, 1.0, grid_cells + 1) * quantity_to_value(box_size_proper_unyt, code_units.length_unit)
    coordinate_proper_code = 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    rho_proper_code = quantity_to_value(hydrostatic_density_profile(
        coordinate_proper_code * code_units.length_unit,
        initial_condition['rho_reference_proper'],
        initial_condition['temperature_proper'],
        initial_condition['mean_molecular_weight'],
        initial_condition['gravity_strength'],
        code_unit_system=code_units,
    ), code_units.density_unit)
    temp_proper_code = np.full(grid_cells, quantity_to_value(initial_condition['temperature_proper'], code_units.temperature_unit))
    return make_initial_condition(
        config,
        boundary_proper_code=boundary_proper_code,
        rho_proper_code=rho_proper_code,
        vel_proper_code=np.zeros(grid_cells),
        temp_proper_code=temp_proper_code,
        mu_dimensionless=np.full(grid_cells, initial_condition['mean_molecular_weight']),
        area_proper_code=np.ones(grid_cells),
    )
def plot_snapshot(outfilename, config, **kwargs):
    """Read a snapshot and compare it with the analytic hydrostatic profile."""
    initial_condition = config['initial_condition']

    code_units_mapping = config["par"].get('units', {}).get('CodeUnits')
    code_units_obj = (
        CodeUnits.from_mapping(code_units_mapping)
        if code_units_mapping is not None
        else CodeUnits.from_mapping({
            'UnitMass_in_cgs': 1.0,
            'UnitLength_in_cgs': 1.0,
            'UnitVelocity_in_cgs': 1.0,
            'UnitCurrent_in_cgs': 1.0,
            'UnitTemp_in_cgs': 1.0,
        })
    )
    nested_config = dict(config)
    nested_config['_code_units'] = code_units_obj
    rout = build_initial_condition(nested_config)
    if code_units_obj is not None:
        rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    color = kwargs.get('color', 'C0')
    nghost = int(config["par"].get('mesh', {}).get('ghost_cells', 0))
    boundary_proper_code = rout.mesh.geometry_state.boundary_proper_code
    x_proper_code_all = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    if nghost > 0:
        # The typed mesh geometry stores physical cell boundaries; ghost
        # cells are present only in the fluid arrays returned by the reader.
        x_proper_code = x_proper_code_all
        rho_proper_code = rout.fluid.rho_proper_code[nghost:-nghost]
        vel_proper_code = rout.fluid.vel_proper_code[nghost:-nghost]
    else:
        x_proper_code = x_proper_code_all
        rho_proper_code = rout.fluid.rho_proper_code
        vel_proper_code = rout.fluid.vel_proper_code
    rho_analytic_cgs_g_cm3_unyt = hydrostatic_density_profile(
        x_proper_code,
        initial_condition['rho_reference_proper'],
        initial_condition['temperature_proper'],
        initial_condition['mean_molecular_weight'],
        initial_condition['gravity_strength'],
        code_unit_system=code_units_obj,
    )
    if code_units_obj is not None:
        x_units = getattr(x_proper_code, 'units', code_units_obj.length_unit.units)
        rho_units = getattr(rho_proper_code, 'units', code_units_obj.density_unit.units)
        vel_units = getattr(vel_proper_code, 'units', code_units_obj.velocity_unit.units)
        x_proper_cgs_cm_unyt = code_quantity_to_cgs(
            x_proper_code, code_units_obj, 'length_cgs_cm'
        ) * unyt.cm
        rho_proper_cgs_g_cm3_unyt = (
            code_quantity_to_cgs(rho_proper_code, code_units_obj, 'density_cgs_g_cm3')
            * (unyt.g / unyt.cm**3)
        )
        vel_proper_cgs_cm_s_unyt = (
            code_quantity_to_cgs(vel_proper_code, code_units_obj, 'velocity_cgs_cm_s')
            * (unyt.cm / unyt.s)
        )
    else:
        x_units = unyt.cm
        rho_units = unyt.g / unyt.cm**3
        vel_units = unyt.cm / unyt.s
        x_proper_cgs_cm_unyt = x_proper_code.to(unyt.cm)
        rho_proper_cgs_g_cm3_unyt = rho_proper_code.to(unyt.g / unyt.cm**3)
        vel_proper_cgs_cm_s_unyt = vel_proper_code.to(unyt.cm / unyt.s)
    rho_analytic_cgs_g_cm3_unyt = rho_analytic_cgs_g_cm3_unyt.to(
        unyt.g / unyt.cm**3
    )
    zero_vel_proper_cgs_cm_s_unyt = np.zeros(len(x_proper_code)) * unyt.cm / unyt.s
    zero_vel_proper_cgs_cm_s_unyt = zero_vel_proper_cgs_cm_s_unyt.to(unyt.cm / unyt.s)

    plt.subplot(1, 2, 1)
    plt.plot(x_proper_cgs_cm_unyt, rho_proper_cgs_g_cm3_unyt, **kwargs)
    plt.plot(
        x_proper_cgs_cm_unyt,
        rho_analytic_cgs_g_cm3_unyt,
        ls='dashed',
        color=color,
    )
    plt.xlabel(rf"$x \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$\rho \; [{rho_units.latex_repr}]$")

    plt.subplot(1, 2, 2)
    plt.plot(x_proper_cgs_cm_unyt, vel_proper_cgs_cm_s_unyt, **kwargs)
    plt.plot(
        x_proper_cgs_cm_unyt,
        zero_vel_proper_cgs_cm_s_unyt,
        ls='dashed',
        color=color,
    )
    plt.xlabel(rf"$x \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$v \; [{vel_units.latex_repr}]$")
