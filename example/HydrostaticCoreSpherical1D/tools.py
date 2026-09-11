"""Analytic setup and diagnostics for the pressure-supported core test."""

import numpy as np
import unyt

from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    GRAVITATIONAL_CONSTANT_CGS,
    PROTON_MASS_CGS,
)
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.initial_condition_writer import InitialConditionWriter


def spherical_cell_centers(boundary_proper_code):
    boundary_proper_code = np.asarray(boundary_proper_code, dtype=float)
    denominator = boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
    center = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    valid = denominator != 0.0
    center[valid] = 0.75 * (
        boundary_proper_code[1:][valid] ** 4 - boundary_proper_code[:-1][valid] ** 4
    ) / denominator[valid]
    return center


def point_mass_density(
    radius_proper_unyt, rho_reference_proper_unyt, temperature_proper_unyt, mu_dimensionless,
    point_mass_unyt, reference_radius_unyt,
):
    """Exact isothermal hydrostatic density around a point mass."""
    radius_proper_cgs_cm = np.asarray(radius_proper_unyt.to_value(unyt.cm), dtype=float)
    ref_cm = float(reference_radius_unyt.to_value(unyt.cm))
    rho_ref_cgs = float(rho_reference_proper_unyt.to_value(unyt.g / unyt.cm**3))
    mass_g = float(point_mass_unyt.to_value(unyt.g))
    sound_speed_squared = (
        BOLTZMANN_CONSTANT_CGS * float(temperature_proper_unyt.to_value(unyt.K))
        / (float(mu_dimensionless) * PROTON_MASS_CGS)
    )
    potential_difference = (
        -GRAVITATIONAL_CONSTANT_CGS * mass_g / radius_proper_cgs_cm
        + GRAVITATIONAL_CONSTANT_CGS * mass_g / ref_cm
    )
    return rho_ref_cgs * np.exp(-potential_difference / sound_speed_squared) * (
        unyt.g / unyt.cm**3
    )


def build_initial_condition(config):
    """Build the typed proper-code IC from the complete nested config."""
    code_unit_system = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    initial_condition = config["initial_condition"]
    grid_cells = int(config["par"]["mesh"]["grid_cells"])
    boundary_proper_unyt = np.linspace(
            initial_condition["radius_inner_proper"],
            initial_condition["radius_outer_proper"], grid_cells + 1,
    )
    coordinate_proper_unyt = spherical_cell_centers(
        quantity_to_value(boundary_proper_unyt, code_unit_system.length_unit)
    ) * code_unit_system.length_unit
    rho_proper_unyt = point_mass_density(
            coordinate_proper_unyt,
            initial_condition["rho_reference_proper"],
            initial_condition["temperature_proper"],
            initial_condition["mean_molecular_weight"],
            initial_condition["point_mass"],
            coordinate_proper_unyt[0],
    )
    writer = InitialConditionWriter(par_config=config["par"], code_units=code_unit_system)
    writer.box_size = writer.radquantity(initial_condition["radius_outer_proper"])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(coordinate_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_unit_system.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial_condition["temperature_proper"])
    writer.simulation.fluid.mu = np.full(grid_cells, float(initial_condition["mean_molecular_weight"]))
    return writer


def analytic_density_code(radius_code, config):
    code_unit_system = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    initial_condition = config["initial_condition"]
    rho_proper_unyt = point_mass_density(
        np.asarray(radius_code) * code_unit_system.length_unit,
        initial_condition["rho_reference_proper"],
        initial_condition["temperature_proper"],
        initial_condition["mean_molecular_weight"],
        initial_condition["point_mass"],
        float(radius_code[0]) * code_unit_system.length_unit,
    )
    return quantity_to_value(rho_proper_unyt, code_unit_system.density_unit)
