"""Analytic setup and diagnostics for the pressure-supported core test."""

import numpy as np
import unyt

from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    GRAVITATIONAL_CONSTANT_CGS,
    PROTON_MASS_CGS,
)
from radhydropy.units import CodeUnits, code_unit_scales, quantity_to_value
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
from radhydropy.rsim import Rsim


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
    radius_unyt, rho_ref_unyt, temperature_unyt, mu_dimensionless,
    point_mass_unyt, reference_radius_unyt,
):
    """Exact isothermal hydrostatic density around a point mass."""
    radius_cgs_cm = np.asarray(radius_unyt.to_value(unyt.cm), dtype=float)
    ref_cm = float(reference_radius_unyt.to_value(unyt.cm))
    rho_ref_cgs = float(rho_ref_unyt.to_value(unyt.g / unyt.cm**3))
    mass_g = float(point_mass_unyt.to_value(unyt.g))
    sound_speed_squared = (
        BOLTZMANN_CONSTANT_CGS * float(temperature_unyt.to_value(unyt.K))
        / (float(mu_dimensionless) * PROTON_MASS_CGS)
    )
    potential_difference = (
        -GRAVITATIONAL_CONSTANT_CGS * mass_g / radius_cgs_cm
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
    result = Rsim(config["par"])
    result.par.mesh.grid_cells = grid_cells
    result.par.simulation.coordinate_system = "spherical"
    result.par.simulation.time_proper_code = 0.0
    result.par.simulation.box_size_proper_code = np.asarray(
            [float(initial_condition["radius_outer_proper"].to_value(code_unit_system.length_unit))]
    )

    result.mesh.boundary_proper_code = np.linspace(
            float(initial_condition["radius_inner_proper"].to_value(code_unit_system.length_unit)),
            float(initial_condition["radius_outer_proper"].to_value(code_unit_system.length_unit)),
            grid_cells + 1,
    )
    result.mesh.x_proper_code = spherical_cell_centers(result.mesh.boundary_proper_code)
    result.mesh.area_proper_code = 4.0 * np.pi * result.mesh.boundary_proper_code[:-1] ** 2
    result.mesh.volume_proper_code = (
            (result.mesh.boundary_proper_code[1:] ** 3 - result.mesh.boundary_proper_code[:-1] ** 3)
            * 4.0 * np.pi / 3.0
    )
    result.fluid.rho_proper_code = point_mass_density(
            result.mesh.x_proper_code * code_unit_system.length_unit,
            initial_condition["rho_reference_proper"],
            initial_condition["temperature_proper"],
            initial_condition["mean_molecular_weight"],
            initial_condition["point_mass"],
            result.mesh.x_proper_code[0] * code_unit_system.length_unit,
    )
    scales = code_unit_scales(code_unit_system)
    result.fluid.rho_proper_code = quantity_to_value(
        result.fluid.rho_proper_code, code_unit_system.density_unit
    )
    result.fluid.temp_proper_code = np.full(
            grid_cells,
            float(initial_condition["temperature_proper"].to_value(unyt.K))
            / scales["temperature_cgs_K"],
    )
    result.fluid.mu = np.full(grid_cells, float(initial_condition["mean_molecular_weight"]))
    result.fluid.vel_proper_code = np.zeros(grid_cells)
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=result.mesh.x_proper_code,
        boundary_proper_code=result.mesh.boundary_proper_code,
        width_proper_code=np.diff(result.mesh.boundary_proper_code),
        area_proper_code=result.mesh.area_proper_code,
        volume_proper_code=result.mesh.volume_proper_code,
    )
    result.fluid.pre_proper_code = result.fluid.rho_proper_code * result.fluid.temp_proper_code
    result.fluid.time_proper_code = 0.0
    result.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    result.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS, rho_proper_code=result.fluid.rho_proper_code,
        vel_proper_code=result.fluid.vel_proper_code, pre_proper_code=result.fluid.pre_proper_code,
        temp_proper_code=result.fluid.temp_proper_code, time_proper_code=0.0, mu_dimensionless=result.fluid.mu,
    )
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    return result


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
