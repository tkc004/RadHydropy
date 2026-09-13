"""Helpers for the Einstein--de Sitter dark-matter shell growth test."""

import numpy as np

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, _gravitational_constant_code, quantity_to_value


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])


def volume_midpoint_boundaries(
    radius_inner_comoving_code, radius_outer_comoving_code, number,
):
    boundaries = np.linspace(
        radius_inner_comoving_code**3, radius_outer_comoving_code**3, number + 1
    )
    return boundaries**(1.0 / 3.0)


def make_shells(config, overdensity=None):
    initial_condition = config['initial_condition']
    code_unit_system = code_units_from_config(config)
    cosmology_config = config['par']['cosmology']
    cosmology = EinsteinDeSitter.from_code_units(
        code_unit_system,
        t_ref=quantity_to_value(cosmology_config['cosmology_t_ref'], code_unit_system.time_unit),
        a_ref=float(cosmology_config['cosmology_a_ref']),
    )
    default_number = int(initial_condition.get('number_of_shells', 2))
    number_inner = int(initial_condition.get('number_of_inner_shells', default_number // 2))
    number_outer = int(initial_condition.get('number_of_outer_shells', number_inner))
    radius_inner_comoving_code = quantity_to_value(
        initial_condition['radius_inner_comoving'], code_unit_system.length_unit
    )
    radius_outer_comoving_code = quantity_to_value(
        initial_condition['radius_outer_comoving'], code_unit_system.length_unit
    )
    radius_perturbation_comoving_code = quantity_to_value(
        initial_condition['radius_perturbation_comoving'], code_unit_system.length_unit
    )
    inner_boundaries = volume_midpoint_boundaries(
        radius_inner_comoving_code, radius_perturbation_comoving_code, number_inner
    )
    outer_boundaries = volume_midpoint_boundaries(
        radius_perturbation_comoving_code, radius_outer_comoving_code, number_outer
    )
    boundaries = np.concatenate((inner_boundaries, outer_boundaries[1:]))
    radius_comoving_code = ((boundaries[:-1]**3 + boundaries[1:]**3) / 2.0)**(1.0 / 3.0)
    volume_comoving_code = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    time_cosmic_code = quantity_to_value(
        initial_condition['time_cosmic'], code_unit_system.time_unit
    )
    scale_factor_dimensionless = float(cosmology.scale_factor(time_cosmic_code))
    rho_comoving_code = float(cosmology.background_density(time_cosmic_code)) * scale_factor_dimensionless**3
    delta = float(initial_condition['overdensity'] if overdensity is None else overdensity)
    inside = radius_comoving_code < radius_perturbation_comoving_code
    mass_comoving_code = rho_comoving_code * volume_comoving_code * (1.0 + delta * inside)
    hubble_code = float(cosmology.hubble(time_cosmic_code))
    vel_supercomoving_code = np.zeros_like(radius_comoving_code)
    vel_supercomoving_code[inside] = (
        -scale_factor_dimensionless**2 * hubble_code * delta
        * radius_comoving_code[inside] / 3.0
    )
    vel_supercomoving_code[~inside] = (
        -scale_factor_dimensionless**2 * hubble_code * delta * radius_perturbation_comoving_code**3
        / (3.0 * radius_comoving_code[~inside]**2)
    )
    writer = InitialConditionWriter(
        ic_config=config["initial_condition"],
        par_config=config['par'],
        code_units=code_unit_system,
    )
    writer.simulation.par.dark_matter = DarkMatterShells(
        radius_comoving_code, vel_supercomoving_code, mass_comoving_code,
        softening=quantity_to_value(
            initial_condition['softening'], code_unit_system.length_unit
        ),
        code_units=code_unit_system,
    )
    return writer.simulation.par.dark_matter, boundaries


def lagrangian_boundary_acceleration(radius_comoving_code, enclosed_mass_comoving_code, rho_comoving_code,
                                     scale_factor, code_unit_system):
    g_code = _gravitational_constant_code(code_unit_system)
    background_mass_comoving_code = 4.0 * np.pi / 3.0 * rho_comoving_code * radius_comoving_code**3
    return -g_code * scale_factor * (enclosed_mass_comoving_code - background_mass_comoving_code) / radius_comoving_code**2


def overdensity_inside(radius_comoving_code, target_mass_comoving_code, rho_comoving_code):
    background_mass_comoving_code = 4.0 * np.pi / 3.0 * rho_comoving_code * radius_comoving_code**3
    return float(target_mass_comoving_code / background_mass_comoving_code - 1.0)


def step_lagrangian_boundary(radius_comoving_code, vel_supercomoving_code, dt_supercomoving_code,
                             enclosed_mass_comoving_code, rho_comoving_start_code,
                             rho_comoving_end_code, scale_factor_start,
                             scale_factor_end, code_unit_system):
    acceleration = lagrangian_boundary_acceleration(
        radius_comoving_code, enclosed_mass_comoving_code, rho_comoving_start_code,
        scale_factor_start,
        code_unit_system,
    )
    vel_supercomoving_half_code = (
        vel_supercomoving_code + 0.5 * dt_supercomoving_code * acceleration
    )
    radius_comoving_new_code = (
        radius_comoving_code + dt_supercomoving_code * vel_supercomoving_half_code
    )
    acceleration_new = lagrangian_boundary_acceleration(
        radius_comoving_new_code, enclosed_mass_comoving_code, rho_comoving_end_code,
        scale_factor_end,
        code_unit_system,
    )
    return (
        radius_comoving_new_code,
        vel_supercomoving_half_code + 0.5 * dt_supercomoving_code * acceleration_new,
    )
