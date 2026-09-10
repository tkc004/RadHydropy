"""Analytic combined gas/dark-matter background helpers."""

import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits, quantity_to_value


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])


def make_shell(config):
    initial_condition = config['initial_condition']
    code_unit_system = code_units_from_config(config)
    central_mass_dimensionless = float(initial_condition['central_dark_matter_mass_dimensionless'])
    gas_density_dimensionless = float(initial_condition['uniform_gas_density_dimensionless'])
    vel_proper_code = quantity_to_value(
        initial_condition['vel_proper'], code_unit_system.velocity_unit
    )

    def enclosed_mass(radius_code):
        radius_code = np.asarray(radius_code, dtype=float)
        return central_mass_dimensionless + 4.0 * np.pi / 3.0 * gas_density_dimensionless * radius_code**3

    return DarkMatterShells(
        radius=[initial_condition['radius_initial_orbit_dimensionless']],
        velocity=[vel_proper_code],
        mass=[initial_condition['shell_mass_dimensionless']],
        angular_momentum=[initial_condition['specific_angular_momentum_dimensionless']],
        softening=initial_condition['softening_dimensionless'],
        fixed_enclosed_mass=enclosed_mass,
        code_units=code_unit_system,
    )


def enclosed_mass(radius_code, config):
    initial_condition = config['initial_condition']
    radius_code = np.asarray(radius_code, dtype=float)
    return (
        float(initial_condition['central_dark_matter_mass_dimensionless'])
        + 4.0 * np.pi / 3.0
        * float(initial_condition['uniform_gas_density_dimensionless']) * radius_code**3
    )
