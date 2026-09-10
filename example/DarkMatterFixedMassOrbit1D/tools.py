"""Helpers for the fixed-enclosed-mass dark-matter orbit benchmark."""

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits, quantity_to_value


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])


def make_shell(config):
    initial_condition = config['initial_condition']
    code_unit_system = code_units_from_config(config)
    vel_proper_code = quantity_to_value(
        initial_condition['vel_proper'], code_unit_system.velocity_unit
    )
    return DarkMatterShells(
        radius=[initial_condition['radius_initial_orbit_dimensionless']],
        velocity=[vel_proper_code],
        mass=[initial_condition['shell_mass_dimensionless']],
        angular_momentum=[initial_condition['specific_angular_momentum_dimensionless']],
        softening=initial_condition['softening_dimensionless'],
        fixed_enclosed_mass=initial_condition['central_mass_dimensionless'],
        code_units=code_unit_system,
    )
