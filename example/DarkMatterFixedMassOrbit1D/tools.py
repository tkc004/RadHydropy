"""Helpers for the fixed-enclosed-mass dark-matter orbit benchmark."""

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])


def make_shell(config):
    initial_condition = config['initial_condition']
    code_unit_system = code_units_from_config(config)
    return DarkMatterShells(
        radius=[initial_condition['radius_initial_dimensionless']],
        velocity=[initial_condition['vel_proper']],
        mass=[initial_condition['shell_mass']],
        angular_momentum=[initial_condition['specific_angular_momentum']],
        softening=initial_condition['softening'],
        fixed_enclosed_mass=initial_condition['central_mass'],
        code_units=code_unit_system,
    )
