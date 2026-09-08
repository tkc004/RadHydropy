"""Helpers for the fixed-enclosed-mass dark-matter orbit benchmark."""

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def load_units(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])


def make_shell(initial_condition, code_unit_system):
    return DarkMatterShells(
        radius=[initial_condition['initial_radius']],
        velocity=[initial_condition['initial_velocity']],
        mass=[initial_condition['shell_mass']],
        angular_momentum=[initial_condition['specific_angular_momentum']],
        softening=initial_condition['softening'],
        fixed_enclosed_mass=initial_condition['central_mass'],
        code_units=code_unit_system,
    )
