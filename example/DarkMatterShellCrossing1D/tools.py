"""Helpers for the pure dark-matter shell-crossing example."""

import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def make_shells(config):
    initial_condition = config['initial_condition']
    code_unit_system = code_units_from_config(config)
    number = int(initial_condition['number_of_shells'])
    radius_dimensionless = np.linspace(
        initial_condition['radius_inner_dimensionless'], initial_condition['radius_outer_dimensionless'], number
    )
    mass_dimensionless = np.full(number, initial_condition['total_mass_dimensionless'] / number)
    velocity_dimensionless = initial_condition['velocity_scale_dimensionless'] * radius_dimensionless
    angular_momentum_dimensionless = (
        initial_condition['angular_momentum_fraction']
        * np.sqrt(radius_dimensionless)
    )
    return DarkMatterShells(
        radius_dimensionless,
        velocity_dimensionless,
        mass_dimensionless,
        angular_momentum=angular_momentum_dimensionless,
        softening=initial_condition['softening_dimensionless'],
        code_units=code_unit_system,
    )


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
