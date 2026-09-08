"""Helpers for the pure dark-matter shell-crossing example."""

import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def make_shells(config):
    initial_condition = config['initial_condition']
    code_unit_system = code_units_from_config(config)
    number = int(initial_condition['number_of_shells'])
    radius = np.linspace(
        initial_condition['inner_radius'], initial_condition['outer_radius'], number
    )
    mass = np.full(number, initial_condition['total_mass'] / number)
    velocity = initial_condition['initial_velocity_scale'] * radius
    angular_momentum = (
        initial_condition['angular_momentum_fraction']
        * np.sqrt(radius)
    )
    return DarkMatterShells(
        radius,
        velocity,
        mass,
        angular_momentum=angular_momentum,
        softening=initial_condition['softening'],
        code_units=code_unit_system,
    )


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
