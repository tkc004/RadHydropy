"""Helpers for the pure dark-matter shell-crossing example."""

import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def make_shells(icparams, code_units):
    number = int(icparams['number_of_shells'])
    radius = np.linspace(
        icparams['inner_radius'], icparams['outer_radius'], number
    )
    mass = np.full(number, icparams['total_mass'] / number)
    velocity = icparams['initial_velocity_scale'] * radius
    angular_momentum = (
        icparams['angular_momentum_fraction']
        * np.sqrt(radius)
    )
    return DarkMatterShells(
        radius,
        velocity,
        mass,
        angular_momentum=angular_momentum,
        softening=icparams['softening'],
        code_units=code_units,
    )


def load_units(runparams):
    return CodeUnits.from_mapping(runparams['units']['CodeUnits'])
