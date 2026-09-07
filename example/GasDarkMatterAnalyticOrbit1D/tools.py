"""Analytic combined gas/dark-matter background helpers."""

import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def load_units(par_config):
    return CodeUnits.from_mapping(par_config['units']['CodeUnits'])


def make_shell(initial_condition, code_unit_system):
    central_mass = float(initial_condition['central_dark_matter_mass'])
    gas_density = float(initial_condition['uniform_gas_density'])

    def enclosed_mass(radius):
        radius = np.asarray(radius, dtype=float)
        return central_mass + 4.0 * np.pi / 3.0 * gas_density * radius**3

    return DarkMatterShells(
        radius=[initial_condition['initial_radius']],
        velocity=[initial_condition['initial_velocity']],
        mass=[initial_condition['shell_mass']],
        angular_momentum=[initial_condition['specific_angular_momentum']],
        softening=initial_condition['softening'],
        fixed_enclosed_mass=enclosed_mass,
        code_unit_system=code_unit_system,
    )


def enclosed_mass(radius, initial_condition):
    radius = np.asarray(radius, dtype=float)
    return (
        float(initial_condition['central_dark_matter_mass'])
        + 4.0 * np.pi / 3.0
        * float(initial_condition['uniform_gas_density']) * radius**3
    )
