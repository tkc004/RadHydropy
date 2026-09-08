"""Analytic combined gas/dark-matter background helpers."""

import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])


def make_shell(config):
    initial_condition = config['initial_condition']
    code_unit_system = code_units_from_config(config)
    central_mass = float(initial_condition['central_dark_matter_mass'])
    gas_density = float(initial_condition['uniform_gas_density'])

    def enclosed_mass(radius_code):
        radius_code = np.asarray(radius_code, dtype=float)
        return central_mass + 4.0 * np.pi / 3.0 * gas_density * radius_code**3

    return DarkMatterShells(
        radius=[initial_condition['initial_radius']],
        velocity=[initial_condition['vel_proper']],
        mass=[initial_condition['shell_mass']],
        angular_momentum=[initial_condition['specific_angular_momentum']],
        softening=initial_condition['softening'],
        fixed_enclosed_mass=enclosed_mass,
        code_units=code_unit_system,
    )


def enclosed_mass(radius_code, config):
    initial_condition = config['initial_condition']
    radius_code = np.asarray(radius_code, dtype=float)
    return (
        float(initial_condition['central_dark_matter_mass'])
        + 4.0 * np.pi / 3.0
        * float(initial_condition['uniform_gas_density']) * radius_code**3
    )
