"""Analytic combined gas/dark-matter background helpers."""

import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def load_units(runparams):
    return CodeUnits.from_mapping(runparams['units']['CodeUnits'])


def make_shell(icparams, code_units):
    central_mass = float(icparams['central_dark_matter_mass'])
    gas_density = float(icparams['uniform_gas_density'])

    def enclosed_mass(radius):
        radius = np.asarray(radius, dtype=float)
        return central_mass + 4.0 * np.pi / 3.0 * gas_density * radius**3

    return DarkMatterShells(
        radius=[icparams['initial_radius']],
        velocity=[icparams['initial_velocity']],
        mass=[icparams['shell_mass']],
        angular_momentum=[icparams['specific_angular_momentum']],
        softening=icparams['softening'],
        fixed_enclosed_mass=enclosed_mass,
        code_units=code_units,
    )


def enclosed_mass(radius, icparams):
    radius = np.asarray(radius, dtype=float)
    return (
        float(icparams['central_dark_matter_mass'])
        + 4.0 * np.pi / 3.0
        * float(icparams['uniform_gas_density']) * radius**3
    )
