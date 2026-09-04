"""Helpers for the fixed-enclosed-mass dark-matter orbit benchmark."""

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def load_units(runparams):
    return CodeUnits.from_mapping(runparams['units']['CodeUnits'])


def make_shell(icparams, code_units):
    return DarkMatterShells(
        radius=[icparams['initial_radius']],
        velocity=[icparams['initial_velocity']],
        mass=[icparams['shell_mass']],
        angular_momentum=[icparams['specific_angular_momentum']],
        softening=icparams['softening'],
        fixed_enclosed_mass=icparams['central_mass'],
        code_units=code_units,
    )
