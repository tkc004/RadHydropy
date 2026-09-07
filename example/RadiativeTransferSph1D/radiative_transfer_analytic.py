"""Analytic optically thin spherical radiation-density profiles."""

import numpy as np
import unyt

from radhydropy.constants import SPEED_OF_LIGHT_CGS
from radhydropy.units import CodeUnits, code_quantity_to_cgs, code_unit_scales


def _normalize_code_units(code_unit_system):
    if code_unit_system is None:
        raise ValueError("code_units is required")
    return CodeUnits.from_mapping(code_unit_system)


def finite_volume_density(boundary_proper_code, volume_proper_code, source_photon_rate, code_unit_system):
    """Return the finite-volume average photon density."""

    code_unit_system = _normalize_code_units(code_unit_system)
    if hasattr(boundary_proper_code, 'to_value'):
        boundary_cgs_cm = boundary_proper_code.to_value(unyt.cm)
    else:
        boundary_cgs_cm = code_quantity_to_cgs(boundary_proper_code, code_unit_system, 'length_cgs_cm')
    if hasattr(volume_proper_code, 'to_value'):
        volume_cgs_cm3 = volume_proper_code.to_value(unyt.cm**3)
    else:
        volume_cgs_cm3 = code_quantity_to_cgs(volume_proper_code, code_unit_system, 'volume_cgs_cm3')
    if hasattr(source_photon_rate, 'to_value'):
        source_rate_s = source_photon_rate.to_value(1.0 / unyt.s)
    else:
        source_rate_s = (
            np.asarray(source_photon_rate, dtype=float)
            * code_unit_scales(code_unit_system)['photon_rate_per_s']
        )
    dr = boundary_cgs_cm[1:] - boundary_cgs_cm[:-1]
    speed_of_light = SPEED_OF_LIGHT_CGS
    density = source_rate_s * dr / volume_cgs_cm3 / speed_of_light
    return density * (1.0 / unyt.cm**3)


def point_density(radius, source_photon_rate, code_unit_system):
    """Return pointwise ``Q / (4 pi r^2 c)`` photon density."""

    code_unit_system = _normalize_code_units(code_unit_system)
    if hasattr(radius, 'to_value'):
        radius_cgs_cm = radius.to_value(unyt.cm)
    else:
        radius_cgs_cm = code_quantity_to_cgs(radius, code_unit_system, 'length_cgs_cm')
    if hasattr(source_photon_rate, 'to_value'):
        source_rate_s = source_photon_rate.to_value(1.0 / unyt.s)
    else:
        source_rate_s = (
            np.asarray(source_photon_rate, dtype=float)
            * code_unit_scales(code_unit_system)['photon_rate_per_s']
        )
    speed_of_light = SPEED_OF_LIGHT_CGS
    density = source_rate_s / (4.0 * np.pi * radius_cgs_cm**2 * speed_of_light)
    return density * (1.0 / unyt.cm**3)
