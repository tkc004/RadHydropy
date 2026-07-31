"""Analytic optically thin spherical radiation-density profiles."""

import numpy as np
import unyt

from radhydropy.constants import SPEED_OF_LIGHT_CGS
from radhydropy.units import CodeUnits, code_quantity_to_cgs, code_unit_scales


def _normalize_code_units(code_units):
    if code_units is None:
        raise ValueError("code_units is required")
    return CodeUnits.from_mapping(code_units)


def finite_volume_density(boundary, volume, source_photon_rate, code_units):
    """Return the finite-volume average photon density."""

    code_units = _normalize_code_units(code_units)
    if hasattr(boundary, 'to_value'):
        boundary_cm = boundary.to_value(unyt.cm)
    else:
        boundary_cm = code_quantity_to_cgs(boundary, code_units, 'length_cm')
    if hasattr(volume, 'to_value'):
        volume_cm3 = volume.to_value(unyt.cm**3)
    else:
        volume_cm3 = code_quantity_to_cgs(volume, code_units, 'volume_cm3')
    if hasattr(source_photon_rate, 'to_value'):
        source_rate_s = source_photon_rate.to_value(1.0 / unyt.s)
    else:
        source_rate_s = (
            np.asarray(source_photon_rate, dtype=float)
            * code_unit_scales(code_units)['photon_rate_per_s']
        )
    dr = boundary_cm[1:] - boundary_cm[:-1]
    speed_of_light = SPEED_OF_LIGHT_CGS
    density = source_rate_s * dr / volume_cm3 / speed_of_light
    return density * (1.0 / unyt.cm**3)


def point_density(radius, source_photon_rate, code_units):
    """Return pointwise ``Q / (4 pi r^2 c)`` photon density."""

    code_units = _normalize_code_units(code_units)
    if hasattr(radius, 'to_value'):
        radius_cm = radius.to_value(unyt.cm)
    else:
        radius_cm = code_quantity_to_cgs(radius, code_units, 'length_cm')
    if hasattr(source_photon_rate, 'to_value'):
        source_rate_s = source_photon_rate.to_value(1.0 / unyt.s)
    else:
        source_rate_s = (
            np.asarray(source_photon_rate, dtype=float)
            * code_unit_scales(code_units)['photon_rate_per_s']
        )
    speed_of_light = SPEED_OF_LIGHT_CGS
    density = source_rate_s / (4.0 * np.pi * radius_cm**2 * speed_of_light)
    return density * (1.0 / unyt.cm**3)
