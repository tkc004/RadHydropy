"""Compton heating and cooling from an isotropic CMB background."""

import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, SPEED_OF_LIGHT_CGS


THOMSON_CROSS_SECTION_CGS = float(
    unyt.physical_constants.thomson_cross_section_cgs.to_value(unyt.cm**2)
)
ELECTRON_MASS_CGS = float(
    unyt.physical_constants.electron_mass_cgs.to_value(unyt.g)
)
RADIATION_DENSITY_CONSTANT_CGS = float(
    unyt.physical_constants.radiation_density_constant_cgs.to_value(
        unyt.erg / (unyt.cm**3 * unyt.K**4)
    )
)


def cmb_compton_rate(
    temperature_K,
    electron_density_cm3,
    enabled=False,
    redshift=0.0,
    cmb_temperature_0_K=2.7255,
):
    """Return CMB Compton heating/cooling in ``erg cm^-3 s^-1``.

    A positive result heats the gas.  The source is disabled unless
    ``enabled`` is true, so existing runs are unchanged by the new option.
    """
    temperature_K = np.asarray(temperature_K, dtype=float)
    electron_density_cm3 = np.asarray(electron_density_cm3, dtype=float)
    if not enabled:
        return np.zeros_like(temperature_K)
    redshift = float(redshift)
    if redshift < -1.0:
        raise ValueError("compton_cmb_redshift must be greater than or equal to -1")
    cmb_temperature = float(cmb_temperature_0_K) * (1.0 + redshift)
    coefficient = (
        4.0
        * THOMSON_CROSS_SECTION_CGS
        * SPEED_OF_LIGHT_CGS
        * RADIATION_DENSITY_CONSTANT_CGS
        * BOLTZMANN_CONSTANT_CGS
        / (ELECTRON_MASS_CGS * SPEED_OF_LIGHT_CGS**2)
    )
    return coefficient * electron_density_cm3 * cmb_temperature**4 * (
        cmb_temperature - temperature_K
    )
