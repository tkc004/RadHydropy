"""Analytic fixed-temperature hydrogen recombination solution."""

import numpy as np
import unyt

import radhydropy.thermo_networks.hydrogen as rth


def recombination_rate_cgs_s(temperature_proper_cgs_K_unyt, hydrogen_number_density_cgs_cm3_unyt):
    """Return ``nH alpha_B`` for case-B recombination."""

    alpha_B = rth._cgs_alpha_B(temperature_proper_cgs_K_unyt.to_value(unyt.K))
    nH = hydrogen_number_density_cgs_cm3_unyt.to_value(1.0 / unyt.cm**3)
    return alpha_B * nH / unyt.s


def ionized_fraction_dimensionless(
    time_proper_yr,
    initial_neutral_fraction_dimensionless,
    temperature_proper_cgs_K_unyt,
    hydrogen_number_density_cgs_cm3_unyt,
):
    """Return the pure case-B ionized fraction."""

    time_proper_yr_unyt = np.asarray(time_proper_yr) * unyt.yr
    rate_time = (
        recombination_rate_cgs_s(
            temperature_proper_cgs_K_unyt,
            hydrogen_number_density_cgs_cm3_unyt,
        )
        * time_proper_yr_unyt
    ).value
    y0 = 1.0 - initial_neutral_fraction_dimensionless
    return y0 / (1.0 + y0 * rate_time)
