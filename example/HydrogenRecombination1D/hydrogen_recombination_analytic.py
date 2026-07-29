"""Analytic fixed-temperature hydrogen recombination solution."""

import numpy as np
import unyt

import radhydropy.thermo_networks.hydrogen as rth


def recombination_rate(temperature, hydrogen_number_density):
    """Return ``nH alpha_B`` for case-B recombination."""

    alpha_B = rth._cgs_alpha_B(temperature.to_value(unyt.K))
    nH = hydrogen_number_density.to_value(1.0 / unyt.cm**3)
    return alpha_B * nH / unyt.s


def ionized_fraction(time, initial_neutral_fraction, temperature, hydrogen_number_density):
    """Return the pure case-B ionized fraction."""

    time = np.asarray(time) * unyt.yr
    rate_time = (
        recombination_rate(temperature, hydrogen_number_density)
        * time
    ).value
    y0 = 1.0 - initial_neutral_fraction
    return y0 / (1.0 + y0 * rate_time)
