"""Analytic fixed-field photoionization without recombination."""

import numpy as np
import unyt

import radhydropy.hydrogen as rh


def photoionization_rate(photon_number_density, sigma_gamma):
    """Return ``c sigma_gamma n_gamma``."""

    return rh.photoionization_frequency(photon_number_density, sigma_gamma)


def neutral_fraction(
    time,
    initial_neutral_fraction,
    photon_number_density,
    sigma_gamma,
):
    """Return the exponential fixed-field neutral fraction."""

    time = np.asarray(time) * unyt.yr
    rate_photo = photoionization_rate(
        photon_number_density,
        sigma_gamma,
    ).to_value(1.0 / unyt.s)
    return initial_neutral_fraction * np.exp(-rate_photo * time.to_value(unyt.s))
