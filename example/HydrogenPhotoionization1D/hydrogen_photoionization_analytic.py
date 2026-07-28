"""Analytic fixed-field hydrogen photoionization solution."""

import numpy as np
import unyt

import radhydropy.chemistry_species.hydrogen as rh


def recombination_rate(temperature, hydrogen_number_density):
    """Return ``nH alpha_B``."""

    alpha_B = rh.alpha_B(temperature).to(unyt.cm**3 / unyt.s)
    nH = hydrogen_number_density.to(1.0 / unyt.cm**3)
    return (alpha_B * nH).to(1.0 / unyt.s)


def photoionization_rate(photon_number_density, sigma_gamma):
    """Return ``c sigma_gamma n_gamma``."""

    return rh.photoionization_frequency(photon_number_density, sigma_gamma)


def neutral_fraction(
    time,
    initial_neutral_fraction,
    temperature,
    hydrogen_number_density,
    photon_number_density,
    sigma_gamma,
):
    """Return the fixed-radiation neutral fraction including recombinations."""

    time = np.asarray(time) * unyt.yr
    rate_rec = recombination_rate(
        temperature,
        hydrogen_number_density,
    ).to_value(1.0 / unyt.s)
    rate_photo = photoionization_rate(
        photon_number_density,
        sigma_gamma,
    ).to_value(1.0 / unyt.s)
    time_s = time.to_value(unyt.s)
    x0 = initial_neutral_fraction

    if rate_rec == 0.0:
        return x0 * np.exp(-rate_photo * time_s)

    discriminant = np.sqrt(rate_photo**2 + 4.0 * rate_rec * rate_photo)
    root_low = (
        rate_photo + 2.0 * rate_rec - discriminant
    ) / (2.0 * rate_rec)
    root_high = (
        rate_photo + 2.0 * rate_rec + discriminant
    ) / (2.0 * rate_rec)
    ratio_initial = (x0 - root_low) / (x0 - root_high)
    ratio = ratio_initial * np.exp(-discriminant * time_s)
    return (root_low - ratio * root_high) / (1.0 - ratio)
