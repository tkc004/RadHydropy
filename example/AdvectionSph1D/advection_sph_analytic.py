"""Analytic/reference profiles for spherical advection examples."""

import numpy as np


def gaussian(radius, inverse_width, center):
    """Return a Gaussian profile."""

    return np.exp(-np.power(inverse_width * (radius - center), 2.0))


def expanding_quantity(geometry_index, alpha, time, radius, inverse_width, center):
    """Return the analytic homologous-expansion profile."""

    return (
        np.exp(-(geometry_index + 1.0) * alpha * time)
        * gaussian(radius * np.exp(-alpha * time), inverse_width, center)
    )


def top_hat_density_profile(
    radius,
    time,
    velocity,
    boxsize,
    density_high,
    density_low_factor=0.01,
    left_fraction=0.25,
    right_fraction=0.75,
):
    """Return the advected top-hat density reference profile."""

    x1 = left_fraction * boxsize + time * velocity
    x2 = right_fraction * boxsize + time * velocity
    rho = density_high * np.ones(len(radius))
    if x1 > boxsize:
        x1 -= boxsize
    if x2 > boxsize:
        x2 -= boxsize
    if x2 > x1:
        rho[np.logical_or(radius < x1, radius > x2)] = (
            density_low_factor * density_high
        )
    if x1 > x2:
        rho[radius < x1] = density_low_factor * density_high
    return rho
