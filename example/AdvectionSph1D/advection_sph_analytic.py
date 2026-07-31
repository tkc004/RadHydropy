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
    """Return the spherical advected top-hat density reference profile.

    The cartesian reference used a pure translation.  In spherical symmetry,
    a fluid element conserves ``rho * r^2`` along characteristics for a
    constant radial velocity, so the density acquires a geometric dilution
    factor of ``(r0 / r)^2`` where ``r0 = r - v t`` is the launch radius.
    """

    if hasattr(radius, "to_value"):
        radius = radius.to_value()
    radius = np.asarray(radius, dtype=float)
    if hasattr(time, "to_value"):
        time = time.to_value()
    if hasattr(velocity, "to_value"):
        velocity = velocity.to_value()
    if hasattr(boxsize, "to_value"):
        boxsize = boxsize.to_value()
    launch_radius = radius - time * velocity
    initial_density = density_low_factor * density_high * np.ones_like(radius)

    inside = np.logical_and(launch_radius >= 0.0, launch_radius <= boxsize)
    initial_density[
        np.logical_and(
            launch_radius >= left_fraction * boxsize,
            launch_radius <= right_fraction * boxsize,
        )
    ] = density_high

    rho = np.zeros_like(radius)
    positive = radius > 0.0
    rho[inside & positive] = (
        initial_density[inside & positive]
        * (launch_radius[inside & positive] / radius[inside & positive]) ** 2.0
    )
    return rho
