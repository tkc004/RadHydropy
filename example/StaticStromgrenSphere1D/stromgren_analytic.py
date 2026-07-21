"""Analytic static Stromgren sphere helpers."""

import numpy as np
import unyt


def stromgren_radius(source_photon_rate, hydrogen_number_density, alpha_B):
    """Return the on-the-spot Stromgren radius."""

    radius_cubed = (
        3.0
        * source_photon_rate
        / (4.0 * np.pi * alpha_B * hydrogen_number_density**2)
    )
    return radius_cubed**(1.0 / 3.0)


def stromgren_optical_depth(
    source_photon_rate,
    hydrogen_number_density,
    sigma_gamma,
    alpha_B,
):
    """Return ``tau_S = nH sigma_gamma R_S``."""

    radius = stromgren_radius(
        source_photon_rate,
        hydrogen_number_density,
        alpha_B,
    )
    return (hydrogen_number_density * sigma_gamma * radius).to_value('')


def recombination_time(hydrogen_number_density, alpha_B):
    """Return the case-B recombination time."""

    return (1.0 / (alpha_B * hydrogen_number_density)).to(unyt.Myr)


def ionization_front_radius(time, source_photon_rate, hydrogen_number_density, alpha_B):
    """Return ``R_I(t) = R_S [1 - exp(-t / tau_r)]^(1/3)``."""

    radius = stromgren_radius(source_photon_rate, hydrogen_number_density, alpha_B)
    time_rec = recombination_time(hydrogen_number_density, alpha_B)
    value = 1.0 - np.exp(-(time / time_rec).to_value(''))
    return radius * value**(1.0 / 3.0)


def neutral_fraction_profile(
    radius,
    hydrogen_number_density,
    sigma_gamma,
    alpha_B,
    source_photon_rate,
    inner_radius=0.1 * unyt.kpc,
    nsteps=60000,
):
    """Return the static analytic neutral fraction profile.

    The ODE is integrated in the optical-depth variable
    ``rn = r nH sigma_gamma``:

    ``dx/drn = x(1-x)/(1+x) * (x + 2/rn)``.
    """

    radius = radius.to(unyt.cm)
    inner_radius = inner_radius.to(unyt.cm)
    x_analytic = (
        hydrogen_number_density
        * alpha_B
        * 4.0
        * np.pi
        / sigma_gamma
        / source_photon_rate
        * radius**2
    ).to_value('')
    x_analytic = np.clip(x_analytic, 1.0e-300, 1.0 - 1.0e-12)

    integrate = radius > inner_radius
    if not np.any(integrate):
        return x_analytic

    optical_depth_start = (
        inner_radius
        * hydrogen_number_density
        * sigma_gamma
    ).to_value('')
    optical_depth_end = (
        np.amax(radius[integrate])
        * hydrogen_number_density
        * sigma_gamma
    ).to_value('')
    optical_depth_grid = np.linspace(optical_depth_start, optical_depth_end, nsteps)
    x_grid = np.empty_like(optical_depth_grid)
    x_grid[0] = (
        hydrogen_number_density
        * alpha_B
        * 4.0
        * np.pi
        / sigma_gamma
        / source_photon_rate
        * inner_radius**2
    ).to_value('')

    def derivative(optical_depth, x):
        x = np.clip(x, 1.0e-300, 1.0 - 1.0e-12)
        return x * (1.0 - x) * (x + 2.0 / optical_depth) / (1.0 + x)

    for i in range(len(optical_depth_grid) - 1):
        optical_depth = optical_depth_grid[i]
        h = optical_depth_grid[i + 1] - optical_depth
        x = x_grid[i]
        k1 = derivative(optical_depth, x)
        k2 = derivative(optical_depth + 0.5 * h, x + 0.5 * h * k1)
        k3 = derivative(optical_depth + 0.5 * h, x + 0.5 * h * k2)
        k4 = derivative(optical_depth + h, x + h * k3)
        x_grid[i + 1] = np.clip(
            x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
            1.0e-300,
            1.0 - 1.0e-12,
        )

    optical_depth_target = (
        radius[integrate]
        * hydrogen_number_density
        * sigma_gamma
    ).to_value('')
    x_analytic[integrate] = np.interp(
        optical_depth_target,
        optical_depth_grid,
        x_grid,
    )
    return x_analytic
