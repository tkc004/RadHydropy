"""Analytic/reference profiles for spherical advection examples."""

import numpy as np


def gaussian(radius_proper_code, inverse_width_code, center_proper_code):
    """Return a Gaussian profile."""

    return np.exp(-np.power(inverse_width_code * (radius_proper_code - center_proper_code), 2.0))


def expanding_quantity(geometry_index, alpha_code, time_proper_code,
                       radius_proper_code, inverse_width_code,
                       center_proper_code):
    """Return the analytic homologous-expansion profile."""

    return (
        np.exp(-(geometry_index + 1.0) * alpha_code * time_proper_code)
        * gaussian(radius_proper_code * np.exp(-alpha_code * time_proper_code),
                   inverse_width_code, center_proper_code)
    )


def top_hat_density_profile(
    radius_proper_code,
    time_proper_code,
    vel_proper_code,
    box_size_proper_code,
    rho_high_proper_code,
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

    if hasattr(radius_proper_code, "to_value"):
        radius_proper_code = radius_proper_code.to_value()
    radius_proper_code = np.asarray(radius_proper_code, dtype=float)
    if hasattr(time_proper_code, "to_value"):
        time_proper_code = time_proper_code.to_value()
    if hasattr(vel_proper_code, "to_value"):
        vel_proper_code = vel_proper_code.to_value()
    if hasattr(box_size_proper_code, "to_value"):
        box_size_proper_code = box_size_proper_code.to_value()
    launch_radius_proper_code = radius_proper_code - time_proper_code * vel_proper_code
    rho_proper_code = density_low_factor * rho_high_proper_code * np.ones_like(radius_proper_code)

    inside = np.logical_and(launch_radius_proper_code >= 0.0, launch_radius_proper_code <= box_size_proper_code)
    rho_proper_code[
        np.logical_and(
            launch_radius_proper_code >= left_fraction * box_size_proper_code,
            launch_radius_proper_code <= right_fraction * box_size_proper_code,
        )
    ] = rho_high_proper_code

    rho_proper_code_result = np.zeros_like(radius_proper_code)
    positive = radius_proper_code > 0.0
    rho_proper_code_result[inside & positive] = (
        rho_proper_code[inside & positive]
        * (launch_radius_proper_code[inside & positive] / radius_proper_code[inside & positive]) ** 2.0
    )
    return rho_proper_code_result
