# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Analytic cosmological Strömgren-sphere reference solution."""

from typing import Any

import numpy as np
import unyt
from scipy.integrate import solve_ivp


def front_radius_comoving(
    scale_factor: Any,
    initial_scale_factor: float,
    initial_time: Any,
    source_photon_rate: Any,
    hydrogen_number_density_initial: Any,
    alpha_b: Any,
) -> Any:
    """Return the recombination-only cosmological ionization-front radius.

    For an Einstein--de Sitter background, ``x=t/t_i=(a/a_i)**(3/2)`` and
    ``y=(r_I/r_S,i)**3`` obeys ``dy/dx=lambda_i*(1-y/x**2)``.  This is the
    numerically stable form of the exponential-integral solution used in the
    original reference script.
    """
    scale_factor = np.asarray(scale_factor, dtype=float)
    if np.any(scale_factor < initial_scale_factor):
        raise ValueError("scale_factor must not precede the source turn-on")
    x = (scale_factor / initial_scale_factor) ** 1.5
    time_initial_s = initial_time.to_value(unyt.s)
    density_initial = hydrogen_number_density_initial.to_value(1.0 / unyt.cm**3)
    alpha_value = alpha_b.to_value(unyt.cm**3 / unyt.s)
    source_rate = source_photon_rate.to_value(1.0 / unyt.s)
    lambda_initial = alpha_value * density_initial * time_initial_s
    radius_stromgren_initial_cm = (
        3.0 * source_rate / (4.0 * np.pi * alpha_value * density_initial**2)
    ) ** (1.0 / 3.0)

    def derivative(x_value, y_value):
        return lambda_initial * (1.0 - y_value / x_value**2)

    unique_x, inverse = np.unique(x, return_inverse=True)
    solution = solve_ivp(
        derivative,
        (1.0, float(unique_x[-1])),
        np.array([0.0]),
        t_eval=unique_x,
        rtol=2.0e-10,
        atol=1.0e-12,
    )
    if not solution.success:
        raise RuntimeError(f"cosmological Strömgren reference failed: {solution.message}")
    y = np.maximum(solution.y[0][inverse], 0.0)
    radius_comoving_cm = radius_stromgren_initial_cm * y ** (1.0 / 3.0) / initial_scale_factor
    return radius_comoving_cm * unyt.cm


def stromgren_radius_proper(
    scale_factor: Any,
    initial_scale_factor: float,
    hydrogen_number_density_initial: Any,
    source_photon_rate: Any,
    alpha_b: Any,
) -> Any:
    """Return the instantaneous proper Strömgren radius."""
    density = hydrogen_number_density_initial * (initial_scale_factor / scale_factor) ** 3
    return (
        3.0 * source_photon_rate / (4.0 * np.pi * alpha_b * density**2)
    ) ** (1.0 / 3.0)
