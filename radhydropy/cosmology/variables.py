# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Conversions between physical and supercomoving fluid variables."""

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _as_float_array(value: Any) -> FloatArray:
    return cast("FloatArray", np.asarray(value, dtype=float))


def supercomoving_scale(
    par: Any,
    tau_supercomoving_code: Any = None,
) -> tuple[float, float]:
    """Return ``(a, H)`` at the active supercomoving time."""
    if tau_supercomoving_code is None:
        tau_supercomoving_code = par.tau_supercomoving_code
    tau = float(np.asarray(tau_supercomoving_code, dtype=float))
    cosmology_parameters = par.cosmology
    cosmology = getattr(cosmology_parameters, "model", cosmology_parameters)
    _, scale_factor, hubble = cosmology.background_state_from_supercomoving(tau)
    return float(scale_factor), float(hubble)


def to_supercomoving_density(density: Any, scale_factor: float) -> FloatArray:
    return cast("FloatArray", _as_float_array(density) * scale_factor**3)


def to_supercomoving_temperature(
    temperature: Any,
    scale_factor: float,
    gamma: float,
) -> FloatArray:
    return cast(
        "FloatArray",
        _as_float_array(temperature) * scale_factor ** (3.0 * (gamma - 1.0)),
    )


def to_supercomoving_velocity(
    velocity: Any,
    radius: Any,
    scale_factor: float,
    hubble: float,
) -> FloatArray:
    """Convert proper velocity at comoving radius ``x`` to ``v``."""
    proper_radius = scale_factor * _as_float_array(radius)
    return cast(
        "FloatArray",
        scale_factor * (_as_float_array(velocity) - hubble * proper_radius),
    )


def physical_density(density: Any, scale_factor: float) -> FloatArray:
    return cast("FloatArray", _as_float_array(density) / scale_factor**3)


def physical_temperature(
    temperature: Any,
    scale_factor: float,
    gamma: float,
) -> FloatArray:
    return cast(
        "FloatArray",
        _as_float_array(temperature) / scale_factor ** (3.0 * (gamma - 1.0)),
    )


def physical_pressure(pressure: Any, scale_factor: float, gamma: float) -> FloatArray:
    return cast("FloatArray", _as_float_array(pressure) / scale_factor ** (3.0 * gamma))


def physical_velocity(
    velocity: Any,
    radius: Any,
    scale_factor: float,
    hubble: float,
) -> FloatArray:
    return cast(
        "FloatArray",
        hubble * scale_factor * _as_float_array(radius)
        + (
            _as_float_array(velocity) / scale_factor
        ),
    )


def physical_fields(
    radius: Any,
    density: Any,
    velocity: Any,
    temperature: Any,
    cosmology: Any,
    tau: Any,
    gamma: float,
) -> dict[str, FloatArray]:
    """Convert a supercomoving field bundle to physical variables."""
    _, a, hubble = cosmology.background_state_from_supercomoving(tau)
    return {
        "radius": physical_radius(radius, a),
        "density": physical_density(density, a),
        "velocity": physical_velocity(velocity, radius, a, hubble),
        "temperature": physical_temperature(temperature, a, gamma),
    }


def supercomoving_fields(
    radius: Any,
    density: Any,
    velocity: Any,
    temperature: Any,
    cosmology: Any,
    tau: Any,
    gamma: float,
) -> dict[str, FloatArray]:
    """Convert a physical field bundle to supercomoving variables."""
    _, a, hubble = cosmology.background_state_from_supercomoving(tau)
    return {
        "radius": cast("FloatArray", _as_float_array(radius) / a),
        "density": to_supercomoving_density(density, a),
        "velocity": to_supercomoving_velocity(velocity, radius, a, hubble),
        "temperature": to_supercomoving_temperature(temperature, a, gamma),
    }


def physical_radius(radius: Any, scale_factor: float) -> FloatArray:
    """Convert a comoving radius to a proper radius."""
    return cast("FloatArray", scale_factor * _as_float_array(radius))
