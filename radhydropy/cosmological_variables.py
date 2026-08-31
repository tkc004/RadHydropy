"""Conversions between physical and supercomoving fluid variables."""

import numpy as np


def supercomoving_scale(par, time=None):
    """Return ``(a, H)`` at the active supercomoving time."""
    if time is None:
        time = getattr(getattr(par, 'simulation', None), 'current_time', 0.0)
    tau = float(np.asarray(time, dtype=float))
    cosmology = par.cosmology
    _, scale_factor, hubble = cosmology.background_state_from_supercomoving(tau)
    return float(scale_factor), float(hubble)


def to_supercomoving_density(density, scale_factor):
    return np.asarray(density, dtype=float) * scale_factor**3


def to_supercomoving_temperature(temperature, scale_factor, gamma):
    return np.asarray(temperature, dtype=float) * scale_factor ** (3.0 * (gamma - 1.0))


def to_supercomoving_velocity(velocity, radius, scale_factor, hubble):
    """Convert proper velocity at comoving radius ``x`` to ``v``."""
    proper_radius = scale_factor * np.asarray(radius, dtype=float)
    return scale_factor * (
        np.asarray(velocity, dtype=float) - hubble * proper_radius
    )


def physical_density(density, scale_factor):
    return np.asarray(density, dtype=float) / scale_factor**3


def physical_temperature(temperature, scale_factor, gamma):
    return np.asarray(temperature, dtype=float) / scale_factor ** (3.0 * (gamma - 1.0))


def physical_pressure(pressure, scale_factor, gamma):
    return np.asarray(pressure, dtype=float) / scale_factor ** (3.0 * gamma)


def physical_velocity(velocity, radius, scale_factor, hubble):
    return hubble * scale_factor * np.asarray(radius, dtype=float) + (
        np.asarray(velocity, dtype=float) / scale_factor
    )


def physical_fields(radius, density, velocity, temperature, cosmology, tau, gamma):
    """Convert a supercomoving field bundle to physical variables."""
    _, a, hubble = cosmology.background_state_from_supercomoving(tau)
    return {
        "radius": physical_radius(radius, a),
        "density": physical_density(density, a),
        "velocity": physical_velocity(velocity, radius, a, hubble),
        "temperature": physical_temperature(temperature, a, gamma),
    }


def supercomoving_fields(radius, density, velocity, temperature, cosmology, tau, gamma):
    """Convert a physical field bundle to supercomoving variables."""
    _, a, hubble = cosmology.background_state_from_supercomoving(tau)
    return {
        "radius": np.asarray(radius, dtype=float) / a,
        "density": to_supercomoving_density(density, a),
        "velocity": to_supercomoving_velocity(velocity, radius, a, hubble),
        "temperature": to_supercomoving_temperature(temperature, a, gamma),
    }


def physical_radius(radius, scale_factor):
    """Convert a comoving radius to a proper radius."""
    return scale_factor * np.asarray(radius, dtype=float)
