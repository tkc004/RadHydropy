"""Weaver et al. (1977) energy-driven stellar-wind bubble helper."""

from __future__ import annotations

import math

import numpy as np
import unyt


WEAVER_RADIUS_COEFFICIENT = (250.0 / (308.0 * math.pi)) ** (1.0 / 5.0)
WEAVER_PRESSURE_COEFFICIENT = (5.0 / (22.0 * math.pi)) * WEAVER_RADIUS_COEFFICIENT ** -3


def _to_value(quantity, unit):
    if hasattr(quantity, 'to_value'):
        return quantity.to_value(unit)
    return np.asarray(quantity, dtype=float)


def wind_luminosity(rho_outflow, vel_outflow, injection_radius):
    """Return the mechanical luminosity of the injected stellar wind."""

    rho = _to_value(rho_outflow, unyt.g / unyt.cm**3)
    vel = _to_value(vel_outflow, unyt.cm / unyt.s)
    rinj = _to_value(injection_radius, unyt.cm)
    luminosity = 2.0 * math.pi * rinj**2 * rho * vel**3
    return unyt.unyt_array(luminosity, unyt.erg / unyt.s)


def shock_radius(time, rho_ambient, rho_outflow, vel_outflow, injection_radius):
    """Return the Weaver forward-shock radius."""

    t = _to_value(time, unyt.s)
    density = _to_value(rho_ambient, unyt.g / unyt.cm**3)
    luminosity = wind_luminosity(rho_outflow, vel_outflow, injection_radius).to_value(
        unyt.erg / unyt.s
    )
    radius = WEAVER_RADIUS_COEFFICIENT * (luminosity / density) ** 0.2 * t**0.6
    return unyt.unyt_array(radius, unyt.cm)


def shock_velocity(time, rho_ambient, rho_outflow, vel_outflow, injection_radius):
    """Return the Weaver forward-shock velocity."""

    radius = shock_radius(time, rho_ambient, rho_outflow, vel_outflow, injection_radius)
    t = _to_value(time, unyt.s)
    velocity = 0.6 * radius.to_value(unyt.cm) / t
    return unyt.unyt_array(velocity, unyt.cm / unyt.s)


def bubble_pressure(time, rho_ambient, rho_outflow, vel_outflow, injection_radius):
    """Return the interior pressure of the energy-driven bubble."""

    t = _to_value(time, unyt.s)
    density = _to_value(rho_ambient, unyt.g / unyt.cm**3)
    luminosity = wind_luminosity(rho_outflow, vel_outflow, injection_radius).to_value(
        unyt.erg / unyt.s
    )
    pressure = WEAVER_PRESSURE_COEFFICIENT * luminosity**0.4 * density**0.6 * t**(-0.8)
    return unyt.unyt_array(pressure, unyt.dyn / unyt.cm**2)


def weaver_solution(time, rho_ambient, rho_outflow, vel_outflow, injection_radius):
    """Return the Weaver radius, velocity, and pressure."""

    return (
        shock_radius(time, rho_ambient, rho_outflow, vel_outflow, injection_radius),
        shock_velocity(time, rho_ambient, rho_outflow, vel_outflow, injection_radius),
        bubble_pressure(time, rho_ambient, rho_outflow, vel_outflow, injection_radius),
    )
