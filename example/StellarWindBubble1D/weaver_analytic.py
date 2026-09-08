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


def wind_luminosity(rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt):
    """Return the mechanical luminosity of the injected stellar wind."""

    rho_outflow_proper_cgs_g_cm3 = _to_value(rho_outflow_proper_unyt, unyt.g / unyt.cm**3)
    vel_outflow_proper_cgs_cm_s = _to_value(vel_outflow_proper_unyt, unyt.cm / unyt.s)
    radius_injection_proper_cgs_cm = _to_value(radius_injection_proper_unyt, unyt.cm)
    luminosity_proper_cgs_erg_s = 2.0 * math.pi * radius_injection_proper_cgs_cm**2 * rho_outflow_proper_cgs_g_cm3 * vel_outflow_proper_cgs_cm_s**3
    return unyt.unyt_array(luminosity_proper_cgs_erg_s, unyt.erg / unyt.s)


def shock_radius(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt):
    """Return the Weaver forward-shock radius."""

    time_proper_cgs_s = _to_value(time_proper_code, unyt.s)
    rho_ambient_proper_cgs_g_cm3 = _to_value(rho_ambient_proper_unyt, unyt.g / unyt.cm**3)
    luminosity_proper_cgs_erg_s = wind_luminosity(rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt).to_value(
        unyt.erg / unyt.s
    )
    radius_shock_proper_cgs_cm = WEAVER_RADIUS_COEFFICIENT * (luminosity_proper_cgs_erg_s / rho_ambient_proper_cgs_g_cm3) ** 0.2 * time_proper_cgs_s**0.6
    return unyt.unyt_array(radius_shock_proper_cgs_cm, unyt.cm)


def shock_velocity(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt):
    """Return the Weaver forward-shock velocity."""

    radius_shock_proper_unyt = shock_radius(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt)
    time_proper_cgs_s = _to_value(time_proper_code, unyt.s)
    vel_shock_proper_cgs_cm_s = 0.6 * radius_shock_proper_unyt.to_value(unyt.cm) / time_proper_cgs_s
    return unyt.unyt_array(vel_shock_proper_cgs_cm_s, unyt.cm / unyt.s)


def bubble_pressure(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt):
    """Return the interior pressure of the energy-driven bubble."""

    time_proper_cgs_s = _to_value(time_proper_code, unyt.s)
    rho_ambient_proper_cgs_g_cm3 = _to_value(rho_ambient_proper_unyt, unyt.g / unyt.cm**3)
    luminosity_proper_cgs_erg_s = wind_luminosity(rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt).to_value(
        unyt.erg / unyt.s
    )
    pressure_bubble_proper_cgs_dyn_cm2 = WEAVER_PRESSURE_COEFFICIENT * luminosity_proper_cgs_erg_s**0.4 * rho_ambient_proper_cgs_g_cm3**0.6 * time_proper_cgs_s**(-0.8)
    return unyt.unyt_array(pressure_bubble_proper_cgs_dyn_cm2, unyt.dyn / unyt.cm**2)


def weaver_solution(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt):
    """Return the Weaver radius, velocity, and pressure."""

    return (
        shock_radius(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt),
        shock_velocity(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt),
        bubble_pressure(time_proper_code, rho_ambient_proper_unyt, rho_outflow_proper_unyt, vel_outflow_proper_unyt, radius_injection_proper_unyt),
    )
