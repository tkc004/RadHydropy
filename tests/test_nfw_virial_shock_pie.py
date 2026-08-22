"""Analytic diagnostics for the HM12 PIE NFW virial-shock example."""

import importlib.util
from pathlib import Path

import numpy as np


TOOLS_PATH = (
    Path(__file__).resolve().parents[1]
    / 'example'
    / 'NFWVirialShockPIE1D'
    / 'tools.py'
)
SPEC = importlib.util.spec_from_file_location('nfw_virial_shock_pie_tools', TOOLS_PATH)
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOLS)


def test_strong_shock_analytic_limit():
    result = TOOLS._finite_mach_shock(
        rho=1.0e-27,
        temperature=1.0e4,
        velocity=-500.0,
        shock_speed=0.0,
        gamma=5.0 / 3.0,
        mu=0.59,
    )

    assert result['mach_number'] > 10.0
    assert np.isclose(result['density_ratio_analytic'], 4.0, rtol=2.0e-2)
    assert np.isclose(result['pressure_ratio_analytic'],
                      2.0 * (5.0 / 3.0) / (5.0 / 3.0 + 1.0)
                      * result['mach_number']**2,
                      rtol=2.0e-2)


def test_subsonic_rankine_hugoniot_input_is_not_a_shock():
    result = TOOLS._finite_mach_shock(
        rho=1.0e-29,
        temperature=3.0e4,
        velocity=-10.0,
        shock_speed=0.0,
        gamma=5.0 / 3.0,
        mu=0.59,
    )

    assert result['mach_number'] < 1.0


def test_net_heating_has_infinite_positive_cooling_time():
    cooling_time, net_time, advection_time = TOOLS._thermal_times(
        pressure=1.0e-16,
        density=1.0e-28,
        net_rate=-1.0e-30,
        radius_kpc=100.0,
        speed_km_s=100.0,
        gamma=5.0 / 3.0,
    )

    assert np.isinf(cooling_time)
    assert net_time < 0.0
    assert advection_time > 0.0


def test_shock_detector_requires_compression_heating_and_deceleration():
    radius = np.arange(8.0) * 20.0 + 60.0
    density = np.ones(8)
    temperature = np.ones(8) * 1.0e4
    velocity = np.ones(8) * -300.0
    density[:4] = 4.0
    temperature[:4] = 4.0e4
    velocity[:4] = -100.0

    index, location = TOOLS._locate_shock(
        radius, density, temperature, velocity, virial_radius_kpc=100.0
    )

    assert index == 3
    assert location == radius[3]


def test_shock_detector_rejects_smooth_pie_profile():
    radius = np.arange(8.0) * 20.0 + 60.0
    density = 1.0 / radius**1.8
    temperature = 1.0e4 * (radius / radius[-1])**-0.2
    velocity = -radius

    index, location = TOOLS._locate_shock(
        radius, density, temperature, velocity, virial_radius_kpc=100.0
    )

    assert index is None
    assert location is None
