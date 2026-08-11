import importlib.util
from pathlib import Path

import numpy as np
import unyt


TOOLS_PATH = (
    Path(__file__).resolve().parents[1]
    / 'example'
    / 'NFWVirialShock1D'
    / 'tools.py'
)
SPEC = importlib.util.spec_from_file_location('nfw_virial_shock_tools_test', TOOLS_PATH)
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOLS)


def test_strong_shock_rankine_hugoniot_limit_for_gamma_five_thirds():
    density_ratio, temperature_ratio = TOOLS.rankine_hugoniot_ratios(1.0e6)

    assert np.isclose(density_ratio, 4.0, rtol=1.0e-5)
    assert np.isclose(temperature_ratio, 0.3125 * 1.0e12, rtol=1.0e-5)


def test_mach_one_has_no_rankine_hugoniot_jump():
    density_ratio, temperature_ratio = TOOLS.rankine_hugoniot_ratios(1.0)

    assert np.isclose(density_ratio, 1.0)
    assert np.isclose(temperature_ratio, 1.0)


def test_cosmic_mean_baryon_density_scales_with_redshift():
    rho0 = TOOLS.cosmic_mean_baryon_density(
        70.0 * unyt.km / (unyt.s * unyt.Mpc), 0.049, 0.0
    )
    rho10 = TOOLS.cosmic_mean_baryon_density(
        70.0 * unyt.km / (unyt.s * unyt.Mpc), 0.049, 10.0
    )

    assert np.isclose((rho10 / rho0).value, 11.0**3)


def test_hubble_velocity_is_constructed_from_background_expansion():
    rate = TOOLS.hubble_rate(
        70.0 * unyt.km / (unyt.s * unyt.Mpc), 0.3, 0.7, 0.0
    )

    assert np.isclose(rate.to_value(unyt.km / (unyt.s * unyt.Mpc)), 70.0)
