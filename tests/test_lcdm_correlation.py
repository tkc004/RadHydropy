import importlib.util
from pathlib import Path

import h5py
import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "lcdm_correlation.py"
_SPEC = importlib.util.spec_from_file_location("lcdm_correlation_test_module", _MODULE_PATH)
_LCDM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LCDM)

generate_lcdm_correlation_table = _LCDM.generate_lcdm_correlation_table
linear_correlation_from_power_spectrum = _LCDM.linear_correlation_from_power_spectrum
linear_matter_power_spectrum = _LCDM.linear_matter_power_spectrum


def test_table_schema(tmp_path):
    filename = tmp_path / "lcdm_linear_correlation.h5"
    generate_lcdm_correlation_table(
        filename,
        radius_mpc_h=np.geomspace(1.0e-2, 3.0e3, 64),
        k_hmpc=np.geomspace(1.0e-5, 1.0e3, 4096),
    )

    with h5py.File(filename, "r") as handle:
        assert set(handle) == {"radius_mpc_h", "correlation", "k_hmpc", "power"}

        radius = handle["radius_mpc_h"][:]
        k = handle["k_hmpc"][:]
        power = handle["power"][:]
        correlation = handle["correlation"][:]

        assert np.all(np.diff(radius) > 0.0)
        assert np.all(np.diff(k) > 0.0)
        assert np.all(power >= 0.0)
        assert np.all(np.isfinite(correlation))
        assert handle.attrs["sigma8"] == 0.811


def test_sigma8_normalization():
    k = np.geomspace(1.0e-5, 1.0e3, 8192)
    power = linear_matter_power_spectrum(k)

    kr = 8.0 * k
    window = 3.0 * (np.sin(kr) - kr * np.cos(kr)) / kr**3
    sigma8 = np.sqrt(
        np.trapz(k**3 * power * window**2, np.log(k))
        / (2.0 * np.pi**2)
    )

    assert np.isclose(sigma8, 0.811, rtol=1.0e-4)


def test_gaussian_power_spectrum_integral():
    amplitude = 2.0
    alpha = 1.0
    k = np.geomspace(1.0e-5, 30.0, 20000)
    radius = np.geomspace(0.05, 5.0, 30)
    power = amplitude * np.exp(-alpha * k**2)

    measured = linear_correlation_from_power_spectrum(radius, k, power)
    expected = (
        amplitude / (8.0 * np.pi**1.5 * alpha**1.5)
        * np.exp(-radius**2 / (4.0 * alpha))
    )

    assert np.allclose(measured, expected, rtol=2.0e-4, atol=1.0e-7)


def test_correlation_convergence():
    # At very large radii the small residual correlation is sensitive to the
    # finite k-range and endpoint cancellation.  Test the well-resolved
    # nonlinear-scale range here; the table-shape test separately checks the
    # large-radius tail.
    radius = np.geomspace(0.1, 10.0, 50)

    coarse = generate_lcdm_correlation_table(
        radius_mpc_h=radius,
        k_hmpc=np.geomspace(1.0e-5, 1.0e3, 4096),
    )["correlation"]
    fine = generate_lcdm_correlation_table(
        radius_mpc_h=radius,
        k_hmpc=np.geomspace(1.0e-5, 1.0e3, 16384),
    )["correlation"]

    assert np.allclose(coarse, fine, rtol=5.0e-3, atol=2.0e-5)


def test_lcdm_correlation_shape():
    result = generate_lcdm_correlation_table()
    radius = result["radius_mpc_h"]
    correlation = result["correlation"]

    assert correlation[radius < 1.0].max() > 1.0
    assert abs(correlation[-1]) < 1.0e-2
    assert np.all(np.isfinite(correlation))
