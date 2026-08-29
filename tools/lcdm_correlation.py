"""Linear LCDM power spectra and real-space correlation functions.

Wavenumbers use ``h/Mpc``; radii use ``Mpc/h``; power spectra use
``(Mpc/h)^3``.  The Fourier-Bessel integral is exact for the supplied
tabulated linear power spectrum.  The built-in spectrum uses the analytic
Eisenstein--Hu no-wiggle transfer shape.
"""

from pathlib import Path
import importlib.util

import h5py
import numpy as np

try:
    from tools.cosmology import LambdaCDM
except ModuleNotFoundError:
    # The virial-shock example loads this file directly while its local
    # ``tools.py`` module shadows the repository-level ``tools`` package.
    _COSMOLOGY_FILE = Path(__file__).with_name("cosmology.py")
    _COSMOLOGY_SPEC = importlib.util.spec_from_file_location(
        "radhydropy_physical_cosmology", _COSMOLOGY_FILE
    )
    _COSMOLOGY_MODULE = importlib.util.module_from_spec(_COSMOLOGY_SPEC)
    _COSMOLOGY_SPEC.loader.exec_module(_COSMOLOGY_MODULE)
    LambdaCDM = _COSMOLOGY_MODULE.LambdaCDM


def _validate_lcdm_parameters(omega_m, omega_lambda, omega_b=None):
    LambdaCDM(omega_m=omega_m, omega_lambda=omega_lambda)
    if omega_b is not None and not (0.0 < omega_b < omega_m):
        raise ValueError("require 0 < omega_b < omega_m")


def eisenstein_hu_nowiggle_transfer(
    k_hmpc, omega_m=0.315, omega_b=0.049, h=0.674,
    omega_lambda=0.685, theta_cmb=2.7255 / 2.7
):
    """Return the Eisenstein--Hu no-wiggle transfer function."""
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    if np.any(k_hmpc <= 0.0):
        raise ValueError("wavenumbers must be positive")
    _validate_lcdm_parameters(omega_m, omega_lambda, omega_b)
    if h <= 0.0:
        raise ValueError("require h > 0")
    k_mpc = k_hmpc * float(h)
    q = k_mpc * float(theta_cmb) ** 2 / (float(omega_m) * float(h) ** 2)
    log_term = np.log(1.0 + 2.34 * q) / (2.34 * q)
    denominator = 1.0 + 3.89 * q + (16.1 * q) ** 2
    denominator += (5.46 * q) ** 3 + (6.71 * q) ** 4
    return log_term / denominator**0.25


def linear_matter_power_spectrum_shape(
    k_hmpc, omega_m=0.315, omega_b=0.049, h=0.674, n_s=0.965,
    omega_lambda=0.685,
):
    """Return the unnormalized ``k**n_s T(k)**2`` power-spectrum shape."""
    transfer = eisenstein_hu_nowiggle_transfer(
        k_hmpc, omega_m=omega_m, omega_b=omega_b, h=h,
        omega_lambda=omega_lambda,
    )
    return np.asarray(k_hmpc, dtype=float) ** float(n_s) * transfer**2


def linear_matter_power_spectrum(
    k_hmpc, omega_m=0.315, omega_b=0.049, h=0.674,
    n_s=0.965, sigma8=0.811, omega_lambda=0.685,
):
    """Return a sigma8-normalized linear matter power spectrum."""
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    shape = linear_matter_power_spectrum_shape(
        k_hmpc, omega_m=omega_m, omega_b=omega_b, h=h, n_s=n_s,
        omega_lambda=omega_lambda,
    )
    k_norm = np.geomspace(1.0e-5, 1.0e3, 8192)
    shape_norm = linear_matter_power_spectrum_shape(
        k_norm, omega_m=omega_m, omega_b=omega_b, h=h, n_s=n_s,
        omega_lambda=omega_lambda,
    )
    kr = 8.0 * k_norm
    window = 3.0 * (np.sin(kr) - kr * np.cos(kr)) / np.maximum(kr**3, 1.0e-30)
    sigma8_shape = np.sqrt(
        np.trapz(k_norm**3 * shape_norm * window**2, np.log(k_norm))
        / (2.0 * np.pi**2)
    )
    return shape * (float(sigma8) / max(sigma8_shape, 1.0e-300)) ** 2


def plot_lcdm_transfer_function(
    filename=None, k_hmpc=None, omega_m=0.315, omega_b=0.049, h=0.674,
    omega_lambda=0.685,
):
    """Plot the dimensionless Eisenstein--Hu transfer function.

    The horizontal axis is ``k`` in ``h/Mpc``.  If ``filename`` is supplied,
    the figure is saved there and the figure is closed; the sampled ``(k, T)``
    arrays are returned in all cases.
    """
    import matplotlib.pyplot as plt

    if k_hmpc is None:
        k_hmpc = np.geomspace(1.0e-4, 1.0e2, 512)
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    transfer = eisenstein_hu_nowiggle_transfer(
        k_hmpc, omega_m=omega_m, omega_b=omega_b, h=h,
        omega_lambda=omega_lambda,
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.loglog(k_hmpc, transfer, color="tab:blue", linewidth=2.0)
    axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel(r"$T(k)$")
    axis.set_title("LCDM linear matter transfer function")
    axis.grid(True, which="both", alpha=0.25)
    axis.set_ylim(bottom=1.0e-4)
    figure.tight_layout()

    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=200)
        plt.close(figure)
    return k_hmpc, transfer


def plot_linear_matter_power_spectrum(
    filename=None, k_hmpc=None, omega_m=0.315, omega_b=0.049,
    h=0.674, n_s=0.965, sigma8=0.811, omega_lambda=0.685,
):
    """Plot the sigma8-normalized linear matter power spectrum."""
    import matplotlib.pyplot as plt

    if k_hmpc is None:
        k_hmpc = np.geomspace(1.0e-4, 1.0e2, 512)
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    power = linear_matter_power_spectrum(
        k_hmpc, omega_m=omega_m, omega_b=omega_b, h=h,
        n_s=n_s, sigma8=sigma8, omega_lambda=omega_lambda,
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.loglog(k_hmpc, power, color="tab:green", linewidth=2.0)
    axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
    axis.set_title(r"Linear matter power spectrum ($\sigma_8=%.3f$)" % sigma8)
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()

    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=200)
        plt.close(figure)
    return k_hmpc, power


def plot_linear_correlation_from_power_spectrum(
    filename=None, radius_mpc_h=None, k_hmpc=None, power=None,
    omega_m=0.315, omega_b=0.049, h=0.674, n_s=0.965, sigma8=0.811,
    omega_lambda=0.685,
):
    """Plot ``xi(r)`` computed from a tabulated or built-in linear ``P(k)``.

    The default plotted range ends at 50 Mpc/h, before finite-k endpoint
    ringing dominates the very small large-radius correlation signal.  The
    correlation-integral routine and generated table are not range-limited.
    """
    import matplotlib.pyplot as plt

    if radius_mpc_h is None:
        radius_mpc_h = np.geomspace(1.0e-2, 50.0, 512)
    if k_hmpc is None:
        k_hmpc = np.geomspace(1.0e-5, 1.0e3, 8192)
    radius_mpc_h = np.asarray(radius_mpc_h, dtype=float)
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    if power is None:
        power = linear_matter_power_spectrum(
            k_hmpc, omega_m=omega_m, omega_b=omega_b, h=h,
            n_s=n_s, sigma8=sigma8, omega_lambda=omega_lambda,
        )
    power = np.asarray(power, dtype=float)
    correlation = linear_correlation_from_power_spectrum(
        radius_mpc_h, k_hmpc, power
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    if np.all(correlation > 0.0):
        axis.loglog(radius_mpc_h, correlation, color="tab:red", linewidth=2.0)
    else:
        axis.semilogx(radius_mpc_h, correlation, color="tab:red", linewidth=2.0)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_yscale("symlog", linthresh=1.0e-3)
    axis.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    axis.set_ylabel(r"$\xi(r)$")
    axis.set_title("Linear matter correlation function")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()

    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=200)
        plt.close(figure)
    return radius_mpc_h, correlation


def linear_correlation_from_power_spectrum(radius_mpc_h, k_hmpc, power):
    """Compute ``xi(r)`` exactly for a supplied tabulated ``P(k)``."""
    radius_mpc_h = np.asarray(radius_mpc_h, dtype=float)
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    power = np.asarray(power, dtype=float)
    if radius_mpc_h.ndim != 1 or k_hmpc.ndim != 1 or power.ndim != 1:
        raise ValueError("radius, k, and power must be one-dimensional")
    if k_hmpc.size != power.size or np.any(k_hmpc <= 0.0):
        raise ValueError("k and power must have equal length and positive k")
    if np.any(~np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("power spectrum must be finite and non-negative")
    order = np.argsort(k_hmpc)
    k = k_hmpc[order]
    p = power[order]
    kr = np.outer(radius_mpc_h, k)
    j0 = np.sinc(kr / np.pi)
    integrand = k[None, :] ** 3 * p[None, :] * j0
    return np.trapz(integrand, np.log(k), axis=1) / (2.0 * np.pi**2)


def load_lcdm_correlation_table(filename):
    """Load a previously generated correlation table from HDF5."""
    with h5py.File(filename, "r") as handle:
        result = {
            "radius_mpc_h": handle["radius_mpc_h"][:],
            "correlation": handle["correlation"][:],
            "k_hmpc": handle["k_hmpc"][:],
            "power": handle["power"][:],
            "attributes": dict(handle.attrs),
        }
    return result


def generate_lcdm_correlation_table(
    filename=None, radius_mpc_h=None, k_hmpc=None,
    omega_m=0.315, omega_b=0.049, h=0.674, n_s=0.965, sigma8=0.811,
    omega_lambda=0.685, k_min_hmpc=None,
):
    """Generate a linear correlation table, optionally with a box cutoff.

    ``k_min_hmpc`` removes modes larger than the modeled comoving box.  The
    same cutoff is used in the correlation integral and in the stored power
    spectrum, so the table describes the finite-volume realization rather
    than an infinite-volume correlation function.
    """
    if radius_mpc_h is None:
        radius_mpc_h = np.geomspace(1.0e-2, 3.0e3, 1024)
    if k_hmpc is None:
        k_lower = 1.0e-5 if k_min_hmpc is None else float(k_min_hmpc)
        if k_lower <= 0.0:
            raise ValueError("k_min_hmpc must be positive")
        k_hmpc = np.geomspace(k_lower, 1.0e3, 8192)
    radius_mpc_h = np.asarray(radius_mpc_h, dtype=float)
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    if k_min_hmpc is not None and np.min(k_hmpc) < float(k_min_hmpc):
        raise ValueError("k_hmpc contains modes below k_min_hmpc")
    power = linear_matter_power_spectrum(
        k_hmpc, omega_m=omega_m, omega_b=omega_b, h=h,
        n_s=n_s, sigma8=sigma8, omega_lambda=omega_lambda,
    )
    correlation = linear_correlation_from_power_spectrum(
        radius_mpc_h, k_hmpc, power
    )
    result = {
        "radius_mpc_h": radius_mpc_h,
        "correlation": correlation,
        "k_hmpc": k_hmpc,
        "power": power,
    }
    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(filename, "w") as handle:
            for key, values in result.items():
                handle.create_dataset(key, data=values)
            for key, value in {
                "omega_m": omega_m, "omega_b": omega_b, "h": h,
                "omega_lambda": omega_lambda, "n_s": n_s, "sigma8": sigma8,
            }.items():
                handle.attrs[key] = float(value)
            if k_min_hmpc is not None:
                handle.attrs["k_min_hmpc"] = float(k_min_hmpc)
            handle.attrs["transfer_function"] = "Eisenstein-Hu no-wiggle"
    return result
