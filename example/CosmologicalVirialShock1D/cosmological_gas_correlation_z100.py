"""Adiabatic gas collapse from the z=100 LCDM correlation-function IC."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.example_config import load_example_parameters
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_gas_correlation_z100.yaml"
)


def load_correlation_table(config_filename, runparams):
    filename = Path(runparams["linear_correlation_table_filename"])
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def plot_density_evolution(times, radius, density, virial_radius, scale_factors,
                           filename, ymin=None):
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int)
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected):
        axes[0].loglog(radius, np.maximum(density[index], 1.0e-30),
                       color=color, lw=1.7, label="t = %.2f Gyr" % times[index])
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axes[0].axvline(
                virial_radius[index] / scale_factors[index],
                color=color, ls="--", lw=0.9, alpha=0.65,
            )
    axes[0].set_ylabel(r"proper gas density [code mass / kpc$^3$]")
    if ymin is not None and float(ymin) > 0.0:
        axes[0].set_ylim(bottom=float(ymin))
    axes[0].set_title(
        "Gas density evolution from the z=100 LCDM correlation IC\n"
        "solid: gas density; dashed: corresponding virial radius"
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    finite = np.isfinite(virial_radius) & (virial_radius > 0.0)
    if np.any(finite):
        axes[1].plot(times[finite], virial_radius[finite], "k.-", label=r"$r_{200}$")
    else:
        axes[1].text(
            0.5, 0.5, "no resolved $r_{200}$ yet",
            transform=axes[1].transAxes, ha="center", va="center",
        )
    if times.size > 1:
        axes[1].set_xlim(times[0], times[-1])
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("proper radius [kpc]")
    axes[1].grid(alpha=0.25)
    if np.any(finite):
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_mass_history(history, filename):
    """Plot masses interior to the measured virial, shock, and disc radii."""
    time = history["time_Gyr"]
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    axis.plot(time, history["mvir"], color="black", lw=1.8,
              label=r"$M(<r_{\rm vir})$")
    axis.plot(time, history["mshock"], color="tab:red", lw=1.8,
              label=r"$M(<r_{\rm shock})$")
    axis.plot(time, history["mdisc"], color="tab:blue", lw=1.8,
              label=r"$M(<r_{\rm disc})$")
    axis.set_yscale("log")
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel(r"total mass [$10^{10}\,M_\odot$]")
    axis.set_title("Mass interior to virial, shock, and centrifugal/disc radii\n"
                   "adiabatic gas + live dark matter")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_radius_history(history, filename):
    """Plot the evolving shock, virial, disc, and target-mass radii."""
    time = history["time_Gyr"]
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    axis.plot(time, history["rshock_kpc"], color="tab:red", lw=1.8,
              label=r"$r_{\rm shock}$")
    axis.plot(time, history["rdisc_kpc"], color="tab:blue", lw=1.8,
              label=r"$r_{\rm disc}$")
    axis.plot(time, history["rtarget_kpc"], color="0.45", lw=1.2,
              ls=":", label=r"$r(M_{\rm target})$")
    axis.plot(time, history["rvir_kpc"], color="black", lw=2.0,
              ls="--", marker="o", markevery=max(1, len(time) // 12),
              ms=3.0, label=r"$r_{\rm vir}$")
    axis.set_yscale("log")
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel("proper radius [kpc]")
    axis.set_title("Evolution of shock, virial, and disc radii\n"
                   "adiabatic gas + live dark matter")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def _log_radial_bin_profile(radius, values, weights=None, bin_count=48,
                            log_weighted=False):
    """Return mass-weighted mean values in logarithmic radial bins."""
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(radius) & np.isfinite(values) & np.isfinite(weights)
        & (radius > 0.0) & (values > 0.0) & (weights > 0.0)
    )
    if not np.any(valid):
        return np.empty(0), np.empty(0)
    log_edges = np.linspace(
        np.log10(radius[valid].min()),
        np.log10(radius[valid].max()),
        max(8, int(bin_count)) + 1,
    )
    indices = np.digitize(np.log10(radius[valid]), log_edges) - 1
    centers = []
    binned = []
    for index in range(len(log_edges) - 1):
        selected = indices == index
        if np.any(selected):
            centers.append(10.0 ** (0.5 * (log_edges[index] + log_edges[index + 1])))
            selected_values = values[valid][selected]
            selected_weights = weights[valid][selected]
            if log_weighted:
                binned.append(10.0 ** np.average(
                    np.log10(np.maximum(selected_values, 1.0e-30)),
                    weights=selected_weights,
                ))
            else:
                binned.append(np.average(selected_values, weights=selected_weights))
    return np.asarray(centers), np.asarray(binned)


def plot_temperature_evolution(times, radius, density, temperature, virial_radius,
                               splashback_radius, scale_factors,
                               virial_temperature, filename,
                               minimum_temperature=None,
                               radial_bin_count=32, inner_radius=None,
                               box_boundary=None):
    """Plot temperature against comoving radius and evolving halo markers."""
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int)
    )
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected):
        comoving_radius = radius
        # Reconstruct spherical cell volumes from neighboring cell centers;
        # the common scale-factor volume cancels in the mass weighting.
        cell_edges = np.empty(comoving_radius.size + 1, dtype=float)
        if comoving_radius.size > 1:
            cell_edges[1:-1] = np.sqrt(comoving_radius[:-1] * comoving_radius[1:])
            cell_edges[0] = comoving_radius[0] ** 2 / cell_edges[1]
            cell_edges[-1] = comoving_radius[-1] ** 2 / cell_edges[-2]
        else:
            cell_edges[:] = (0.5 * comoving_radius[0], 1.5 * comoving_radius[0])
        cell_volume = np.maximum(np.diff(cell_edges ** 3), 0.0)
        mass_weight = np.asarray(density[index], dtype=float) * cell_volume
        binned_radius, binned_temperature = _log_radial_bin_profile(
            comoving_radius, temperature[index], weights=mass_weight,
            bin_count=radial_bin_count, log_weighted=True,
        )
        axes[0].loglog(
            binned_radius, np.maximum(binned_temperature, 1.0e-30),
            color=color, lw=1.7, label="t = %.2f Gyr" % times[index],
        )
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axes[0].axvline(
                virial_radius[index] / scale_factors[index],
                color=color, ls="--", lw=0.9, alpha=0.65,
            )
        if np.isfinite(virial_temperature[index]) and virial_temperature[index] > 0.0:
            axes[0].axhline(
                virial_temperature[index], color=color, ls=":", lw=1.0,
                alpha=0.7,
            )
        if np.isfinite(splashback_radius[index]) and splashback_radius[index] > 0.0:
            splashback_comoving = splashback_radius[index] / scale_factors[index]
            axes[0].axvline(
                splashback_comoving, color=color, ls="-.", lw=1.2, alpha=0.85,
            )
            if binned_radius.size and binned_temperature.size:
                temperature_at_splashback = np.interp(
                    splashback_comoving, binned_radius, binned_temperature,
                    left=np.nan, right=np.nan,
                )
                if np.isfinite(temperature_at_splashback) and temperature_at_splashback > 0.0:
                    axes[0].plot(
                        splashback_comoving, temperature_at_splashback,
                        marker="s", ms=4.5, color=color, mec="black", mew=0.35,
                        linestyle="None", zorder=5,
                    )
    if inner_radius is not None and float(inner_radius) > 0.0:
        axes[0].axvline(
            float(inner_radius), color="black", ls=":", lw=1.4,
            label="inner gas radius",
        )
    if box_boundary is not None and float(box_boundary) > 0.0:
        axes[0].axvline(
            float(box_boundary), color="black", ls="-", lw=1.2,
            label="box boundary",
        )
    axes[0].set_xlabel("comoving radius [kpc]")
    axes[0].set_ylabel("physical gas temperature [K]")
    if minimum_temperature is not None and float(minimum_temperature) > 0.0:
        axes[0].set_ylim(bottom=float(minimum_temperature))
    axes[0].set_title(
        "Gas temperature evolution from the z=100 LCDM IC\n"
        "solid T; dotted Tvir; dashed r200; dash-dot + squares rsp"
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    finite = np.isfinite(virial_radius) & (virial_radius > 0.0)
    if np.any(finite):
        axes[1].plot(
            times[finite], virial_radius[finite] / scale_factors[finite],
            "k.-", label=r"$r_{200}$ (comoving)",
        )
    else:
        axes[1].text(
            0.5, 0.5, "no resolved $r_{200}$ yet",
            transform=axes[1].transAxes, ha="center", va="center",
        )
    if times.size > 1:
        axes[1].set_xlim(times[0], times[-1])
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("comoving radius [kpc]")
    axes[1].grid(alpha=0.25)
    if np.any(finite):
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_temperature_density_evolution(
    times, density, temperature, filename, bin_count=48, ymin=0.1,
    density_to_nH_cm3=1.0,
):
    """Plot cell temperature against physical hydrogen number density."""
    rho_values = (
        np.asarray(density, dtype=float).ravel()
        * float(density_to_nH_cm3)
    )
    temp_values = np.asarray(temperature, dtype=float).ravel()
    valid = (
        np.isfinite(rho_values) & np.isfinite(temp_values)
        & (rho_values > 0.0) & (temp_values > 0.0)
    )
    fig, axis = plt.subplots(figsize=(7.5, 6.0))
    if np.any(valid):
        log_rho = np.log10(rho_values[valid])
        log_temp = np.log10(temp_values[valid])
        bin_count = max(8, int(bin_count))
        rho_edges = np.linspace(log_rho.min(), log_rho.max(), bin_count + 1)
        temp_edges = np.linspace(log_temp.min(), log_temp.max(), bin_count + 1)
        counts, _, _ = np.histogram2d(log_rho, log_temp,
                                      bins=(rho_edges, temp_edges))
        rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
        temp_centers = 0.5 * (temp_edges[:-1] + temp_edges[1:])
        count_max = float(counts.max())
        if count_max > 1.0:
            levels = np.geomspace(1.0, count_max, 16)
        else:
            levels = np.array([0.5, 1.5])
        image = axis.contourf(
            10.0 ** rho_centers,
            10.0 ** temp_centers,
            np.ma.masked_less_equal(counts.T, 0.0),
            levels=levels,
            norm=LogNorm(vmin=1.0, vmax=max(1.0, count_max)),
            cmap="magma",
        )
        fig.colorbar(image, ax=axis, label="cell count")
    axis.set_xlabel(r"physical hydrogen number density $n_H$ [cm$^{-3}$]")
    axis.set_ylabel("physical gas temperature [K]")
    axis.set_xscale("log")
    axis.set_yscale("log")
    if ymin is not None and float(ymin) > 0.0:
        axis.set_ylim(bottom=float(ymin))
    if np.any(valid):
        axis.set_ylim(top=float(np.nanmax(temp_values[valid])))
    axis.set_title("Gas temperature-density distribution")
    axis.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_velocity_evolution(
    times, radius, density, velocity, virial_radius, scale_factors, filename,
    radial_bin_count=48,
):
    """Plot mass-weighted absolute physical radial velocity profiles."""
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int)
    )
    colors = plt.get_cmap("cividis")(np.linspace(0.05, 0.95, selected.size))
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    for color, index in zip(colors, selected):
        proper_radius = radius * scale_factors[index]
        cell_edges = np.empty(proper_radius.size + 1, dtype=float)
        if proper_radius.size > 1:
            cell_edges[1:-1] = np.sqrt(proper_radius[:-1] * proper_radius[1:])
            cell_edges[0] = proper_radius[0] ** 2 / cell_edges[1]
            cell_edges[-1] = proper_radius[-1] ** 2 / cell_edges[-2]
        else:
            cell_edges[:] = (0.5 * proper_radius[0], 1.5 * proper_radius[0])
        mass_weight = np.asarray(density[index], dtype=float) * np.maximum(
            np.diff(cell_edges**3), 0.0
        )
        binned_radius, binned_velocity = _log_radial_bin_profile(
            proper_radius,
            np.maximum(np.asarray(velocity[index], dtype=float), 0.0),
            weights=mass_weight,
            bin_count=radial_bin_count,
        )
        axis.loglog(
            binned_radius,
            np.maximum(binned_velocity, 1.0e-12),
            color=color,
            lw=1.7,
            label="t = %.2f Gyr" % times[index],
        )
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axis.axvline(
                virial_radius[index], color=color, ls="--", lw=0.9, alpha=0.65
            )
    axis.set_xlabel("proper radius [kpc]")
    axis.set_ylabel(r"mass-weighted $|v_r|$ [km s$^{-1}$]")
    axis.set_title(
        "Mass-weighted absolute gas radial velocity from the z=100 LCDM IC\n"
        "solid: $|v_r|$; dashed: corresponding $r_{200}$"
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="best", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_dark_matter_density_evolution(dm_profiles, filename, bin_count=128):
    """Plot smoothed mass-binned physical DM profiles versus radius.

    The saved NPZ retains every live shell.  This figure intentionally uses
    fewer radial bins so that one-shell Poisson structure does not look like
    physical density oscillations.
    """
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    selected = np.unique(
        np.linspace(0, len(dm_profiles) - 1, min(9, len(dm_profiles))).astype(int)
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, selected.size))
    all_comoving = np.concatenate([
        np.asarray(profile["dm_radius_kpc"], dtype=float)
        / float(profile["scale_factor"])
        for profile in dm_profiles
    ])
    plot_bin_count = min(int(bin_count), 48)
    bin_edges = np.geomspace(
        max(1.0e-8, np.nanmin(all_comoving) * 0.9),
        np.nanmax(all_comoving) * 1.1,
        plot_bin_count + 1,
    )
    bin_radii = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    for index, color in zip(selected, colors):
        profile = dm_profiles[index]
        scale_factor = float(profile["scale_factor"])
        radius = np.asarray(profile["dm_radius_kpc"], dtype=float)
        density = np.asarray(profile["dm_density_code"], dtype=float)
        mass = np.asarray(profile["dm_mass"], dtype=float)
        core_mass = float(profile.get("dm_central_core_mass", 0.0))
        core_radius = float(profile.get("dm_central_core_radius_kpc", 0.0)) / scale_factor
        comoving_radius = radius / scale_factor
        valid = (
            np.isfinite(comoving_radius) & np.isfinite(density)
            & np.isfinite(mass) & (comoving_radius > 0.0) & (mass > 0.0)
        )
        shell_radius = comoving_radius[valid]
        shell_mass = mass[valid]
        if shell_radius.size < 1:
            continue
        mass_in_bin, _ = np.histogram(
            shell_radius, bins=bin_edges, weights=shell_mass
        )
        if core_mass > 0.0 and core_radius > 0.0:
            core_bin = int(np.searchsorted(bin_edges, core_radius, side="right") - 1)
            if 0 <= core_bin < mass_in_bin.size:
                mass_in_bin[core_bin] += core_mass
        bin_volume = (
            4.0 * np.pi / 3.0 * scale_factor**3
            * np.diff(bin_edges**3)
        )
        binned_density = mass_in_bin / np.maximum(bin_volume, 1.0e-30)
        valid_bins = binned_density > 0.0
        axis.loglog(
            bin_radii[valid_bins], binned_density[valid_bins],
            color=color, lw=1.6, label="t = %.2f" % profile["time_Gyr"],
        )
    axis.set_xlabel("comoving radius [kpc]")
    axis.set_ylabel(r"dark-matter density [code mass / kpc$^3$]")
    axis.set_title("Live dark-matter density evolution")
    axis.grid(alpha=0.25, which="both")
    axis.legend(title="cosmic time [Gyr]", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_baryon_normalized_density_comparison(
    gas_profiles, dm_profiles, baryon_fraction, filename,
):
    """Compare gas and DM profiles after removing their cosmic fractions."""
    selected = np.unique(
        np.linspace(0, len(gas_profiles) - 1, min(9, len(gas_profiles))).astype(int)
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, selected.size))
    fb = float(baryon_fraction)
    all_comoving = np.concatenate([
        np.asarray(profile["dm_radius_kpc"], dtype=float)
        / float(profile["scale_factor"])
        for profile in dm_profiles
    ])
    bin_edges = np.geomspace(
        max(1.0e-8, np.nanmin(all_comoving) * 0.9),
        np.nanmax(all_comoving) * 1.1, 49,
    )
    bin_radii = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    for color, index in zip(colors, selected):
        gas = gas_profiles[index]
        dm = dm_profiles[index]
        scale_factor = float(gas["scale_factor"])
        gas_radius = np.asarray(gas["radius_proper_kpc"], dtype=float)
        gas_density_raw = np.asarray(gas["density_proper_code"], dtype=float)
        dm_radius = np.asarray(dm["dm_radius_kpc"], dtype=float)
        dm_mass = np.asarray(dm["dm_mass"], dtype=float)
        proper_edges = scale_factor * bin_edges
        gas_edges = np.empty(gas_radius.size + 1)
        gas_edges[1:-1] = np.sqrt(gas_radius[:-1] * gas_radius[1:])
        gas_edges[0] = gas_radius[0] ** 2 / gas_edges[1]
        gas_edges[-1] = gas_radius[-1] ** 2 / gas_edges[-2]
        gas_volume = 4.0 * np.pi / 3.0 * np.diff(gas_edges**3)
        gas_mass_bin, _ = np.histogram(
            gas_radius / scale_factor, bins=bin_edges,
            weights=gas_density_raw * gas_volume,
        )
        dm_mass_bin, _ = np.histogram(
            dm_radius / scale_factor, bins=bin_edges, weights=dm_mass,
        )
        bin_volume = 4.0 * np.pi / 3.0 * np.diff(proper_edges**3)
        gas_density = gas_mass_bin / np.maximum(bin_volume, 1.0e-300) / fb
        dm_density = dm_mass_bin / np.maximum(bin_volume, 1.0e-300) / (1.0 - fb)
        valid = (gas_density > 0.0) & (dm_density > 0.0)
        label = "t = %.2f Gyr" % gas["time_Gyr"]
        axes[0].loglog(bin_radii[valid], gas_density[valid], color=color, lw=1.5, label=label + " gas/$f_b$")
        axes[0].loglog(bin_radii[valid], dm_density[valid], color=color, lw=1.0, ls="--", alpha=0.85, label=label + " DM/$1-f_b$")
        axes[1].semilogx(bin_radii[valid], gas_density[valid] / dm_density[valid], color=color, lw=1.5)
    axes[0].set_ylabel(r"density / cosmic fraction")
    axes[0].set_title("Gas versus dark matter (densities normalized by cosmic fractions)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.25, which="both")
    axes[1].axhline(1.0, color="black", ls=":", lw=1.0)
    axes[1].set_ylabel(r"$(\rho_g/f_b)/(\rho_{DM}/(1-f_b))$")
    axes[1].set_xlabel("proper radius [kpc]")
    axes[1].set_ylim(1.0e-2, 1.0e2)
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def _pad_profile_history(profiles, key):
    """Pack variable-length shell profiles into a NaN-padded 2D array."""
    arrays = [np.asarray(item[key], dtype=float).ravel() for item in profiles]
    width = max((array.size for array in arrays), default=0)
    result = np.full((len(arrays), width), np.nan, dtype=float)
    for row, array in enumerate(arrays):
        result[row, :array.size] = array
    return result


def _energy_audit_state(sim):
    """Return conserved gas-energy diagnostics for the physical cells."""
    first = int(sim.par.noghost)
    last = first + int(sim.par.nogrid)
    rho = np.asarray(sim.fluid.rho[first:last], dtype=float)
    vel = np.asarray(sim.fluid.vel[first:last], dtype=float)
    volume = np.asarray(sim.mesh.vol[first:last], dtype=float)
    mass = np.asarray(sim.fluid.Mass[first:last], dtype=float)
    total_energy = np.asarray(sim.fluid.Energy[first:last], dtype=float)
    kinetic_density = 0.5 * rho * vel**2
    kinetic_energy = float(np.sum(kinetic_density * volume))
    total_energy_value = float(np.sum(total_energy))
    return {
        "total_gas_mass": float(np.sum(mass)),
        "total_gas_energy": total_energy_value,
        "kinetic_energy": kinetic_energy,
        "thermal_energy": total_energy_value - kinetic_energy,
    }


def run(config_filename=DEFAULT_CONFIG, final_time_override=None,
        output_suffix=None):
    config_filename = Path(config_filename).resolve()
    runparams, icparams = load_example_parameters(config_filename)
    units = CodeUnits.from_mapping(runparams["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams["cosmology_t_ref"]),
        a_ref=float(runparams["cosmology_a_ref"]),
    )
    correlation_table = load_correlation_table(config_filename, runparams)
    output_dir = Path(runparams["savedir"])
    figure_prefix = str(
        runparams.get("figure_prefix", "CosmologicalGasCorrelationZ100")
    )
    if output_suffix:
        output_dir = output_dir.with_name(output_dir.name + str(output_suffix))
        figure_prefix += str(output_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_filename = output_dir / "InitialCondition.hdf5"

    initial = et.Simwrap(
        icparams, units, cosmology, correlation_table=correlation_table
    )
    rio.writehdf5(initial, ic_filename)
    dm = et.make_dark_matter(
        icparams, units, cosmology, correlation_table=correlation_table
    )

    baryon_fraction = float(icparams["baryon_fraction"])
    gas_mass = float(np.sum(initial.fluid.rho * initial.mesh.vol))
    dm_mass = float(np.sum(dm.mass))
    measured_fraction = gas_mass / max(gas_mass + dm_mass, 1.0e-30)
    if not np.isclose(measured_fraction, baryon_fraction, rtol=0.02):
        raise RuntimeError(
            "initial gas/total mass fraction does not match baryon_fraction"
        )
    initial_temperature = float(np.median(initial.fluid.temp)) / float(
        cosmology.scale_factor(float(icparams["initial_cosmic_time"]))
    ) ** 2
    expected_temperature = float(icparams["cmb_temperature_0"]) * (
        1.0 / float(cosmology.scale_factor(float(icparams["initial_cosmic_time"])))
    )
    if not np.isclose(initial_temperature, expected_temperature, rtol=1.0e-8):
        raise RuntimeError("initial gas temperature is not the z=100 CMB temperature")

    local = dict(runparams)
    local.update({"ICfilename": str(ic_filename), "outdir": str(output_dir),
                  "savedir": str(output_dir)})
    sim = Rsim(local)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.fluid.time = float(np.asarray(sim.par.time).flat[0])
    dm_for_gas = (
        et.VolumeSmoothedDarkMatter(dm)
        if bool(runparams.get("smooth_dm_force_for_gas", False))
        else dm
    )
    sim.par.gravity = Gravity(
        selfgravity=True, cosmological=True, cosmology=sim.par.cosmology,
        dark_matter=dm_for_gas, code_units=sim.par.CodeUnits,
    )
    sim.par.dark_matter = dm
    sim.par.dark_matter_background_fraction = 1.0 - baryon_fraction
    sim.par.gas_background_fraction = baryon_fraction

    initial_time = float(icparams["initial_cosmic_time"])
    initial_a = float(cosmology.scale_factor(initial_time))
    sim.par.mu_inflow = float(icparams.get("mu", 0.59))
    minimum_temperature = runparams.get("minimum_temperature", None)
    if minimum_temperature is not None:
        if hasattr(minimum_temperature, "to_value"):
            minimum_temperature = float(minimum_temperature.to_value("K"))
        else:
            minimum_temperature = float(minimum_temperature)

    transition_redshift = runparams.get("thermochemistry_transition_redshift")
    transition_tau = None
    if transition_redshift is not None:
        transition_redshift = float(transition_redshift)
        transition_scale_factor = 1.0 / (1.0 + transition_redshift)
        transition_time = float(
            cosmology.t_ref
            * (transition_scale_factor / cosmology.a_ref) ** 1.5
        )
        transition_tau = float(cosmology.supercomoving_time(transition_time))

    active_network = None

    def configure_thermochemistry(cosmic_time):
        """Select the pre/post-transition source at the current redshift."""
        nonlocal active_network
        if transition_redshift is None:
            return
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        redshift = max(0.0, 1.0 / scale_factor - 1.0)
        if redshift > transition_redshift:
            sim.par.thermochemistry_network = "hydrogen"
            sim.par.metal_pie_enabled = False
            sim.par.hydrogen_chemistry = True
            sim.par.hydrogen_recombination = True
            sim.par.hydrogen_collisional_ionization = True
            sim.par.hydrogen_atomic_cooling = True
            sim.par.hydrogen_update_mu = True
            sim.par.hydrogen_thermal_coupling = True
            sim.par.compton_cmb_enabled = True
        else:
            sim.par.thermochemistry_network = "pie_uvbg_cooling"
            sim.par.metal_pie_enabled = True
            sim.par.metal_pie_redshift = transition_redshift
            sim.par.hydrogen_chemistry = False
            sim.par.hydrogen_update_mu = False
            sim.par.hydrogen_thermal_coupling = False
            sim.par.compton_cmb_enabled = False
        if sim.par.thermochemistry_network != active_network:
            active_network = sim.par.thermochemistry_network
            print(
                "thermochemistry=%s at z=%.6g"
                % (active_network, redshift),
                flush=True,
            )

    def update_cosmic_boundary(cosmic_time):
        """Set the outer cosmic gas state in supercomoving hydro units."""
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        background_physical = float(cosmology.background_density(cosmic_time))
        # The IC is initialized in CMB equilibrium at the starting redshift,
        # so its physical temperature is T_CMB,0 / a_initial.  Evolve that
        # state adiabatically (T proportional to a^-2) for the outer gas.
        temperature_initial = float(icparams["cmb_temperature_0"]) / initial_a
        temperature_physical = temperature_initial * (
            initial_a / scale_factor
        ) ** 2

        # The boundary is specified physically, then converted explicitly to
        # the hydro representation.  For this gamma=5/3 supercomoving case,
        # rho_code = rho_phys*a^3 and T_code = T_phys*a^2; both happen to be
        # constant for a homogeneous adiabatic background, as they should.
        sim.par.rho_inflow = baryon_fraction * background_physical * scale_factor**3
        sim.par.vel_inflow = 0.0
        sim.par.temp_inflow = temperature_physical * scale_factor**2
        sim.par.compton_cmb_redshift = 1.0 / scale_factor - 1.0
        # Hydro stores supercomoving temperature; keep the physical floor at
        # the configured value as the scale factor changes.
        sim.par.hydro_temperature_floor = (
            None
            if minimum_temperature is None
            else minimum_temperature * scale_factor**2
        )

    def preserve_outer_background_cell():
        """Reset the outer active cell to the analytic EdS reservoir state."""
        first = int(sim.par.noghost)
        index = first + int(sim.par.nogrid) - 1
        old_mass = float(np.asarray(sim.fluid.Mass, dtype=float)[index])
        old_energy = float(np.asarray(sim.fluid.Energy, dtype=float)[index])
        rho = float(np.asarray(sim.par.rho_inflow, dtype=float))
        velocity = float(np.asarray(sim.par.vel_inflow, dtype=float))
        temperature = float(np.asarray(sim.par.temp_inflow, dtype=float))
        mu = float(np.asarray(sim.par.mu_inflow, dtype=float))
        volume = float(np.asarray(sim.mesh.vol, dtype=float)[index])
        pressure = float(np.asarray(
            sim.fluid.eos.pressure(rho, temperature, mu), dtype=float
        ))
        sim.fluid.rho[index] = rho
        sim.fluid.vel[index] = velocity
        sim.fluid.temp[index] = temperature
        sim.fluid.mu[index] = mu
        sim.fluid.pre[index] = pressure
        sim.fluid.Mass[index] = rho * volume
        sim.fluid.Mom[index] = rho * velocity * volume
        sim.fluid.Energy[index] = float(np.asarray(
            sim.fluid.eos.total_energy_density(rho, velocity, pressure),
            dtype=float,
        )) * volume
        thermal_energy_density = float(np.asarray(
            sim.fluid.eos.thermal_energy_density(pressure), dtype=float,
        ))
        if hasattr(sim.fluid, "eth"):
            sim.fluid.eth[index] = thermal_energy_density
        if hasattr(sim.fluid, "InternalEnergy"):
            # SetConserved intentionally preserves the active-cell dual-energy
            # field.  The explicitly reset EdS reservoir must therefore
            # synchronize its conserved thermal energy here as well.
            sim.fluid.InternalEnergy[index] = thermal_energy_density * volume
        return (
            float(np.asarray(sim.fluid.Mass, dtype=float)[index]) - old_mass,
            float(np.asarray(sim.fluid.Energy, dtype=float)[index]) - old_energy,
        )

    configure_thermochemistry(initial_time)
    update_cosmic_boundary(initial_time)
    preserve_outer_background_cell()
    # The timestep estimate must see the current comoving reservoir in the
    # ghost cells, rather than the default InflowSph state from SetInitFluid.
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, sim.fluid)

    final_time = (
        float(final_time_override)
        if final_time_override is not None
        else float(runparams["final_cosmic_time"])
    )
    target_tau = float(cosmology.supercomoving_time(final_time))
    cadence = float(runparams.get("gas_profile_cadence", 0.10))
    next_snapshot = initial_time
    gas_profiles = []
    radius_history = []
    dm_profiles = []
    steps = 0

    def save_snapshot(cosmic_time):
        gas_profile = et.gas_density_profile(sim, cosmic_time, cosmology)
        first = int(sim.par.noghost)
        last = first + int(sim.par.nogrid)
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        gas_profile["temperature_physical_K"] = (
            np.asarray(sim.fluid.temp[first:last], dtype=float) / scale_factor**2
        )
        physical_velocity = cosmology.physical_velocity(
            np.asarray(sim.mesh.coordinate[first:last], dtype=float),
            np.asarray(sim.fluid.vel[first:last], dtype=float),
            float(sim.fluid.time),
        )
        signed_velocity_km_s = (
            np.asarray(physical_velocity, dtype=float)
            * float(sim.par.CodeUnits.velocity_in_cgs) / 1.0e5
        )
        gas_profile["radial_velocity_physical_km_s"] = signed_velocity_km_s
        gas_profile["velocity_physical_km_s"] = np.abs(signed_velocity_km_s)
        gas_profiles.append(gas_profile)
        radius_history.append(et.profiles(sim, dm, cosmic_time, cosmology, icparams))
        dm_profile = et.density_profiles(sim, dm, cosmic_time, cosmology)
        dm_profile["scale_factor"] = scale_factor
        dm_profiles.append(dm_profile)

    save_snapshot(initial_time)
    audit_initial = _energy_audit_state(sim)
    energy_audit = {
        "step": [0],
        "time_Gyr": [initial_time * sim.par.CodeUnits.time_unit.to_value("Gyr")],
        "dt": [0.0],
        "scale_factor": [initial_a],
        **{key: [value] for key, value in audit_initial.items()},
        "gravitational_work": [0.0],
        "hydro_boundary_energy_flux": [0.0],
        "background_reservoir_mass_change": [0.0],
        "background_reservoir_energy_change": [0.0],
        "thermochemistry_energy_change": [0.0],
        "energy_closure_residual": [0.0],
        "inner_wall_momentum_flux": [0.0],
        "inner_wall_energy_flux": [0.0],
    }
    next_snapshot += cadence
    while float(sim.fluid.time) < target_tau - 1.0e-12:
        cosmic_start = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        configure_thermochemistry(cosmic_start)
        update_cosmic_boundary(cosmic_start)
        preserve_outer_background_cell()
        # Keep the outer ghost reservoir synchronized before GetStepTime().
        sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
        sim.solver.SetConserved(sim.mesh, sim.fluid)
        dt = min(float(sim.GetStepTime()), target_tau - float(sim.fluid.time))
        if transition_tau is not None and float(sim.fluid.time) < transition_tau:
            dt = min(dt, transition_tau - float(sim.fluid.time))
        # Capture the finite inner-wall Riemann flux before Step refreshes the
        # temporary face arrays.
        wall_face = int(sim.par.noghost)
        sim.solver.SetInterFaceFlux(
            sim.mesh, sim.fluid, sim.par.boundcond,
            method=getattr(sim.par, "riemann_solver", "Rusanov"),
            order=int(sim.par.order),
        )
        wall_momentum_flux = float(
            np.asarray(sim.fluid.Mom.flux, dtype=float)[wall_face]
        )
        wall_energy_flux = float(
            np.asarray(sim.fluid.Energy.flux, dtype=float)[wall_face]
        )
        # This run has active Compton/atomic or PIE thermal sources.  Using
        # hydro-only mode would select the networks but never apply their
        # energy update.
        sim.Step(dt=dt, mode="hydro_sources")
        energy_audit["inner_wall_momentum_flux"].append(
            wall_momentum_flux
        )
        energy_audit["inner_wall_energy_flux"].append(
            wall_energy_flux
        )
        steps += 1
        cosmic_time = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        update_cosmic_boundary(cosmic_time)
        reservoir_mass_change, reservoir_energy_change = (
            preserve_outer_background_cell()
        )
        audit_state = _energy_audit_state(sim)
        previous_energy = energy_audit["total_gas_energy"][-1]
        energy_change = audit_state["total_gas_energy"] - previous_energy
        gravity_work = float(getattr(sim, "last_gravity_work", 0.0))
        boundary_flux = float(
            getattr(sim, "last_hydro_boundary_energy_flux", 0.0)
        )
        thermo_change = float(
            getattr(sim, "last_thermochemistry_energy_change", 0.0)
        )
        for key, value in audit_state.items():
            energy_audit[key].append(value)
        energy_audit["step"].append(steps)
        energy_audit["time_Gyr"].append(
            cosmic_time * sim.par.CodeUnits.time_unit.to_value("Gyr")
        )
        energy_audit["dt"].append(dt)
        energy_audit["scale_factor"].append(
            float(cosmology.scale_factor(cosmic_time))
        )
        energy_audit["gravitational_work"].append(gravity_work)
        energy_audit["hydro_boundary_energy_flux"].append(boundary_flux)
        energy_audit["background_reservoir_mass_change"].append(
            reservoir_mass_change
        )
        energy_audit["background_reservoir_energy_change"].append(
            reservoir_energy_change
        )
        energy_audit["thermochemistry_energy_change"].append(thermo_change)
        energy_audit["energy_closure_residual"].append(
            energy_change
            - boundary_flux
            - reservoir_energy_change
            - gravity_work
            - thermo_change
        )
        if steps == 1 or steps % 100 == 0:
            first = int(sim.par.noghost)
            last = first + int(sim.par.nogrid)
            inner = slice(first, min(last, first + 16))
            print(
                "step=%d cosmic_time=%.6g dt=%.6g crossing_dt=%.6g "
                "Tmax_inner=%.6g vmax_inner=%.6g csmax_inner=%.6g"
                % (
                    steps, cosmic_time, dt, dm.crossing_timestep(),
                    np.nanmax(np.asarray(sim.fluid.temp[inner], dtype=float)),
                    np.nanmax(np.abs(np.asarray(sim.fluid.vel[inner], dtype=float))),
                    np.nanmax(np.asarray(sim.fluid.cs[inner], dtype=float)),
                ),
                flush=True,
            )
        if cosmic_time >= next_snapshot or cosmic_time >= final_time - 1.0e-10:
            save_snapshot(cosmic_time)
            while next_snapshot <= cosmic_time + 1.0e-12:
                next_snapshot += cadence

    times = np.asarray([item["time_Gyr"] for item in gas_profiles])
    radius = np.asarray(gas_profiles[0]["radius_comoving_kpc"])
    density = np.asarray([item["density_proper_code"] for item in gas_profiles])
    temperature = np.asarray(
        [item["temperature_physical_K"] for item in gas_profiles]
    )
    velocity = np.asarray(
        [item["velocity_physical_km_s"] for item in gas_profiles]
    )
    radial_velocity = np.asarray(
        [item["radial_velocity_physical_km_s"] for item in gas_profiles]
    )
    scale_factors = np.asarray([item["scale_factor"] for item in gas_profiles])
    virial_radius = np.asarray([item["rvir_kpc"] for item in radius_history])
    splashback_radius = np.asarray(
        [item["rsplashback_kpc"] for item in radius_history]
    )
    virial_temperature = np.asarray([item["tvir_K"] for item in radius_history])
    history = {
        key: np.asarray([item[key] for item in radius_history])
        for key in radius_history[0]
    }
    data_file = output_dir / (figure_prefix + ".npz")
    np.savez(data_file, **history, scale_factor=scale_factors,
             radius_comoving_kpc=radius, density_proper_code=density,
             temperature_physical_K=temperature,
             velocity_physical_km_s=velocity,
             radial_velocity_physical_km_s=radial_velocity,
             rvir_proper_kpc=virial_radius,
             virial_temperature_K=virial_temperature)
    energy_audit_file = output_dir / (figure_prefix + "_EnergyAudit.npz")
    np.savez(energy_audit_file, **{
        key: np.asarray(value, dtype=float)
        for key, value in energy_audit.items()
    })
    figure = output_dir / (figure_prefix + ".jpg")
    radius_figure = output_dir / (figure_prefix + "_Radii.jpg")
    plot_mass_history(history, figure)
    plot_radius_history(history, radius_figure)
    temperature_figure = output_dir / (figure_prefix + "_Temperatures.jpg")
    temperature_plot_ymin = runparams.get(
        "temperature_plot_ymin", minimum_temperature
    )
    if hasattr(temperature_plot_ymin, "to_value"):
        temperature_plot_ymin = float(temperature_plot_ymin.to_value("K"))
    else:
        temperature_plot_ymin = float(temperature_plot_ymin)
    plot_temperature_evolution(
        times, radius, density, temperature, virial_radius, splashback_radius,
        scale_factors,
        virial_temperature,
        temperature_figure,
        minimum_temperature=temperature_plot_ymin,
        inner_radius=float(icparams.get("inner_wall_radius_comoving", icparams["rmin"])),
        box_boundary=float(icparams["rmax"]),
    )
    density_figure = output_dir / (figure_prefix + "_Densities.jpg")
    cosmic_gas_density_z0 = baryon_fraction * float(
        cosmology.background_density(cosmology.t_ref)
    )
    plot_density_evolution(
        times, radius, density, virial_radius, scale_factors, density_figure,
        ymin=0.1 * cosmic_gas_density_z0,
    )
    temperature_density_figure = output_dir / (
        figure_prefix + "_TemperatureDensity.jpg"
    )
    plot_temperature_density_evolution(
        times, density, temperature, temperature_density_figure, ymin=0.1,
        density_to_nH_cm3=(
            float(sim.par.CodeUnits.mass_in_cgs)
            / float(sim.par.CodeUnits.length_in_cgs) ** 3
            * float(icparams["hydrogen_mass_fraction"])
            / PROTON_MASS_CGS
        ),
    )
    velocity_figure = output_dir / (figure_prefix + "_Velocity.jpg")
    plot_velocity_evolution(
        times, radius, density, velocity, virial_radius, scale_factors,
        velocity_figure,
    )
    dm_figure = output_dir / (figure_prefix + "_DarkMatterDensities.jpg")
    plot_dark_matter_density_evolution(
        dm_profiles, dm_figure,
        bin_count=int(runparams.get("dm_density_bins", 128)),
    )
    density_comparison_figure = output_dir / (
        figure_prefix + "_GasDarkMatterBaryonNormalized.jpg"
    )
    plot_baryon_normalized_density_comparison(
        gas_profiles, dm_profiles, icparams["baryon_fraction"],
        density_comparison_figure,
    )
    dm_data_file = output_dir / (figure_prefix + "_DarkMatterDensities.npz")
    np.savez(
        dm_data_file,
        time_Gyr=np.asarray([item["time_Gyr"] for item in dm_profiles]),
        scale_factor=np.asarray([item["scale_factor"] for item in dm_profiles]),
        radius_kpc=_pad_profile_history(dm_profiles, "dm_radius_kpc"),
        density_code=_pad_profile_history(dm_profiles, "dm_density_code"),
        mass=_pad_profile_history(dm_profiles, "dm_mass"),
        central_core_mass=np.asarray([
            item.get("dm_central_core_mass", 0.0) for item in dm_profiles
        ]),
        central_core_radius_kpc=np.asarray([
            item.get("dm_central_core_radius_kpc", 0.0) for item in dm_profiles
        ]),
    )
    print("initial gas fraction = %.8g" % measured_fraction)
    print("initial gas temperature = %.8g K" % initial_temperature)
    print("final cosmic time = %.8g Gyr" % times[-1])
    dm_substeps = np.asarray(sim.dark_matter_substep_history, dtype=int)
    if dm_substeps.size:
        print(
            "dark-matter substeps per hydro step = %.8g mean, %d max, %d total"
            % (
                np.mean(dm_substeps),
                np.max(dm_substeps),
                np.sum(dm_substeps),
            )
        )
    print("data = %s" % data_file)
    print("figure = %s" % figure)
    print("radius figure = %s" % radius_figure)
    print("temperature figure = %s" % temperature_figure)
    print("temperature-density figure = %s" % temperature_density_figure)
    print("velocity figure = %s" % velocity_figure)
    print("dark-matter figure = %s" % dm_figure)
    print("dark-matter data = %s" % dm_data_file)
    print("gas/DM density comparison = %s" % density_comparison_figure)
    print("energy audit = %s" % energy_audit_file)
    return data_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--final-time", type=float, default=None,
        help="override final cosmic time in Gyr for a short debug run",
    )
    parser.add_argument(
        "--output-suffix", default=None,
        help="append a suffix to the output directory and figure prefix",
    )
    args = parser.parse_args()
    run(
        args.config,
        final_time_override=args.final_time,
        output_suffix=args.output_suffix,
    )
