# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Plotting and numerical helper functions for the cosmological gas example."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc


def _add_redshift_top_axis(axis, times, scale_factors):
    """Add redshift ticks above a cosmic-time x-axis."""
    times = np.asarray(times, dtype=float)
    scale_factors = np.asarray(scale_factors, dtype=float)
    valid = np.isfinite(times) & np.isfinite(scale_factors) & (scale_factors > 0.0)
    if not np.any(valid):
        return
    top_axis = axis.twiny()
    top_axis.set_xlim(axis.get_xlim())
    selected = np.unique(
        np.linspace(
            np.flatnonzero(valid)[0],
            np.flatnonzero(valid)[-1],
            min(7, int(np.count_nonzero(valid))),
        ).astype(int),
    )
    selected = selected[valid[selected]]
    top_axis.set_xticks(times[selected])
    top_axis.set_xticklabels(
        ["%.0f" % (1.0 / scale_factors[index] - 1.0) for index in selected],
    )
    top_axis.set_xlabel("redshift")


def plot_density_evolution(
    times,
    radius_comoving_code,
    rho_comoving_code,
    virial_radius,
    scale_factors,
    filename,
    ymin=None,
):
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int),
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected, strict=False):
        axes[0].loglog(
            radius_comoving_code,
            np.maximum(rho_comoving_code[index], 1.0e-30),
            color=color,
            lw=1.7,
            label=f"t = {times[index]:.2f} Gyr",
        )
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axes[0].axvline(
                virial_radius[index] / scale_factors[index],
                color=color,
                ls="--",
                lw=0.9,
                alpha=0.65,
            )
    axes[0].set_ylabel(r"proper gas density [code mass / kpc$^3$]")
    if ymin is not None and float(ymin) > 0.0:
        axes[0].set_ylim(bottom=float(ymin))
    axes[0].set_title(
        "Gas density evolution from the z=100 LCDM correlation IC\n"
        "solid: gas density; dashed: corresponding virial radius",
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    finite = np.isfinite(virial_radius) & (virial_radius > 0.0)
    if np.any(finite):
        axes[1].plot(times[finite], virial_radius[finite], "k.-", label=r"$r_{200}$")
    else:
        axes[1].text(
            0.5,
            0.5,
            "no resolved $r_{200}$ yet",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    if times.size > 1:
        axes[1].set_xlim(times[0], times[-1])
    axes[1].set_xlabel("cosmic time [Gyr]")
    _add_redshift_top_axis(axes[1], times, scale_factors)
    axes[1].set_ylabel("proper radius [kpc]")
    axes[1].grid(alpha=0.25)
    if np.any(finite):
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_mass_history(history, filename):
    """Plot masses interior to the measured virial, shock, and disc radii."""
    time_cosmic_code = history["time_cosmic_Gyr"]
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    axis.plot(time_cosmic_code, history["mvir"], color="black", lw=1.8, label=r"$M(<r_{\rm vir})$")
    axis.plot(
        time_cosmic_code,
        history["mshock"],
        color="tab:red",
        lw=1.8,
        label=r"$M(<r_{\rm shock})$",
    )
    axis.plot(
        time_cosmic_code,
        history["mdisc"],
        color="tab:blue",
        lw=1.8,
        label=r"$M(<r_{\rm disc})$",
    )
    axis.set_yscale("log")
    axis.set_xlabel("cosmic time [Gyr]")
    _add_redshift_top_axis(axis, time_cosmic_code, history["scale_factor"])
    axis.set_ylabel(r"total mass [$10^{10}\,M_\odot$]")
    axis.set_title(
        "Mass interior to virial, shock, and centrifugal/disc radii\n"
        "adiabatic gas + live dark matter",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_radius_history(history, filename):
    """Plot the evolving shock, virial, disc, and target-mass radii."""
    time_cosmic_code = history["time_cosmic_Gyr"]
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    axis.plot(
        time_cosmic_code,
        history["rshock_kpc"],
        color="tab:red",
        lw=1.8,
        label=r"$r_{\rm shock}$",
    )
    axis.plot(
        time_cosmic_code,
        history["rdisc_kpc"],
        color="tab:blue",
        lw=1.8,
        label=r"$r_{\rm disc}$",
    )
    axis.plot(
        time_cosmic_code,
        history["rtarget_kpc"],
        color="0.45",
        lw=1.2,
        ls=":",
        label=r"$r(M_{\rm target})$",
    )
    axis.plot(
        time_cosmic_code,
        history["rvir_kpc"],
        color="black",
        lw=2.0,
        ls="--",
        marker="o",
        markevery=max(1, len(time_cosmic_code) // 12),
        ms=3.0,
        label=r"$r_{\rm vir}$",
    )
    axis.set_yscale("log")
    axis.set_xlabel("cosmic time [Gyr]")
    _add_redshift_top_axis(axis, time_cosmic_code, history["scale_factor"])
    axis.set_ylabel("proper radius [kpc]")
    axis.set_title("Evolution of shock, virial, and disc radii\nadiabatic gas + live dark matter")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def _log_radial_bin_profile(
    radius_comoving_code,
    values,
    weights=None,
    bin_count=48,
    *,
    log_weighted=False,
):
    """Return mass-weighted mean values in logarithmic radial bins."""
    radius_comoving_code = np.asarray(radius_comoving_code, dtype=float)
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(radius_comoving_code)
        & np.isfinite(values)
        & np.isfinite(weights)
        & (radius_comoving_code > 0.0)
        & (values > 0.0)
        & (weights > 0.0)
    )
    if not np.any(valid):
        return np.empty(0), np.empty(0)
    log_edges = np.linspace(
        np.log10(radius_comoving_code[valid].min()),
        np.log10(radius_comoving_code[valid].max()),
        max(8, int(bin_count)) + 1,
    )
    indices = np.digitize(np.log10(radius_comoving_code[valid]), log_edges) - 1
    centers = []
    binned = []
    for index in range(len(log_edges) - 1):
        selected = indices == index
        if np.any(selected):
            centers.append(10.0 ** (0.5 * (log_edges[index] + log_edges[index + 1])))
            selected_values = values[valid][selected]
            selected_weights = weights[valid][selected]
            if log_weighted:
                binned.append(
                    10.0
                    ** np.average(
                        np.log10(np.maximum(selected_values, 1.0e-30)),
                        weights=selected_weights,
                    ),
                )
            else:
                binned.append(np.average(selected_values, weights=selected_weights))
    return np.asarray(centers), np.asarray(binned)


def plot_temperature_evolution(
    times,
    radius_comoving_code,
    rho_comoving_code,
    temperature_proper_cgs_K,  # noqa: N803
    virial_radius,
    splashback_radius,
    scale_factors,
    virial_temperature,
    filename,
    minimum_temperature=None,
    radial_bin_count=32,
    inner_radius=None,
    box_boundary=None,
):
    """Plot temperature against comoving radius and evolving halo markers."""
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int),
    )
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected, strict=False):
        comoving_radius = radius_comoving_code
        # Reconstruct spherical cell volumes from neighboring cell centers;
        # the common scale-factor volume_comoving_code cancels in the mass weighting.
        cell_edges = np.empty(comoving_radius.size + 1, dtype=float)
        if comoving_radius.size > 1:
            cell_edges[1:-1] = np.sqrt(comoving_radius[:-1] * comoving_radius[1:])
            cell_edges[0] = comoving_radius[0] ** 2 / cell_edges[1]
            cell_edges[-1] = comoving_radius[-1] ** 2 / cell_edges[-2]
        else:
            cell_edges[:] = (0.5 * comoving_radius[0], 1.5 * comoving_radius[0])
        cell_volume = np.maximum(np.diff(cell_edges**3), 0.0)
        mass_weight = np.asarray(rho_comoving_code[index], dtype=float) * cell_volume
        binned_radius, binned_temperature = _log_radial_bin_profile(
            comoving_radius,
            temperature_proper_cgs_K[index],
            weights=mass_weight,
            bin_count=radial_bin_count,
            log_weighted=True,
        )
        axes[0].loglog(
            binned_radius,
            np.maximum(binned_temperature, 1.0e-30),
            color=color,
            lw=1.7,
            label=f"t = {times[index]:.2f} Gyr",
        )
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axes[0].axvline(
                virial_radius[index] / scale_factors[index],
                color=color,
                ls="--",
                lw=0.9,
                alpha=0.65,
            )
        if np.isfinite(virial_temperature[index]) and virial_temperature[index] > 0.0:
            axes[0].axhline(
                virial_temperature[index],
                color=color,
                ls=":",
                lw=1.0,
                alpha=0.7,
            )
        cmb_temperature = 2.7255 / scale_factors[index]
        axes[0].axhline(
            cmb_temperature,
            color=color,
            ls="--",
            lw=0.55,
            alpha=0.65,
            label="CMB temperature" if index == selected[0] else None,
        )
    if inner_radius is not None and float(inner_radius) > 0.0:
        axes[0].axvline(
            float(inner_radius),
            color="black",
            ls=":",
            lw=1.4,
            label="inner gas radius",
        )
    if box_boundary is not None and float(box_boundary) > 0.0:
        axes[0].axvline(
            float(box_boundary),
            color="black",
            ls="-",
            lw=1.2,
            label="box boundary",
        )
    axes[0].set_xlabel("comoving radius [kpc]")
    axes[0].set_ylabel("physical gas temperature [K]")
    if minimum_temperature is not None and float(minimum_temperature) > 0.0:
        axes[0].set_ylim(bottom=float(minimum_temperature))
    axes[0].set_title(
        "Gas temperature evolution from the z=100 LCDM IC\nsolid T; dotted Tvir; dashed r200",
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    finite = np.isfinite(virial_radius) & (virial_radius > 0.0)
    if np.any(finite):
        axes[1].plot(
            times[finite],
            virial_radius[finite] / scale_factors[finite],
            "k.-",
            label=r"$r_{200}$ (comoving)",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "no resolved $r_{200}$ yet",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    if times.size > 1:
        axes[1].set_xlim(times[0], times[-1])
    axes[1].set_xlabel("cosmic time [Gyr]")
    _add_redshift_top_axis(axes[1], times, scale_factors)
    axes[1].set_ylabel("comoving radius [kpc]")
    axes[1].grid(alpha=0.25)
    if np.any(finite):
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_specific_angular_momentum_evolution(
    times,
    radius_comoving_code,
    rho_comoving_code,
    specific_angular_momentum,
    virial_radius,
    splashback_radius,
    scale_factors,
    filename,
    radial_bin_count=32,
):
    """Plot the signed gas specific-angular-momentum profile evolution."""
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int),
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected, strict=False):
        values = np.asarray(specific_angular_momentum[index], dtype=float)
        cell_edges = np.empty(radius_comoving_code.size + 1, dtype=float)
        if radius_comoving_code.size > 1:
            cell_edges[1:-1] = np.sqrt(radius_comoving_code[:-1] * radius_comoving_code[1:])
            cell_edges[0] = radius_comoving_code[0] ** 2 / cell_edges[1]
            cell_edges[-1] = radius_comoving_code[-1] ** 2 / cell_edges[-2]
        else:
            cell_edges[:] = (0.5 * radius_comoving_code[0], 1.5 * radius_comoving_code[0])
        mass_weight = np.asarray(rho_comoving_code[index], dtype=float) * np.maximum(
            np.diff(cell_edges**3),
            0.0,
        )
        binned_radius, binned_j = _log_radial_bin_profile(
            radius_comoving_code,
            values,
            weights=mass_weight,
            bin_count=radial_bin_count,
            log_weighted=False,
        )
        axes[0].semilogx(
            binned_radius,
            binned_j,
            color=color,
            lw=1.7,
            label=f"t = {times[index]:.2f} Gyr",
        )
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axes[0].axvline(
                virial_radius[index] / scale_factors[index],
                color=color,
                ls="--",
                lw=0.9,
                alpha=0.65,
            )
    axes[0].set_xlabel("comoving radius [kpc]")
    axes[0].set_ylabel("specific angular momentum [code units]")
    axes[0].set_title(
        "Gas specific-angular-momentum evolution from the z=100 LCDM IC\n"
        "solid: j; dashed: corresponding virial radius",
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    finite = np.isfinite(virial_radius) & (virial_radius > 0.0)
    if np.any(finite):
        axes[1].plot(
            times[finite],
            virial_radius[finite] / scale_factors[finite],
            "k.-",
            label=r"$r_{200}$",
        )
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("comoving radius [kpc]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_temperature_density_evolution(
    times,
    rho_comoving_code,
    temperature_proper_cgs_K,  # noqa: N803
    filename,
    bin_count=48,
    ymin=0.1,
    density_to_nH_cgs_cm3=1.0,  # noqa: N803
):
    """Plot cell temperature against physical hydrogen number density."""
    rho_values = np.asarray(rho_comoving_code, dtype=float).ravel() * float(density_to_nH_cgs_cm3)
    temp_values = np.asarray(temperature_proper_cgs_K, dtype=float).ravel()
    valid = (
        np.isfinite(rho_values)
        & np.isfinite(temp_values)
        & (rho_values > 0.0)
        & (temp_values > 0.0)
    )
    fig, axis = plt.subplots(figsize=(7.5, 6.0))
    if np.any(valid):
        log_rho = np.log10(rho_values[valid])
        log_temp = np.log10(temp_values[valid])
        bin_count = max(8, int(bin_count))
        rho_edges = np.linspace(log_rho.min(), log_rho.max(), bin_count + 1)
        temp_edges = np.linspace(log_temp.min(), log_temp.max(), bin_count + 1)
        counts, _, _ = np.histogram2d(log_rho, log_temp, bins=(rho_edges, temp_edges))
        count_max = float(counts.max())
        image = axis.pcolormesh(
            10.0**rho_edges,
            10.0**temp_edges,
            np.ma.masked_less_equal(counts.T, 0.0),
            norm=LogNorm(vmin=1.0, vmax=max(1.0, count_max)),
            cmap="magma",
            shading="auto",
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
    times,
    radius_comoving_code,
    rho_comoving_code,
    vel_supercomoving_code,
    virial_radius,
    scale_factors,
    filename,
    radial_bin_count=48,
):
    """Plot mass-weighted absolute physical radial velocity profiles."""
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int),
    )
    colors = plt.get_cmap("cividis")(np.linspace(0.05, 0.95, selected.size))
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    for color, index in zip(colors, selected, strict=False):
        proper_radius = radius_comoving_code * scale_factors[index]
        cell_edges = np.empty(proper_radius.size + 1, dtype=float)
        if proper_radius.size > 1:
            cell_edges[1:-1] = np.sqrt(proper_radius[:-1] * proper_radius[1:])
            cell_edges[0] = proper_radius[0] ** 2 / cell_edges[1]
            cell_edges[-1] = proper_radius[-1] ** 2 / cell_edges[-2]
        else:
            cell_edges[:] = (0.5 * proper_radius[0], 1.5 * proper_radius[0])
        mass_weight = np.asarray(rho_comoving_code[index], dtype=float) * np.maximum(
            np.diff(cell_edges**3),
            0.0,
        )
        binned_radius, binned_velocity = _log_radial_bin_profile(
            proper_radius,
            np.maximum(np.asarray(vel_supercomoving_code[index], dtype=float), 0.0),
            weights=mass_weight,
            bin_count=radial_bin_count,
        )
        axis.loglog(
            binned_radius,
            np.maximum(binned_velocity, 1.0e-12),
            color=color,
            lw=1.7,
            label=f"t = {times[index]:.2f} Gyr",
        )
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axis.axvline(
                virial_radius[index],
                color=color,
                ls="--",
                lw=0.9,
                alpha=0.65,
            )
    axis.set_xlabel("proper radius [kpc]")
    axis.set_ylabel(r"mass-weighted $|v_r|$ [km s$^{-1}$]")
    axis.set_title(
        "Mass-weighted absolute gas radial velocity from the z=100 LCDM IC\n"
        "solid: $|v_r|$; dashed: corresponding $r_{200}$",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="best", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_baryon_fraction_evolution(
    times,
    baryon_fraction,
    halo_mass_msun,
    scale_factors,
    filename,
    cosmic_baryon_fraction,
):
    """Plot normalized baryons and total mass inside resolved r200."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(baryon_fraction, dtype=float)
    scale_factors = np.asarray(scale_factors, dtype=float)
    halo_mass_msun = np.asarray(halo_mass_msun, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axis = axes[0]
    axis.plot(
        times,
        values,
        "o-",
        lw=2,
        label=r"$M_g(<r_{200})/[f_b M_{200}]$",
    )
    axis.axhline(
        1.0,
        color="black",
        ls="--",
        lw=1.0,
        label=rf"cosmic fraction ($f_b={cosmic_baryon_fraction:.3f}$)",
    )
    # ``times`` is already converted to physical Gyr by the caller.  Use an
    # explicit unit-bearing label here so this diagnostic cannot silently
    # regress to the dimensionless cosmology/code-time coordinate.
    axis.set_xlim(float(times[0]), float(times[-1]))
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel("normalized baryon mass fraction")
    axis.set_title(r"Baryon content inside resolved $r_{200}$")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    mass_axis = axes[1]
    mass_axis.plot(
        times,
        halo_mass_msun,
        "s-",
        color="tab:purple",
        lw=2,
        label=r"$M_{\rm vir}=M(<r_{200})$",
    )
    mass_axis.set_yscale("log")
    mass_axis.set_ylabel(r"total halo mass $M_{\rm vir}$ [$M_\odot$]")
    mass_axis.set_xlabel("cosmic time [Gyr]")
    mass_axis.grid(alpha=0.3, which="both")
    mass_axis.legend(frameon=False)
    finite = np.isfinite(times) & np.isfinite(scale_factors) & (scale_factors > 0.0)
    if np.count_nonzero(finite) >= 2:  # noqa: PLR2004
        # Place redshift ticks at the actual saved cosmic-time snapshots.
        # This avoids treating code time as Gyr and avoids an interpolated
        # redshift transform whose labels can be misleading between outputs.
        time_valid = times[finite]
        redshift_valid = 1.0 / scale_factors[finite] - 1.0
        selected = np.unique(
            np.linspace(0, time_valid.size - 1, min(6, time_valid.size), dtype=int),
        )
        top_axis = axis.twiny()
        top_axis.set_xlim(axis.get_xlim())
        top_axis.set_xticks(time_valid[selected])
        top_axis.set_xticklabels([f"{value:.0f}" for value in redshift_valid[selected]])
        top_axis.set_xlabel("redshift z (from saved scale factor)")
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_dark_matter_density_evolution(
    dm_profiles,
    gas_radius,
    filename,
    density_bin_count=None,
):
    """Plot enclosed DM mass and density contrast relative to the background."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    density_axis, mass_axis = axes
    selected = np.unique(
        np.linspace(0, len(dm_profiles) - 1, min(9, len(dm_profiles))).astype(int),
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, selected.size))
    gas_radius = np.asarray(gas_radius, dtype=float)
    valid_radius = np.isfinite(gas_radius) & (gas_radius > 0.0)
    gas_radius = gas_radius[valid_radius]
    if gas_radius.size < 1:
        plt.close(fig)
        return
    # Aggregate the shell profile into fewer logarithmic diagnostic bins.
    # The underlying shell data and enclosed-mass panel remain full resolution.
    bin_count = (
        gas_radius.size
        if density_bin_count is None
        else max(
            1,
            min(int(density_bin_count), gas_radius.size),
        )
    )
    bin_edges = np.geomspace(gas_radius[0], gas_radius[-1], bin_count + 1)
    bin_radii = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    for index, color in zip(selected, colors, strict=False):
        profile = dm_profiles[index]
        scale_factor = float(profile["scale_factor"])
        radius_proper_kpc = np.asarray(profile["dm_radius_proper_kpc"], dtype=float)
        rho_proper_code = np.asarray(profile["dm_rho_proper_code"], dtype=float)
        mass_shell_comoving_code = np.asarray(profile["dm_mass_comoving_code"], dtype=float)
        mass_core_comoving_code = float(profile.get("dm_central_core_mass", 0.0))
        core_radius_comoving_code = (
            float(profile.get("dm_central_core_radius_kpc", 0.0)) / scale_factor
        )
        comoving_radius = radius_proper_kpc / scale_factor
        valid = (
            np.isfinite(comoving_radius)
            & np.isfinite(rho_proper_code)
            & np.isfinite(mass_shell_comoving_code)
            & (comoving_radius > 0.0)
            & (mass_shell_comoving_code > 0.0)
        )
        radius_shell_comoving_code = comoving_radius[valid]
        shell_mass_comoving_code = mass_shell_comoving_code[valid]
        if radius_shell_comoving_code.size < 1:
            continue
        mass_in_bin, _ = np.histogram(
            radius_shell_comoving_code,
            bins=bin_edges,
            weights=shell_mass_comoving_code,
        )
        if mass_core_comoving_code > 0.0 and core_radius_comoving_code > 0.0:
            core_bin = int(np.searchsorted(bin_edges, core_radius_comoving_code, side="right") - 1)
            if 0 <= core_bin < mass_in_bin.size:
                mass_in_bin[core_bin] += mass_core_comoving_code
        bin_volume = 4.0 * np.pi / 3.0 * scale_factor**3 * np.diff(bin_edges**3)
        binned_density = mass_in_bin / np.maximum(bin_volume, 1.0e-30)
        valid_bins = binned_density > 0.0
        background_density = float(profile.get("dm_mean_density_code", np.nan))
        density_contrast = binned_density / max(background_density, 1.0e-300)
        density_axis.loglog(
            bin_radii[valid_bins],
            density_contrast[valid_bins],
            color=color,
            lw=1.6,
            label="t = {:.2f}".format(profile["time_cosmic_Gyr"]),
        )
        cumulative_mass_comoving_code = np.cumsum(shell_mass_comoving_code)
        if mass_core_comoving_code > 0.0:
            cumulative_mass_comoving_code = cumulative_mass_comoving_code + mass_core_comoving_code
        mass_axis.step(
            radius_shell_comoving_code,
            cumulative_mass_comoving_code,
            where="post",
            color=color,
            lw=1.6,
            label="t = {:.2f}".format(profile["time_cosmic_Gyr"]),
        )
    density_axis.axhline(1.0, color="black", lw=0.8, ls="--")
    density_axis.set_xlabel("comoving radius [kpc]")
    density_axis.set_ylabel(r"DM density / background density")
    density_axis.set_title("Live dark-matter density contrast")
    mass_axis.set_xlabel("comoving radius [kpc]")
    mass_axis.set_ylabel("enclosed dark-matter mass [code mass]")
    mass_axis.set_title("Live dark-matter enclosed mass")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
        axis.legend(title="cosmic time [Gyr]", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_baryon_normalized_density_comparison(
    gas_profiles,
    dm_profiles,
    baryon_fraction,
    virial_radius,
    filename,
):
    """Compare gas and DM profiles after removing their cosmic fractions."""
    selected = np.unique(
        np.linspace(0, len(gas_profiles) - 1, min(9, len(gas_profiles))).astype(int),
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, selected.size))
    fb = float(baryon_fraction)
    virial_radius = np.asarray(virial_radius, dtype=float)
    all_comoving = np.concatenate(
        [
            np.asarray(profile["dm_radius_proper_kpc"], dtype=float)
            / float(profile["scale_factor"])
            for profile in dm_profiles
        ],
    )
    bin_edges = np.geomspace(
        max(1.0e-8, np.nanmin(all_comoving) * 0.9),
        np.nanmax(all_comoving) * 1.1,
        49,
    )
    bin_radii = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    for color, index in zip(colors, selected, strict=False):
        gas = gas_profiles[index]
        dm = dm_profiles[index]
        scale_factor = float(gas["scale_factor"])
        gas_radius = np.asarray(gas["radius_proper_kpc"], dtype=float)
        gas_density_raw = np.asarray(gas["rho_proper_code"], dtype=float)
        dm_radius = np.asarray(dm["dm_radius_proper_kpc"], dtype=float)
        dm_mass_comoving_code = np.asarray(dm["dm_mass_comoving_code"], dtype=float)
        proper_edges = scale_factor * bin_edges
        gas_edges = np.empty(gas_radius.size + 1)
        gas_edges[1:-1] = np.sqrt(gas_radius[:-1] * gas_radius[1:])
        gas_edges[0] = gas_radius[0] ** 2 / gas_edges[1]
        gas_edges[-1] = gas_radius[-1] ** 2 / gas_edges[-2]
        gas_volume = 4.0 * np.pi / 3.0 * np.diff(gas_edges**3)
        gas_mass_comoving_code_bin, _ = np.histogram(
            gas_radius / scale_factor,
            bins=bin_edges,
            weights=gas_density_raw * gas_volume,
        )
        dm_mass_comoving_code_bin, _ = np.histogram(
            dm_radius / scale_factor,
            bins=bin_edges,
            weights=dm_mass_comoving_code,
        )
        bin_volume = 4.0 * np.pi / 3.0 * np.diff(proper_edges**3)
        gas_density_comoving_code = (
            gas_mass_comoving_code_bin / np.maximum(bin_volume, 1.0e-300) / fb
        )
        dm_density_comoving_code = (
            dm_mass_comoving_code_bin / np.maximum(bin_volume, 1.0e-300) / (1.0 - fb)
        )
        valid = (gas_density_comoving_code > 0.0) & (dm_density_comoving_code > 0.0)
        label = "t = {:.2f} Gyr".format(gas["time_cosmic_Gyr"])
        axes[0].loglog(
            bin_radii[valid],
            gas_density_comoving_code[valid],
            color=color,
            lw=1.5,
            label=label + " gas/$f_b$",
        )
        axes[0].loglog(
            bin_radii[valid],
            dm_density_comoving_code[valid],
            color=color,
            lw=1.0,
            ls="--",
            alpha=0.85,
            label=label + " DM/$1-f_b$",
        )
        axes[1].semilogx(
            bin_radii[valid],
            gas_density_comoving_code[valid] / dm_density_comoving_code[valid],
            color=color,
            lw=1.5,
        )
        if index < virial_radius.size and np.isfinite(virial_radius[index]):
            rvir_comoving = virial_radius[index] / scale_factor
            axes[0].axvline(
                rvir_comoving,
                color=color,
                ls=":",
                lw=1.5,
                label=label + r" $r_{200}$",
            )
            axes[1].axvline(
                rvir_comoving,
                color=color,
                ls=":",
                lw=1.2,
            )
    axes[0].set_ylabel(r"density / cosmic fraction")
    axes[0].set_title("Gas versus dark matter (densities normalized by cosmic fractions)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.25, which="both")
    axes[1].axhline(1.0, color="black", ls=":", lw=1.0)
    axes[1].set_ylabel(r"$(\rho_g/f_b)/(\rho_{DM}/(1-f_b))$")
    axes[1].set_xlabel("comoving radius [kpc]")
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
        result[row, : array.size] = array
    return result


def _energy_audit_state(sim):
    """Return conserved gas-energy diagnostics for the physical cells."""
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    rho_comoving_code = np.asarray(sim.fluid.rho_comoving_code[first:last], dtype=float)
    vel_supercomoving_code = np.asarray(sim.fluid.vel_supercomoving_code[first:last], dtype=float)
    volume_comoving_code = np.asarray(sim.mesh.volume_comoving_code[first:last], dtype=float)
    gas_mass_comoving_code = np.asarray(sim.fluid.Mass_code[first:last], dtype=float)
    total_energy = np.asarray(sim.fluid.Energy_code[first:last], dtype=float)
    kinetic_density = 0.5 * rho_comoving_code * vel_supercomoving_code**2
    kinetic_energy_code = float(np.sum(kinetic_density * volume_comoving_code))
    total_energy_value = float(np.sum(total_energy))
    return {
        "total_gas_mass_comoving_code": float(np.sum(gas_mass_comoving_code)),
        "total_gas_energy_code": total_energy_value,
        "kinetic_energy_code": kinetic_energy_code,
        "thermal_energy_code": total_energy_value - kinetic_energy_code,
        "dual_energy_pressure_fallback_count": float(
            getattr(sim.solver, "dual_energy_pressure_fallback_count", 0),
        ),
        "dual_energy_synchronization_count": float(
            getattr(sim.solver, "dual_energy_synchronization_count", 0),
        ),
        "dual_energy_floor_count": float(
            getattr(sim.solver, "dual_energy_floor_count", 0),
        ),
        "dual_energy_floor_injected_energy": float(
            getattr(sim.solver, "dual_energy_floor_injected_energy", 0.0),
        ),
        "dual_energy_entropy_limiter_count": float(
            getattr(sim.solver, "dual_energy_entropy_limiter_count", 0),
        ),
    }


def _energy_cell_state(sim):
    """Return per-cell gas energy components for physical cells."""
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    rho_comoving_code = np.asarray(sim.fluid.rho_comoving_code[first:last], dtype=float)
    vel_supercomoving_code = np.asarray(sim.fluid.vel_supercomoving_code[first:last], dtype=float)
    volume_comoving_code = np.asarray(sim.mesh.volume_comoving_code[first:last], dtype=float)
    total = np.asarray(sim.fluid.Energy_code[first:last], dtype=float)
    kinetic = 0.5 * rho_comoving_code * vel_supercomoving_code**2 * volume_comoving_code
    return {
        "mass_code": np.asarray(sim.fluid.Mass_code[first:last], dtype=float).copy(),
        "total_energy_code": total.copy(),
        "kinetic_energy_code": kinetic,
        "thermal_energy_code": total - kinetic,
        "gravitational_work": np.asarray(
            getattr(
                sim,
                "cumulative_gravity_work_by_cell",
                np.zeros(last - first),
            ),
            dtype=float,
        ).copy(),
        "hydro_energy_change": np.asarray(
            getattr(
                sim,
                "cumulative_hydro_energy_change_by_cell",
                np.zeros(last - first),
            ),
            dtype=float,
        ).copy(),
        "thermochemistry_energy_change": np.asarray(
            getattr(
                sim,
                "cumulative_thermochemistry_energy_change_by_cell",
                np.zeros(last - first),
            ),
            dtype=float,
        ).copy(),
        "compression_work": np.asarray(
            getattr(sim, "cumulative_compression_work_by_cell", np.zeros(last - first)),
            dtype=float,
        ).copy(),
        "shock_work": np.asarray(
            getattr(sim, "cumulative_shock_work_by_cell", np.zeros(last - first)),
            dtype=float,
        ).copy(),
        # Same-state dual-energy reconstruction diagnostics.  A missing
        # diagnostic means dual energy was disabled or the solver has not yet
        # reconstructed primitive variables; retain NaNs rather than mixing
        # values from a different timestep.
        "dual_energy_total_thermal": _solver_cell_array(
            sim,
            "dual_energy_total_thermal",
            last - first,
        ),
        "dual_energy_internal_density": _solver_cell_array(
            sim,
            "dual_energy_internal_density",
            last - first,
        ),
        "dual_energy_total_pressure": _solver_cell_array(
            sim,
            "dual_energy_total_pressure",
            last - first,
        ),
        "dual_energy_dual_pressure": _solver_cell_array(
            sim,
            "dual_energy_dual_pressure",
            last - first,
        ),
        "dual_energy_total_valid": _solver_cell_array(
            sim,
            "dual_energy_total_valid",
            last - first,
            dtype=float,
        ),
        "dual_energy_dual_valid": _solver_cell_array(
            sim,
            "dual_energy_dual_valid",
            last - first,
            dtype=float,
        ),
        "dual_energy_pressure_selection_code": _solver_cell_array(
            sim,
            "dual_energy_pressure_selection_code",
            last - first,
            dtype=float,
        ),
    }


def _solver_cell_array(sim, name, size, dtype=float):
    """Return a physical-cell solver diagnostic from the last reconstruction."""
    value = getattr(sim.solver, name, None)
    if value is None:
        return np.full(size, np.nan, dtype=dtype)
    array = np.asarray(value, dtype=dtype).ravel()
    first = int(sim.par.mesh.ghost_cells)
    if array.size < first + size:
        return np.full(size, np.nan, dtype=dtype)
    return array[first : first + size].copy()


def _instantaneous_source_diagnostics(sim, gas_profile):
    """Return same-state physical source and compression diagnostics."""
    state = rtc.source_state(sim.mesh, sim.fluid, sim.par)
    thermal_rate = np.asarray(
        rtc.thermal_rate(
            state,
            rrt.trace_photon_density(state, sim.par),
            sim.par,
        ),
        dtype=float,
    )
    rho_comoving_code = np.asarray(state["rho_cgs_g_cm3"], dtype=float)
    specific_energy = np.asarray(state["specific_energy_cgs_erg_g"], dtype=float)
    temperature_proper_cgs_K = np.asarray(state["temperature_cgs_K"], dtype=float)
    mu = np.asarray(
        state.get(
            "mu",
            sim.fluid.mu[
                int(sim.par.mesh.ghost_cells) : int(sim.par.mesh.ghost_cells)
                + int(sim.par.mesh.grid_cells)
            ],
        ),
        dtype=float,
    )
    radius_proper_cgs_cm = (
        np.asarray(gas_profile["radius_proper_kpc"], dtype=float) * 3.0856775814913673e21
    )
    velocity_cgs_cm_s = (
        np.asarray(
            gas_profile["radial_velocity_proper_km_s"],
            dtype=float,
        )
        * 1.0e5
    )
    divergence = (
        np.gradient(radius_proper_cgs_cm**2 * velocity_cgs_cm_s, radius_proper_cgs_cm)
        / np.maximum(radius_proper_cgs_cm, 1.0e-30) ** 2
    )
    rho_dot = -rho_comoving_code * divergence
    sound_speed = np.sqrt(
        float(sim.par.hydrodynamics.gamma)
        * 1.380649e-16
        * np.maximum(temperature_proper_cgs_K, 0.0)
        / (np.maximum(mu, 1.0e-30) * 1.67262192369e-24),
    )
    mach = np.divide(
        np.abs(velocity_cgs_cm_s),
        sound_speed,
        out=np.full_like(sound_speed, np.nan),
        where=sound_speed > 0.0,
    )
    # thermal_rate is heating minus cooling; q is defined positive for cooling.
    q = -thermal_rate
    # q is volumetric [erg cm^-3 s^-1].  The equivalent expression using a
    # specific cooling rate q/rho is gamma - rho*(q/rho)/(rho_dot*e).
    gamma_eff = np.full_like(rho_comoving_code, np.nan)
    valid = (rho_dot > 0.0) & (specific_energy > 0.0)
    gamma_eff[valid] = float(sim.par.hydrodynamics.gamma) - q[valid] / (
        rho_dot[valid] * specific_energy[valid]
    )
    return {
        "q_cgs_erg_cm3_s": q,
        "rho_dot_cgs_g_cm3_s": rho_dot,
        "specific_energy_cgs_erg_g": specific_energy,
        "local_mach": mach,
        "gamma_eff": gamma_eff,
    }


def _dark_matter_energy_state(dm):
    """Return shell-resolved collisionless energy components.

    Shell IDs are retained so that shell crossings do not turn into apparent
    energy exchanges between rows of the history.
    """
    ids = getattr(dm, "shell_id", None)
    if ids is None:
        ids = np.arange(len(dm.radius), dtype=int)
    ids = np.asarray(ids, dtype=int)
    order = np.argsort(ids)
    radius_comoving_code = np.asarray(dm.radius, dtype=float)[order]
    vel_supercomoving_code = np.asarray(dm.velocity, dtype=float)[order]
    mass_code = np.asarray(dm.mass, dtype=float)[order]
    angular = (
        0.5
        * np.asarray(dm.angular_momentum, dtype=float)[order] ** 2
        / (np.maximum(radius_comoving_code, np.finfo(float).tiny) + float(dm.softening)) ** 2
    )
    total_specific = np.asarray(dm.specific_energy(), dtype=float)[order]
    potential = total_specific - 0.5 * vel_supercomoving_code**2 - angular
    return {
        "id": ids[order],
        "radius_comoving_code": radius_comoving_code,
        "vel_supercomoving_code": vel_supercomoving_code,
        "mass_code": mass_code,
        "kinetic_energy_code": mass_code * 0.5 * vel_supercomoving_code**2,
        "potential_energy_code": mass_code * potential,
        "total_energy_code": mass_code * total_specific,
    }


def _pad_energy_history(history, key, fill=np.nan):
    """Pack variable-width per-snapshot energy arrays."""
    width = max((np.asarray(item[key]).size for item in history), default=0)
    result = np.full((len(history), width), fill, dtype=float)
    for row, item in enumerate(history):
        values = np.asarray(item[key], dtype=float).ravel()
        result[row, : values.size] = values
    return result
