"""Plot saved shock-candidate indicators versus time and comoving radius."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs_correlation_gas"
PREFIX = "CosmologicalGasCorrelationZ100"


def _spherical_divergence(radius, velocity):
    flux = radius**2 * velocity
    divergence = np.empty_like(flux)
    for index, (radius_row, flux_row) in enumerate(zip(radius, flux)):
        divergence[index] = np.gradient(flux_row, radius_row, edge_order=1)
    return divergence / np.maximum(radius, 1.0e-30) ** 2


def _edges(values):
    centers = np.unique(values[np.isfinite(values)])
    if centers.size == 1:
        width = max(abs(centers[0]) * 0.01, 1.0)
        return np.array([centers[0] - width, centers[0] + width])
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _plot_indicator(axis, time, radius, values, title, label, signed=True):
    finite = np.isfinite(time) & np.isfinite(radius) & np.isfinite(values)
    if not np.any(finite):
        return
    time_edges = _edges(time[finite])
    radius_edges = _edges(radius[finite])
    count, _, _ = np.histogram2d(
        time[finite], radius[finite], bins=(time_edges, radius_edges)
    )
    if signed:
        scale = max(float(np.nanmax(np.abs(values[finite]), initial=0.0)), 1.0e-30)
        norm = SymLogNorm(linthresh=scale * 1.0e-4, vmin=-scale, vmax=scale)
        cmap = "coolwarm"
    else:
        positive = finite & (values > 0.0)
        if not np.any(positive):
            return
        norm = LogNorm(vmin=max(float(np.nanmin(values[positive])), 1.0e-12),
                       vmax=float(np.nanmax(values[positive])))
        cmap = "magma"
        finite &= values > 0.0
    weighted, _, _ = np.histogram2d(
        time[finite], radius[finite], bins=(time_edges, radius_edges),
        weights=values[finite],
    )
    mean = np.divide(weighted, count, out=np.full_like(weighted, np.nan), where=count > 0)
    image = axis.pcolormesh(
        time_edges, radius_edges, np.ma.masked_invalid(mean.T),
        cmap=cmap, norm=norm, shading="flat",
    )
    axis.set_title(title)
    axis.set_ylabel("comoving radius [kpc]")
    axis.grid(alpha=0.2)
    axis.figure.colorbar(image, ax=axis, label="mean " + label)


def main(output=OUTPUT, prefix=PREFIX, gamma=5.0 / 3.0,
         mu=0.59, exclude_outer_cells=2):
    output = Path(output)
    data = np.load(output / (prefix + ".npz"))
    time = np.asarray(data["time_Gyr"], dtype=float)
    scale = np.asarray(data["scale_factor"], dtype=float)
    comoving_radius = np.asarray(data["radius_comoving_kpc"], dtype=float)
    proper_radius = comoving_radius[None, :] * scale[:, None]
    density = np.asarray(data["density_proper_code"], dtype=float)
    temperature = np.asarray(data["temperature_physical_cgs_K"], dtype=float)
    velocity = np.asarray(data["radial_velocity_physical_km_s"], dtype=float)
    count = max(3, comoving_radius.size - max(0, int(exclude_outer_cells)))
    comoving_radius, proper_radius, density, temperature, velocity = (
        array[..., :count] for array in
        (comoving_radius, proper_radius, density, temperature, velocity)
    )

    divergence = _spherical_divergence(proper_radius, velocity)
    entropy = temperature / np.maximum(density, 1.0e-300) ** (float(gamma) - 1.0)
    midpoint_radius = 0.5 * (comoving_radius[1:] + comoving_radius[:-1])
    density_jump = np.log10(
        np.maximum(density[:, :-1], 1.0e-300)
        / np.maximum(density[:, 1:], 1.0e-300)
    )
    entropy_jump = np.log10(
        np.maximum(entropy[:, :-1], 1.0e-300)
        / np.maximum(entropy[:, 1:], 1.0e-300)
    )
    sound_speed = np.sqrt(
        float(gamma) * 1.380649e-16 * np.maximum(temperature, 0.0)
        / (float(mu) * 1.67262192369e-24)
    ) / 1.0e5
    pair_sound_speed = 0.5 * (sound_speed[:, 1:] + sound_speed[:, :-1])
    valid_temperature_pair = (temperature[:, 1:] > 1.0) & (temperature[:, :-1] > 1.0)
    velocity_jump_mach = np.divide(
        np.abs(velocity[:, 1:] - velocity[:, :-1]),
        pair_sound_speed,
        out=np.full_like(pair_sound_speed, np.nan),
        where=(pair_sound_speed > 0.0) & valid_temperature_pair,
    )
    time_cells = np.broadcast_to(time[:, None], proper_radius.shape)
    time_pairs = np.broadcast_to(time[:, None], (time.size, midpoint_radius.size))
    plot_radius = np.broadcast_to(comoving_radius[None, :], proper_radius.shape)
    plot_midpoint_radius = np.broadcast_to(
        midpoint_radius[None, :], (time.size, midpoint_radius.size)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    _plot_indicator(
        axes[0, 0], time_cells, plot_radius, divergence,
        r"Velocity divergence $\nabla\cdot v$", "divergence [km s$^{-1}$ kpc$^{-1}$]",
    )
    _plot_indicator(
        axes[0, 1], time_pairs, plot_midpoint_radius, density_jump,
        r"Density jump proxy $\log_{10}(\rho_{\rm inner}/\rho_{\rm outer})$",
        "log density jump", signed=True,
    )
    _plot_indicator(
        axes[1, 0], time_pairs, plot_midpoint_radius, velocity_jump_mach,
        r"Velocity-jump Mach proxy $|\Delta v|/c_s$", "Mach proxy", signed=False,
    )
    _plot_indicator(
        axes[1, 1], time_pairs, plot_midpoint_radius, entropy_jump,
        r"Entropy jump $\log_{10}(S_{\rm inner}/S_{\rm outer})$",
        "log entropy jump", signed=True,
    )
    for axis in axes[-1]:
        axis.set_xlabel("cosmic time [Gyr]")
    fig.suptitle("Shock-candidate indicators (outer boundary cells excluded)")
    fig.tight_layout()
    figure = output / (prefix + "_ShockCandidates_TimeRadius.jpg")
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    print("shock-candidate figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--mu", type=float, default=0.59)
    parser.add_argument("--exclude-outer-cells", type=int, default=2)
    args = parser.parse_args()
    main(args.output_dir, args.prefix, args.gamma, args.mu, args.exclude_outer_cells)
