"""Plot the defined gas entropy proxy, ``T / rho**(gamma - 1)``."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs_correlation_gas"
PREFIX = "CosmologicalGasCorrelationZ100"


def _add_redshift_top_axis(axis, times, scale_factors):
    """Add redshift ticks above a cosmic-time x-axis."""
    times = np.asarray(times, dtype=float)
    scale_factors = np.asarray(scale_factors, dtype=float)
    valid = np.isfinite(times) & np.isfinite(scale_factors) & (scale_factors > 0.0)
    if not np.any(valid):
        return
    top_axis = axis.twiny()
    top_axis.set_xlim(axis.get_xlim())
    indices = np.flatnonzero(valid)
    selected = indices[np.unique(np.linspace(0, indices.size - 1,
                                             min(7, indices.size)).astype(int))]
    top_axis.set_xticks(times[selected])
    top_axis.set_xticklabels(
        ["%.0f" % (1.0 / scale_factors[index] - 1.0) for index in selected]
    )
    top_axis.set_xlabel("redshift")


def main(output=OUTPUT, prefix=PREFIX, gamma=5.0 / 3.0,
         exclude_outer_cells=2):
    output = Path(output)
    data = np.load(output / (prefix + ".npz"))
    times = np.asarray(data["time_Gyr"], dtype=float)
    radius = np.asarray(data["radius_comoving_kpc"], dtype=float)
    density = np.asarray(data["density_proper_code"], dtype=float)
    temperature = np.asarray(data["temperature_physical_cgs_K"], dtype=float)
    scale = np.asarray(data["scale_factor"], dtype=float)
    rvir = np.asarray(data["rvir_proper_kpc"], dtype=float)
    rshock = np.asarray(data["rshock_kpc"], dtype=float)
    entropy = temperature / np.maximum(density, 1.0e-300) ** (float(gamma) - 1.0)
    cell_count = max(1, radius.size - max(0, int(exclude_outer_cells)))
    radius = radius[:cell_count]
    entropy = entropy[:, :cell_count]

    selected = np.unique(np.linspace(0, len(times) - 1, min(9, len(times))).astype(int))
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected):
        valid = (
            np.isfinite(radius) & np.isfinite(entropy[index])
            & (radius > 0.0) & (entropy[index] > 0.0)
        )
        axes[0].loglog(
            radius[valid], entropy[index, valid], color=color, lw=1.7,
            label="t = %.2f Gyr" % times[index],
        )
        if np.isfinite(rvir[index]) and rvir[index] > 0.0:
            axes[0].axvline(
                rvir[index] / scale[index], color=color, ls="--",
                lw=0.9, alpha=0.65,
            )
        if np.isfinite(rshock[index]) and rshock[index] > 0.0:
            axes[0].axvline(
                rshock[index] / scale[index], color=color, ls="-.",
                lw=0.9, alpha=0.8,
            )

    axes[0].set_xlabel("comoving radius [kpc]")
    axes[0].set_ylabel(r"$T/\rho^{\gamma-1}$ [K / code-density$^{\gamma-1}$]")
    axes[0].set_title(
        r"Gas entropy evolution: $S = T/\rho^{\gamma-1}$"
        "\nsolid S; dashed $r_{200}$; dash-dot $r_{\\rm shock}$"
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)

    finite = np.isfinite(rvir) & (rvir > 0.0)
    finite_shock = np.isfinite(rshock) & (rshock > 0.0)
    if np.any(finite):
        axes[1].plot(
            times[finite], rvir[finite] / scale[finite], "k.-",
            label=r"$r_{200}$ (comoving)",
        )
    if np.any(finite_shock):
        axes[1].plot(
            times[finite_shock], rshock[finite_shock] / scale[finite_shock],
            "r.-", label=r"$r_{\rm shock}$ (comoving)",
        )
    if np.any(finite) or np.any(finite_shock):
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(
            0.5, 0.5, "no resolved $r_{200}$ yet",
            transform=axes[1].transAxes, ha="center", va="center",
        )
    axes[1].set_xlabel("cosmic time [Gyr]")
    _add_redshift_top_axis(axes[1], times, scale)
    axes[1].set_ylabel("comoving radius [kpc]")
    axes[1].grid(alpha=0.25)
    if times.size > 1:
        axes[1].set_xlim(times[0], times[-1])
    fig.tight_layout()
    figure = output / (prefix + "_Entropy.jpg")
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    print("entropy figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--exclude-outer-cells", type=int, default=2)
    args = parser.parse_args()
    main(args.output_dir, args.prefix, args.gamma, args.exclude_outer_cells)
