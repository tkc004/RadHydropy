"""Plot energy accounting inside an evolving multiple of the virial radius."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs_correlation_gas_compton_atomic"
PREFIX = "CosmologicalGasCorrelationZ100_ComptonAtomic"


def _aperture_sum(data, profiles, key, radius_factor=2.0):
    radius = np.asarray(profiles["radius_comoving_kpc"], dtype=float)
    scale = np.asarray(profiles["scale_factor"], dtype=float)
    rvir = np.asarray(profiles["rvir_proper_kpc"], dtype=float)
    values = np.asarray(data[key], dtype=float)
    result = np.full(values.shape[0], np.nan)
    for index in range(values.shape[0]):
        if not np.isfinite(rvir[index]):
            continue
        mask = radius * scale[index] <= radius_factor * rvir[index]
        result[index] = np.nansum(values[index, mask])
    return result


def main(output=OUTPUT, prefix=PREFIX, radius_factor=2.0):
    output = Path(output)
    data = np.load(output / (prefix + "_EnergyByCellAndShell.npz"))
    profiles = np.load(output / (prefix + ".npz"))
    time = np.asarray(data["gas_time_Gyr"], dtype=float)
    rvir = np.asarray(profiles["rvir_proper_kpc"], dtype=float)
    if not np.any(np.isfinite(rvir)):
        raise RuntimeError("no resolved virial-radius snapshots found")

    changes = {
        name: _aperture_sum(data, profiles, key, radius_factor)
        for name, key in {
            "delta_total": "gas_delta_total_energy",
            "delta_thermal": "gas_delta_thermal_energy",
            "delta_kinetic": "gas_delta_kinetic_energy",
            "hydro": "gas_hydro_energy_change",
            "gravity": "gas_gravitational_work",
            "thermochemistry": "gas_thermochemistry_energy_change",
            "residual": "gas_energy_balance_residual",
        }.items()
    }
    changes["accounting_sum"] = (
        changes["hydro"] + changes["gravity"] + changes["thermochemistry"]
    )

    figure = output / (prefix + "_2RvirEnergyBalance_TimeEvolution.jpg")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    # Circles denote remaining-energy components; process terms use distinct
    # non-circular markers so the energy partition is visually unambiguous.
    axes[0].plot(time, changes["delta_total"], "o-", lw=2, label=r"$\Delta E_i$")
    axes[0].plot(
        time, changes["delta_thermal"], "o-", lw=2,
        label=r"$\Delta E_{i,\rm thermal}$",
    )
    axes[0].plot(
        time, changes["delta_kinetic"], "o-", lw=2,
        label=r"$\Delta E_{i,\rm kinetic}$",
    )
    axes[0].plot(
        time, changes["hydro"], "^-",
        label=r"hydrodynamic flux energy change $\Delta E_{i,\rm flux}$",
    )
    axes[0].plot(time, changes["gravity"], "s-", label="gravitational work")
    axes[0].plot(time, changes["thermochemistry"], "D-", label="thermochemistry")
    axes[0].plot(time, changes["accounting_sum"], "P--", lw=2, label="accounting sum")
    axes[0].set_ylabel("energy change [code units]")
    axes[0].set_title(r"Energy balance inside $2r_{\rm vir}(t)$")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(time, changes["residual"], "x-", lw=2, label="balance residual")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("energy change [code units]")
    axes[1].set_title("Cell-wise balance residual inside aperture")
    axes[1].legend(frameon=False, fontsize=9)
    if time.size >= 2:
        for axis in axes:
            axis.set_xlim(float(time[0]), float(time[-1]))
    scale_factor = np.asarray(profiles["scale_factor"], dtype=float)
    finite = np.isfinite(time) & np.isfinite(scale_factor) & (scale_factor > 0.0)
    time_valid = time[finite]
    redshift_valid = 1.0 / scale_factor[finite] - 1.0
    if time_valid.size >= 2:
        # Use exact snapshot locations.  A secondary-axis interpolation
        # extrapolates the final point when its locator requests z=0, even
        # though this run stops at z~5; explicit ticks prevent that error.
        selected = np.unique(np.linspace(
            0, time_valid.size - 1, min(7, time_valid.size), dtype=int
        ))
        top_axis = axes[0].twiny()
        top_axis.set_xlim(axes[0].get_xlim())
        top_axis.set_xticks(time_valid[selected])
        top_axis.set_xticklabels(["%.0f" % value for value in redshift_valid[selected]])
        top_axis.set_xlabel("redshift z (from saved scale factor)")
    resolved = np.isfinite(rvir)
    if np.any(resolved) and not np.all(resolved):
        first_resolved = time[np.flatnonzero(resolved)[0]]
        for axis in axes:
            axis.axvspan(
                float(time[0]), float(first_resolved),
                color="0.85", alpha=0.35, lw=0,
            )
        axes[0].text(
            0.02, 0.96, "no resolved halo: $r_{200}$ undefined",
            transform=axes[0].transAxes, va="top", fontsize=9, color="0.25",
        )
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    print("2-rvir energy balance figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--radius-factor", type=float, default=2.0)
    args = parser.parse_args()
    main(args.output_dir, args.prefix, args.radius_factor)
