"""Compare saved gas and dark-matter profiles for crossing-batch runs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs_correlation_gas_crossing_comparison"
RUNS = [
    (0.0, HERE / "outputs_correlation_gas_crossing_f0p00",
     "CosmologicalGasCorrelationZ100_crossing_f0p00"),
    (0.01, HERE / "outputs_correlation_gas_batch_001_0p50",
     "CosmologicalGasCorrelationZ100_batch_001_0p50"),
    (0.5, HERE / "outputs_correlation_gas_batched_crossings",
     "CosmologicalGasCorrelationZ100_batched_crossings"),
]


def _label(fraction):
    return "f = %.2g" % fraction


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    gas_fig, gas_ax = plt.subplots(figsize=(8, 6))
    dm_fig, dm_ax = plt.subplots(figsize=(8, 6))
    rows = []
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(RUNS)))

    for color, (fraction, run_dir, prefix) in zip(colors, RUNS):
        gas = np.load(run_dir / (prefix + ".npz"))
        dm = np.load(run_dir / (prefix + "_DarkMatterDensities.npz"))
        gas_radius = gas["radius_comoving_kpc"] / gas["scale_factor"][-1]
        gas_density = gas["density_proper_code"][-1]
        dm_radius = dm["radius_kpc"][-1]
        dm_density = dm["density_code"][-1]
        gas_ax.loglog(gas_radius, gas_density, color=color, label=_label(fraction))
        dm_ax.loglog(dm_radius, np.maximum(dm_density, 1.0e-300), color=color,
                     label=_label(fraction))
        rows.append((fraction, gas["time_Gyr"][-1], gas["rvir_kpc"][-1],
                     dm["radius_kpc"][-1].max(), dm["mass"][-1].sum()))

    gas_ax.set_xlabel("proper radius [kpc]")
    gas_ax.set_ylabel("gas density [code units]")
    gas_ax.set_title("Final gas profiles vs. crossing batch fraction")
    gas_ax.grid(alpha=0.25, which="both")
    gas_ax.legend()
    gas_fig.tight_layout()
    gas_fig.savefig(OUTPUTS / "GasProfilesByCrossingBatchFraction.jpg", dpi=220)
    plt.close(gas_fig)

    dm_ax.set_xlabel("proper radius [kpc]")
    dm_ax.set_ylabel("dark-matter density [code units]")
    dm_ax.set_title("Final dark-matter profiles vs. crossing batch fraction")
    dm_ax.grid(alpha=0.25, which="both")
    dm_ax.legend()
    dm_fig.tight_layout()
    dm_fig.savefig(OUTPUTS / "DarkMatterProfilesByCrossingBatchFraction.jpg", dpi=220)
    plt.close(dm_fig)

    np.savetxt(
        OUTPUTS / "profile_summary.txt", np.asarray(rows),
        header="fraction final_time_Gyr rvir_kpc max_dm_radius_kpc total_dm_mass",
    )


if __name__ == "__main__":
    main()
