"""Plot energy accounting inside the evolving virial-radius gas halo."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs_correlation_gas_compton_atomic"
PREFIX = "CosmologicalGasCorrelationZ100_ComptonAtomic"


def _halo_change(data, profiles, key, reference_index):
    radius = np.asarray(profiles["radius_comoving_kpc"], dtype=float)
    scale = np.asarray(profiles["scale_factor"], dtype=float)
    rvir = np.asarray(profiles["rvir_proper_kpc"], dtype=float)
    values = np.asarray(data[key], dtype=float)
    reference_mask = radius * scale[reference_index] <= rvir[reference_index]
    reference = np.nansum(values[reference_index, reference_mask])
    result = np.full(values.shape[0], np.nan)
    for index in range(reference_index, values.shape[0]):
        if not np.isfinite(rvir[index]):
            continue
        mask = radius * scale[index] <= rvir[index]
        result[index] = np.nansum(values[index, mask]) - reference
    return result


def main(output=OUTPUT, prefix=PREFIX):
    output = Path(output)
    data = np.load(output / (prefix + "_EnergyByCellAndShell.npz"))
    profiles = np.load(output / (prefix + ".npz"))
    time = np.asarray(data["gas_time_Gyr"], dtype=float)
    rvir = np.asarray(profiles["rvir_proper_kpc"], dtype=float)
    valid = np.flatnonzero(np.isfinite(rvir))
    if valid.size == 0:
        raise RuntimeError("no resolved virial-radius snapshots found")
    reference_index = int(valid[0])

    changes = {
        name: _halo_change(data, profiles, key, reference_index)
        for name, key in {
            "thermal": "gas_thermal_energy",
            "kinetic": "gas_kinetic_energy",
            "gravity": "gas_gravitational_work",
            "compression": "gas_compression_work",
            "shock": "gas_shock_work",
            "thermochemistry": "gas_thermochemistry_energy_change",
        }.items()
    }
    changes["gravity_minus_kinetic"] = changes["gravity"] - changes["kinetic"]
    changes["thermal_accounting_sum"] = (
        changes["compression"] + changes["shock"] + changes["thermochemistry"]
    )

    figure = output / (prefix + "_HaloEnergyAccounting_TimeEvolution.jpg")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(time, changes["thermal"], "o-", label="thermal energy change")
    axes[0].plot(time, changes["kinetic"], "o-", label="kinetic energy change")
    axes[0].plot(time, changes["gravity"], "o-", label="gravitational work")
    axes[0].plot(
        time, changes["gravity_minus_kinetic"], "o--",
        label="gravity work − kinetic change",
    )
    axes[0].set_ylabel("energy change [code units]")
    axes[0].set_title("Energy changes inside the evolving virial-radius halo")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(time, changes["thermal"], "o-", lw=2, label="thermal change")
    axes[1].plot(time, changes["compression"], "o-", label="compression work")
    axes[1].plot(time, changes["shock"], "o-", label="shock work")
    axes[1].plot(time, changes["thermochemistry"], "o-", label="thermochemistry")
    axes[1].plot(
        time, changes["thermal_accounting_sum"], "o--", lw=2,
        label="compression + shock + thermochemistry",
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("energy change [code units]")
    axes[1].set_title("Thermal-energy accounting")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    print("halo energy accounting figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    main(args.output_dir, args.prefix)
