"""Plot the H/He snapshot against the supplied reference profiles."""

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "Output_000.hdf5"
FIGURE = HERE / "HHe_multifrequency_snapshot_vs_reference.jpg"
HYDROGEN_MASS_FRACTION = 0.75
HELIUM_MASS_FRACTION = 0.25
HELIUM_TO_HYDROGEN_NUMBER_RATIO = (
    HELIUM_MASS_FRACTION / (4.0 * HYDROGEN_MASS_FRACTION)
)


def main():
    with h5py.File(SNAPSHOT, "r") as handle:
        data = handle["Data"]
        temperature = np.asarray(data["Temperature"])
        xhi = np.asarray(data["NeutralFraction"])
        xhei = np.asarray(data["HeINeutralFraction"])
        xheii = np.asarray(data["HeIIFraction"])
        xheiii = np.asarray(data["HeIIIFraction"])

    ncell = temperature.size - 4
    radius = (np.arange(ncell, dtype=float) + 0.5) * 10.0 / ncell / 5.4
    snapshot = {
        "H I": xhi[2:-2],
        "H II": 1.0 - xhi[2:-2],
        "He I": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xhei[2:-2],
        "He II": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xheii[2:-2],
        "He III": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xheiii[2:-2],
    }
    references = {
        "H I": "xHITT1D_Stromgren100Myr_HHe.txt",
        "H II": "xHIITT1D_Stromgren100Myr_HHe.txt",
        "He I": "xHeITT1D_Stromgren100Myr_HHe.txt",
        "He II": "xHeIITT1D_Stromgren100Myr_HHe.txt",
        "He III": "xHeIIITT1D_Stromgren100Myr_HHe.txt",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), sharex=True)
    for axis, (species, reference_name) in zip(axes.flat, references.items()):
        axis.plot(radius, np.clip(snapshot[species], 1.0e-12, 1.0),
                  color="tab:blue", label="snapshot: 100 Myr")
        reference = np.loadtxt(HERE / reference_name, delimiter=",")
        axis.scatter(reference[:, 0], 10.0 ** reference[:, 1],
                     color="tab:orange", s=18, label="reference: 100 Myr")
        axis.set_yscale("log")
        axis.set_ylim(1.0e-6, 1.1)
        axis.set_title(species)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    temperature_axis = axes[1, 2]
    temperature_axis.clear()
    temperature_axis.plot(radius, np.clip(temperature[2:-2], 1.0, None),
                           color="tab:red", label="snapshot: 100 Myr")
    temperature_reference = np.loadtxt(
        HERE / "TTT1D_Stromgren100Myr_HHe.txt", delimiter=","
    )
    temperature_axis.scatter(
        temperature_reference[:, 0],
        10.0 ** temperature_reference[:, 1],
        color="tab:orange",
        s=18,
        label="reference: 100 Myr",
    )
    temperature_axis.axhline(1.0e5, color="tab:purple", linestyle="--",
                             label=r"$T_{\rm rad}=10^5$ K")
    temperature_axis.set_yscale("log")
    temperature_axis.set_ylim(1.0e1, 1.0e8)
    temperature_axis.set_title("Temperature")
    temperature_axis.set_ylabel("T [K]")
    temperature_axis.grid(True, which="both", alpha=0.25)
    temperature_axis.legend(frameon=False, fontsize=8)

    axes[1, 0].set_xlabel(r"$r/r_s$, $r_s=5.4$ kpc")
    axes[1, 1].set_xlabel(r"$r/r_s$, $r_s=5.4$ kpc")
    axes[1, 2].set_xlabel(r"$r/r_s$, $r_s=5.4$ kpc")
    axes[0, 0].set_ylabel("mass fraction")
    axes[1, 0].set_ylabel("mass fraction")
    fig.suptitle("H/He multifrequency snapshot vs reference at 100 Myr")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
