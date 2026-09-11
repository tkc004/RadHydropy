"""Plot the H/He snapshot against the supplied reference profiles."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXAMPLE_ROOT.parent
SOURCE_EXAMPLE = EXAMPLE_ROOT / "MultiFrequencyRadiativeTransferSph1D"
for path in (PROJECT_ROOT, EXAMPLE_ROOT, SOURCE_EXAMPLE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import example_utils as eu
import unyt

from multifrequency_tools import active_radarray, load_snapshot


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "Output_000.hdf5"
FIGURE = HERE / "HHe_multifrequency_snapshot_vs_reference.jpg"
CONFIG = HERE / "multifrequency_radiative_transfer_sph1d_hhe_100myr.yaml"
HYDROGEN_MASS_FRACTION = 0.75
HELIUM_MASS_FRACTION = 0.25
HELIUM_TO_HYDROGEN_NUMBER_RATIO = (
    HELIUM_MASS_FRACTION / (4.0 * HYDROGEN_MASS_FRACTION)
)


def main(snapshot_filename=SNAPSHOT, figure_filename=FIGURE,
         config_filename=CONFIG):
    snapshot_filename = Path(snapshot_filename)
    figure_filename = Path(figure_filename)
    config = eu.load_nested_example_config(config_filename)
    snapshot = load_snapshot(snapshot_filename, config)
    active_cells = int(snapshot.par.mesh.grid_cells)
    ghost_cells = int(snapshot.par.mesh.ghost_cells)
    temperature_proper_radarray = active_radarray(
        snapshot.fluid.temp_radarray,
        active_cells,
        ghost_cells,
    )
    temperature_cgs_K = temperature_proper_radarray.to("K").value
    xhi = np.asarray(
        active_radarray(snapshot.fluid.xHI, active_cells, ghost_cells),
        dtype=float,
    )
    xhei = np.asarray(
        active_radarray(snapshot.fluid.xHeI, active_cells, ghost_cells),
        dtype=float,
    )
    xheii = np.asarray(
        active_radarray(snapshot.fluid.xHeII, active_cells, ghost_cells),
        dtype=float,
    )
    xheiii = np.asarray(
        active_radarray(snapshot.fluid.xHeIII, active_cells, ghost_cells),
        dtype=float,
    )
    boundary_proper_radarray = active_radarray(
        snapshot.mesh.boundary_radarray,
        active_cells,
        ghost_cells,
        boundary=True,
    )
    radius_proper_radarray = 0.5 * (
        boundary_proper_radarray[:-1] + boundary_proper_radarray[1:]
    )
    radius_proper_kpc = radius_proper_radarray.to("kpc").value / 5.4
    snapshot = {
        "H I": xhi,
        "H II": 1.0 - xhi,
        "He I": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xhei,
        "He II": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xheii,
        "He III": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xheiii,
    }
    references = {
        "H I": "xHITT1D_Stromgren100Myr_HHe.txt",
        "H II": "xHIITT1D_Stromgren100Myr_HHe.txt",
        "He I": "xHeITT1D_Stromgren100Myr_HHe.txt",
        "He II": "xHeIITT1D_Stromgren100Myr_HHe.txt",
        "He III": "xHeIIITT1D_Stromgren100Myr_HHe.txt",
    }

    snapshot_label = (
        "C²-Ray snapshot: 100 Myr"
        if "C2Ray" in snapshot_filename.stem
        else "snapshot: 100 Myr"
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), sharex=True)
    for axis, (species, reference_name) in zip(axes.flat, references.items()):
        axis.plot(radius_proper_kpc, np.clip(snapshot[species], 1.0e-12, 1.0),
                  color="tab:blue", label=snapshot_label)
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
    temperature_axis.plot(
        radius_proper_kpc,
        np.clip(
            temperature_cgs_K,
            1.0,
            None,
        ),
                           color="tab:red", label=snapshot_label)
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
    fig.suptitle(f"H/He multifrequency {snapshot_label.lower()} vs reference")
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(figure_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=SNAPSHOT, type=Path)
    parser.add_argument("--figure", default=FIGURE, type=Path)
    args = parser.parse_args()
    main(args.snapshot, args.figure)
