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
for path in (PROJECT_ROOT, EXAMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import example_utils as eu
import radhydropy.io as rio
from radhydropy.rsim import Rsim
import unyt


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
    snapshot = Rsim(config["par"])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(snapshot_filename))
    first = int(snapshot.par.mesh.ghost_cells)
    last = first + int(snapshot.par.mesh.grid_cells)
    code_units = snapshot.par.units.CodeUnits
    temperature_proper_code = np.asarray(snapshot.fluid.temp_proper_code)
    xhi = np.asarray(snapshot.fluid.xHI)
    xhei = np.asarray(snapshot.fluid.xHeI)
    xheii = np.asarray(snapshot.fluid.xHeII)
    xheiii = np.asarray(snapshot.fluid.xHeIII)
    boundary_proper_code = np.asarray(snapshot.mesh.boundary_proper_code)

    interior = slice(first, last)
    radius_proper_code = 0.5 * (
        boundary_proper_code[:-1] + boundary_proper_code[1:]
    )
    radius_proper_code = radius_proper_code[interior]
    radius_proper_kpc = (
        radius_proper_code * float(code_units.length_unit.to_value(unyt.kpc)) / 5.4
    )
    snapshot = {
        "H I": xhi[interior],
        "H II": 1.0 - xhi[interior],
        "He I": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xhei[interior],
        "He II": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xheii[interior],
        "He III": HELIUM_TO_HYDROGEN_NUMBER_RATIO * xheiii[interior],
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
            temperature_proper_code[interior]
            * float(code_units.temperature_unit.to_value(unyt.K)),
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
