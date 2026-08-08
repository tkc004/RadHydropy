#!/usr/bin/env python3
"""Read and plot a CHIANTI cooling table for a selected electron density."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TABLE = (
    Path(__file__).resolve().parents[2]
    / "CHIANTI_11.0.2_database"
    / "cooling_tables"
    / "chianti_cooling_table.h5"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CHIANTI cooling rate versus log10 temperature."
    )
    parser.add_argument(
        "table",
        nargs="?",
        type=Path,
        default=DEFAULT_TABLE,
        help=f"Cooling-table HDF5 file. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--electron-density",
        type=float,
        default=1.0,
        help="Electron density in cm^-3. The nearest table value is used.",
    )
    parser.add_argument(
        "--ratio-densities",
        nargs=2,
        type=float,
        metavar=("NE_LOW", "NE_HIGH"),
        help=(
            "Plot Lambda(NE_HIGH) / Lambda(NE_LOW) instead of the cooling "
            "rate. Values are in cm^-3."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <table-stem>_vs_logT.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    table_path = args.table.expanduser().resolve()
    output_path = args.output or table_path.with_name(
        f"{table_path.stem}_vs_logT.png"
    )

    with h5py.File(table_path, "r") as f:
        temperature = f["temperature_K"][:]
        electron_density = f["electron_density_cm-3"][:]
        metallicity = f["metallicity_Zsun"][:]
        cooling = f["cooling_erg_cm3_s"][:]

    density_index = int(np.argmin(np.abs(electron_density - args.electron_density)))
    selected_density = electron_density[density_index]
    log_temperature = np.log10(temperature)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(metallicity)))

    if args.ratio_densities is not None:
        low_requested, high_requested = args.ratio_densities
        low_index = int(np.argmin(np.abs(electron_density - low_requested)))
        high_index = int(np.argmin(np.abs(electron_density - high_requested)))
        low_density = electron_density[low_index]
        high_density = electron_density[high_index]
        output_path = args.output or table_path.with_name(
            f"{table_path.stem}_ratio_ne_{low_density:g}_to_{high_density:g}.png"
        )

        ratio = cooling[:, :, high_index] / cooling[:, :, low_index]
        for color, z, curve in zip(colors, metallicity, ratio):
            ax.plot(
                log_temperature,
                curve,
                color=color,
                linewidth=1.8,
                label=fr"$Z/Z_\odot={z:g}$",
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel(r"$\log_{10}(T\,[\mathrm{K}])$")
        ax.set_ylabel(
            rf"Cooling-rate ratio "
            rf"$\Lambda({high_density:g})/\Lambda({low_density:g})$"
        )
        ax.set_title("Density dependence of CHIANTI cooling")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Metallicity", frameon=False)

        fig.savefig(output_path, dpi=180)
        print(f"Loaded: {table_path}")
        print(
            f"Using nearest table densities: {low_density:g} and "
            f"{high_density:g} cm^-3"
        )
        print(f"Wrote: {output_path}")
        return

    for color, z, rate in zip(
        colors, metallicity, cooling[:, :, density_index]
    ):
        ax.semilogy(
            log_temperature,
            rate,
            color=color,
            linewidth=1.8,
            label=fr"$Z/Z_\odot={z:g}$",
        )

    ax.set_xlabel(r"$\log_{10}(T\,[\mathrm{K}])$")
    ax.set_ylabel(r"Cooling coefficient $\Lambda$ [erg cm$^{3}$ s$^{-1}$]")
    ax.set_title(
        rf"CHIANTI cooling rate at $n_e={selected_density:g}\ \mathrm{{cm}}^{{-3}}$"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Metallicity", frameon=False)

    fig.savefig(output_path, dpi=180)
    print(f"Loaded: {table_path}")
    print(
        f"Requested ne = {args.electron_density:g} cm^-3; "
        f"using nearest table value ne = {selected_density:g} cm^-3"
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
