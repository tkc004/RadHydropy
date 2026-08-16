#!/usr/bin/env python3
"""Plot HM12 photoionization-equilibrium heating and net cooling rates.

The grouped HM12 tables use the axis order
``temperature, density, redshift, metallicity``.  This script selects a
metallicity plane, interpolates in log hydrogen density and redshift, and
plots the rates at a requested hydrogen density.

Net cooling is defined as ``cooling - photoheating``.  The plotted net-rate
quantity is ``log10(abs(cooling - photoheating))``; the absolute value keeps
both net cooling and net heating visible on the logarithmic plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPOSITORY_ROOT / "metal_pie_table" / "metal_pie_hm12_total.h5"
DEFAULT_REDSHIFTS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot log10 net cooling and photoheating rates from an HM12 "
            "MetalPIE HDF5 table."
        )
    )
    parser.add_argument(
        "table",
        nargs="?",
        type=Path,
        default=DEFAULT_TABLE,
        help=f"HM12 MetalPIE table. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--hydrogen-density",
        type=float,
        default=1.0e-4,
        help="Hydrogen density in cm^-3. Default: 1e-4.",
    )
    parser.add_argument(
        "--log-hydrogen-densities",
        nargs="+",
        type=float,
        default=None,
        metavar="LOG_NH",
        help=(
            "Generate one figure for each log10(nH/cm^-3) value. "
            "Overrides --hydrogen-density."
        ),
    )
    parser.add_argument(
        "--redshifts",
        nargs="+",
        type=float,
        default=DEFAULT_REDSHIFTS,
        metavar="Z",
        help=(
            "Redshifts to plot. Values between table nodes are linearly "
            "interpolated. Default: 0 2 4 6 8 10 12."
        ),
    )
    parser.add_argument(
        "--metallicity",
        type=float,
        default=1.0,
        help=(
            "Metallicity in Z/Zsun. The nearest table plane is used. "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--max-log-heating",
        type=float,
        default=None,
        metavar="LOG_RATE",
        help=(
            "Mask heating-rate points with log10(heating) greater than "
            "this threshold."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to a name derived from the table.",
    )
    return parser.parse_args()


def interpolate_axis(values, axis_values, target, axis_number=0):
    """Interpolate values along one axis, clipping target to the table range."""
    target = float(np.clip(target, axis_values[0], axis_values[-1]))
    upper = int(np.searchsorted(axis_values, target, side="right"))
    if upper == 0:
        return np.take(values, 0, axis=axis_number)
    if upper == len(axis_values):
        return np.take(values, -1, axis=axis_number)
    lower = upper - 1
    weight = (target - axis_values[lower]) / (
        axis_values[upper] - axis_values[lower]
    )
    low = np.take(values, lower, axis=axis_number)
    high = np.take(values, upper, axis=axis_number)
    return (1.0 - weight) * low + weight * high


def select_metallicity(table, requested):
    metallicities = np.asarray(table["axes/metallicity_Zsun"], dtype=float)
    if requested is None:
        index = 0
    else:
        index = int(np.argmin(np.abs(metallicities - requested)))
    return index, float(metallicities[index])


def load_rates(table_path, hydrogen_density, redshifts, metallicity):
    with h5py.File(table_path, "r") as handle:
        group = handle["MetalPIE"]
        axes = group["axes"]
        temperature = np.asarray(axes["log10_temperature_K"], dtype=float)
        log_density = np.asarray(
            axes["log10_hydrogen_density_cm-3"], dtype=float
        )
        table_redshifts = np.asarray(axes["redshift"], dtype=float)
        metallicity_index, selected_metallicity = select_metallicity(
            group, metallicity
        )

        if hydrogen_density <= 0.0:
            raise ValueError("--hydrogen-density must be positive.")
        requested_log_density = np.log10(hydrogen_density)
        cooling_name = "cooling_erg_cm3_s"
        heating_name = "photoheating_erg_cm3_s"
        if group.attrs.get("component") == "metals":
            cooling_name = "metal_cooling_erg_cm3_s"
            heating_name = "metal_photoheating_erg_cm3_s"

        cooling = np.asarray(group[f"rates/{cooling_name}"][:, :, :, metallicity_index])
        heating = np.asarray(group[f"rates/{heating_name}"][:, :, :, metallicity_index])

    density_cooling = interpolate_axis(
        cooling, log_density, requested_log_density, axis_number=1
    )
    density_heating = interpolate_axis(
        heating, log_density, requested_log_density, axis_number=1
    )
    cooling_by_redshift = np.stack(
        [interpolate_axis(density_cooling, table_redshifts, z, axis_number=1)
         for z in redshifts]
    )
    heating_by_redshift = np.stack(
        [interpolate_axis(density_heating, table_redshifts, z, axis_number=1)
         for z in redshifts]
    )
    return temperature, cooling_by_redshift, heating_by_redshift, selected_metallicity


def plot_rates(
    table_path,
    output_path,
    hydrogen_density,
    redshifts,
    metallicity,
    max_log_heating=None,
):
    log_temperature, cooling, heating, selected_metallicity = load_rates(
        table_path, hydrogen_density, redshifts, metallicity
    )
    net_cooling = cooling - heating
    absolute_net_cooling = np.maximum(np.abs(net_cooling), 1.0e-99)
    if max_log_heating is not None:
        heating_log = np.log10(np.maximum(heating, 1.0e-99))
        valid_heating = heating_log <= max_log_heating
        absolute_net_cooling = np.where(
            valid_heating, absolute_net_cooling, np.nan
        )
        heating = np.where(heating_log <= max_log_heating, heating, np.nan)

    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(redshifts)))
    for color, redshift, net_curve, heating_curve in zip(
        colors, redshifts, absolute_net_cooling, heating
    ):
        label = rf"$z={redshift:g}$"
        ax.plot(
            log_temperature,
            np.log10(net_curve),
            color=color,
            linewidth=1.8,
            label=label + r" $|$net$|$",
        )
        ax.plot(
            log_temperature,
            np.log10(np.maximum(heating_curve, 1.0e-99)),
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=label + " heating",
        )

    component = "metals" if "metals" in table_path.stem else "H/He + metals"
    ax.set_xlabel(r"$\log_{10}(T\,[\mathrm{K}])$")
    ax.set_ylabel(
        r"$\log_{10}(\mathrm{rate}\,[\mathrm{erg\,cm^{-3}\,s^{-1}}])$"
    )
    ax.set_title(
        rf"HM12 {component}: $n_\mathrm{{H}}={hydrogen_density:g}\,\mathrm{{cm^{{-3}}}}$, "
        rf"$Z/Z_\odot={selected_metallicity:g}$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return selected_metallicity


def main():
    args = parse_args()
    table_path = args.table.expanduser().resolve()
    log_densities = (
        args.log_hydrogen_densities
        if args.log_hydrogen_densities is not None
        else [np.log10(args.hydrogen_density)]
    )
    for log_density in log_densities:
        hydrogen_density = 10.0 ** log_density
        if args.output is None:
            output_path = table_path.with_name(
                f"{table_path.stem}_rates_lognH_{log_density:g}.png"
            )
        else:
            output_path = args.output
            if len(log_densities) > 1:
                output_path = args.output.with_name(
                    f"{args.output.stem}_lognH_{log_density:g}{args.output.suffix}"
                )
        metallicity = plot_rates(
            table_path,
            output_path,
            hydrogen_density,
            args.redshifts,
            args.metallicity,
            args.max_log_heating,
        )
        print(f"Loaded: {table_path}")
        print(
            f"Using log10(nH/cm^-3) = {log_density:g}, "
            f"nH = {hydrogen_density:g} cm^-3, "
            f"metallicity = {metallicity:g} Z/Zsun"
        )
        print(f"Redshifts: {', '.join(f'{z:g}' for z in args.redshifts)}")
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
