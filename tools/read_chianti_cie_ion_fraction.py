#!/usr/bin/env python3
"""Read a CHIANTI CIE table and calculate electron density.

For a hydrogen-nuclei density nH, the calculation is

    ne = nH * sum_X [(nX / nH) * <charge_X(T)>].

The ion fractions are in collisional ionization equilibrium, so they do not
explicitly depend on density. Metallicity scales elements heavier than He.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DATABASE = (
    Path(__file__).resolve().parents[2] / "CHIANTI_11.0.2_database"
)
DEFAULT_TABLE = (
    DEFAULT_DATABASE / "cooling_tables" / "chianti_cie_ion_fractions.h5"
)
DEFAULT_ABUNDANCE = (
    DEFAULT_DATABASE / "abundance" / "sun_photospheric_2015_scott.abund"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate ne from CIE ion fractions, metallicity, nH, and T."
    )
    parser.add_argument(
        "--metallicity",
        type=float,
        required=True,
        help="Metallicity Z/Zsun. H and He are not scaled.",
    )
    parser.add_argument(
        "--nH",
        type=float,
        required=True,
        help="Hydrogen-nuclei density in cm^-3.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        required=True,
        help="Temperature(s) in K.",
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--abundance-file", type=Path, default=DEFAULT_ABUNDANCE)
    parser.add_argument(
        "--show-breakdown",
        action="store_true",
        help="Print each element's contribution to ne/nH.",
    )
    return parser.parse_args()


def read_abundances(filename):
    """Read CHIANTI logarithmic abundances, where log10(H)=12."""
    atomic_number = []
    log_abundance = []
    symbols = []

    for line in filename.read_text().splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        atomic_number.append(int(fields[0]))
        log_abundance.append(float(fields[1]))
        symbols.append(fields[2])

    abundance = 10.0 ** (np.asarray(log_abundance) - 12.0)
    return np.asarray(atomic_number), np.asarray(symbols), abundance


def calculate_electron_density(table_file, abundance_file, metallicity, nH, temperatures):
    if metallicity < 0:
        raise ValueError("metallicity must be non-negative")
    if nH < 0:
        raise ValueError("nH must be non-negative")

    with h5py.File(table_file, "r") as table:
        log_temperature_grid = table["log10_temperature_K"][:]
        fractions = table["ion_fraction"][:]
        table_symbols = table["element_symbol"][:]
        table_atomic_number = table["atomic_number"][:]
        ion_stage = table["ion_stage"][:]

    abundance_atomic_number, abundance_symbols, solar_abundance = read_abundances(
        abundance_file
    )
    if not np.array_equal(table_atomic_number, abundance_atomic_number):
        raise ValueError("Ion-fraction and abundance files contain different elements.")

    temperatures = np.asarray(temperatures, dtype=float)
    log_temperatures = np.log10(temperatures)
    if np.any(temperatures <= 0):
        raise ValueError("temperatures must be positive")
    if np.any(log_temperatures < log_temperature_grid[0]) or np.any(
        log_temperatures > log_temperature_grid[-1]
    ):
        raise ValueError(
            f"temperature range is {10**log_temperature_grid[0]:g} to "
            f"{10**log_temperature_grid[-1]:g} K"
        )

    electron_fraction = np.zeros(temperatures.size)
    contributions = np.zeros((temperatures.size, len(solar_abundance)))

    for element_index, solar_ratio in enumerate(solar_abundance):
        element_fractions = np.vstack(
            [
                np.interp(log_temperatures, log_temperature_grid, curve)
                for curve in fractions[element_index]
            ]
        ).T
        element_fractions /= element_fractions.sum(axis=1, keepdims=True)
        mean_charge = element_fractions @ ion_stage

        scale = 1.0 if element_index < 2 else metallicity
        contributions[:, element_index] = scale * solar_ratio * mean_charge

    electron_fraction = contributions.sum(axis=1)
    electron_density = nH * electron_fraction
    return (
        temperatures,
        electron_density,
        electron_fraction,
        contributions,
        table_symbols,
    )


def main():
    args = parse_args()
    table_file = args.table.expanduser().resolve()
    abundance_file = args.abundance_file.expanduser().resolve()

    result = calculate_electron_density(
        table_file=table_file,
        abundance_file=abundance_file,
        metallicity=args.metallicity,
        nH=args.nH,
        temperatures=args.temperature,
    )
    temperatures, electron_density, electron_fraction, contributions, symbols = result

    print(f"Metallicity Z/Zsun = {args.metallicity:g}")
    print(f"nH = {args.nH:.6e} cm^-3")
    for temperature, ne, fraction in zip(
        temperatures, electron_density, electron_fraction
    ):
        print(
            f"T = {temperature:.6e} K: "
            f"ne = {ne:.6e} cm^-3, ne/nH = {fraction:.6e}"
        )

    if args.show_breakdown:
        for row, temperature in zip(contributions, temperatures):
            print(f"\nElectron contribution at T = {temperature:g} K:")
            for symbol, contribution in zip(symbols, row):
                if contribution > 0:
                    print(f"  {symbol.decode() if isinstance(symbol, bytes) else symbol}: "
                          f"{contribution:.6e} electrons per H nucleus")


if __name__ == "__main__":
    main()
