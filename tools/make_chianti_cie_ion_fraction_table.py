#!/usr/bin/env python3
"""Generate an HDF5 lookup table for CHIANTI CIE ion fractions.

The generated ``ion_fraction`` dataset has axes:

    ion_fraction[element, ion_stage, temperature]

where ``ion_stage`` is the number of electrons removed. For example, stage
0 is neutral hydrogen and stage 1 is H II.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DATABASE = (
    Path(__file__).resolve().parents[2] / "CHIANTI_11.0.2_database"
)
DEFAULT_IONEQ = DEFAULT_DATABASE / "ioneq" / "chianti.ioneq"
DEFAULT_OUTPUT = (
    DEFAULT_DATABASE
    / "cooling_tables"
    / "chianti_cie_ion_fractions.h5"
)

ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CHIANTI collisional-ionization-equilibrium table."
    )
    parser.add_argument(
        "--ioneq-file",
        type=Path,
        default=DEFAULT_IONEQ,
        help=f"CHIANTI .ioneq file. Default: {DEFAULT_IONEQ}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HDF5 file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def read_ioneq_file(filename):
    """Read the CHIANTI fixed-format .ioneq file without changing values."""
    lines = filename.read_text().splitlines()
    n_temperature, n_elements = (int(value) for value in lines[0].split())

    # CHIANTI uses Fortran fixed-width fields: 6 characters for temperatures
    # and 10 characters for each ion fraction. Fixed-width parsing is needed
    # because adjacent zero-valued fields may have no separating whitespace.
    log_temperature = np.asarray(
        [float(lines[1][6 * i : 6 * (i + 1)]) for i in range(n_temperature)]
    )
    if log_temperature.size != n_temperature:
        raise ValueError(
            f"Expected {n_temperature} temperatures, found "
            f"{log_temperature.size} in {filename}."
        )

    fractions = np.zeros(
        (n_elements, n_elements + 1, n_temperature), dtype=np.float64
    )

    row_count = 0
    for line in lines[2:]:
        if line[:5].strip() == "-1":
            break

        header = line[:6].split()
        values = np.asarray(
            [float(line[6 + 10 * i : 6 + 10 * (i + 1)]) for i in range(n_temperature)]
        )
        if len(header) != 2 or values.size != n_temperature:
            raise ValueError(f"Malformed ion-fraction row in {filename}: {line[:40]!r}")

        atomic_number, ion_stage = (int(value) for value in header)
        if not 1 <= atomic_number <= n_elements:
            raise ValueError(f"Invalid atomic number {atomic_number} in {filename}.")
        if not 1 <= ion_stage <= n_elements + 1:
            raise ValueError(f"Invalid ion stage {ion_stage} in {filename}.")

        # CHIANTI stores ion stage 1 as the neutral stage, so convert to the
        # physical charge state convention used in the output: 0 = neutral.
        fractions[atomic_number - 1, ion_stage - 1, :] = values
        row_count += 1

    if row_count == 0:
        raise ValueError(f"No ion-fraction rows found in {filename}.")

    return 10.0 ** log_temperature, fractions, n_elements


def main():
    args = parse_args()
    ioneq_file = args.ioneq_file.expanduser().resolve()
    output_file = args.output.expanduser().resolve()

    if not ioneq_file.is_file():
        raise FileNotFoundError(f"CHIANTI .ioneq file not found: {ioneq_file}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}. Use --overwrite to replace it."
        )

    temperature, fractions, n_elements = read_ioneq_file(ioneq_file)
    if n_elements > len(ELEMENT_SYMBOLS):
        raise ValueError(
            f"The file contains {n_elements} elements, but only "
            f"{len(ELEMENT_SYMBOLS)} element symbols are defined."
        )

    # Every populated element/stage should sum to unity over its ion stages.
    # Keep the source values but record the diagnostic for the user.
    fraction_sums = fractions.sum(axis=1)
    populated = fraction_sums > 0.0
    maximum_sum_error = np.max(np.abs(fraction_sums[populated] - 1.0))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_file, "w") as table:
        table.create_dataset("temperature_K", data=temperature)
        table.create_dataset("log10_temperature_K", data=np.log10(temperature))
        table.create_dataset(
            "atomic_number", data=np.arange(1, n_elements + 1, dtype=np.int32)
        )
        table.create_dataset(
            "element_symbol",
            data=np.asarray(ELEMENT_SYMBOLS[:n_elements], dtype=string_dtype),
        )
        table.create_dataset(
            "ion_stage",
            data=np.arange(n_elements + 1, dtype=np.int32),
        )
        table.create_dataset(
            "ion_fraction",
            data=fractions,
            compression="gzip",
            compression_opts=4,
        )

        table.attrs["description"] = (
            "CHIANTI collisional-ionization-equilibrium ion fractions."
        )
        table.attrs["source_file"] = str(ioneq_file)
        table.attrs["ion_stage_definition"] = (
            "Number of electrons removed: 0=neutral, Z=fully stripped."
        )
        table.attrs["axis_order"] = (
            "ion_fraction[element, ion_stage, temperature]"
        )
        table.attrs["maximum_ion_fraction_sum_error"] = maximum_sum_error

    print(f"Read: {ioneq_file}")
    print(f"Elements: {n_elements}")
    print(f"Temperature grid: {temperature.size} points")
    print(f"Maximum ion-fraction sum error: {maximum_sum_error:.3e}")
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()
