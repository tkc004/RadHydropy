#!/usr/bin/env python3
"""
Generate an optically-thin CHIANTI cooling table as a function of:

    metallicity Z/Zsun,
    temperature T [K],
    electron density ne [cm^-3].

The output cooling coefficient Lambda is intended for use as:

    cooling_rate_per_volume = ne * nH * Lambda

with Lambda in approximately erg cm^3 s^-1.

This script uses the common approximation:

    Lambda(T, ne, Z) =
        Lambda_HHe(T, ne)
        + (Z/Zsun) * [Lambda_solar(T, ne) - Lambda_HHe(T, ne)]

where Lambda_HHe is computed by asking CHIANTI to include only elements with
abundance larger than min_abund_hhe, typically 1e-2, which keeps H and He and
drops metals.

Requirements
------------
    pip install ChiantiPy h5py numpy

The script automatically uses the repository's
``CHIANTI_11.0.2_database`` directory when run from ``RadHydropy/tools``.
You can override that location with ``--xuvtop`` or the ``XUVTOP`` environment
variable, e.g.

    python make_chianti_cooling_table.py --xuvtop /path/to/chianti/database

Example
-------
    python make_chianti_cooling_table.py \
        --output chianti_cooling_table.h5 \
        --abundance sun_photospheric_2015_scott \
        --logT-min 4.0 --logT-max 8.5 --nT 181 \
        --logne-min -8 --logne-max 6 --nne 71 \
        --workers 4 \
        --metallicities 0.0 0.01 0.03 0.1 0.3 1.0 3.0
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py

# The database is kept next to the repository package, rather than inside
# this tools directory:
#
#   RadHydropy/
#   ├── CHIANTI_11.0.2_database/
#   └── RadHydropy/tools/make_chianti_cooling_table.py
#
# Resolve this from __file__ so the script can be run after changing into
# RadHydropy/tools, without requiring the caller to set XUVTOP first.
DEFAULT_XUVTOP = (
    Path(__file__).resolve().parents[2] / "CHIANTI_11.0.2_database"
)

ch = None
_WORKER_XUVTOP = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CHIANTI cooling table in HDF5 format."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="chianti_cooling_table.h5",
        help="Output HDF5 filename.",
    )

    parser.add_argument(
        "--xuvtop",
        type=str,
        default=None,
        help=(
            "Path to the CHIANTI database. Overrides XUVTOP. If neither is "
            "provided, use the repository's CHIANTI_11.0.2_database directory."
        ),
    )

    parser.add_argument(
        "--abundance",
        type=str,
        default="sun_photospheric_2015_scott",
        help=(
            "CHIANTI abundance file name without extension. "
            "Examples: sun_photospheric_2015_scott, sun_coronal_2012_schmelz."
        ),
    )

    parser.add_argument(
        "--logT-min",
        type=float,
        default=2.0,
        help="Minimum log10 temperature in K.",
    )

    parser.add_argument(
        "--logT-max",
        type=float,
        default=8.5,
        help="Maximum log10 temperature in K.",
    )

    parser.add_argument(
        "--nT",
        type=int,
        default=400,
        help="Number of temperature grid points.",
    )

    parser.add_argument(
        "--logne-min",
        type=float,
        default=-6.0,
        help="Minimum log10 electron density in cm^-3.",
    )

    parser.add_argument(
        "--logne-max",
        type=float,
        default=6.0,
        help="Maximum log10 electron density in cm^-3.",
    )

    parser.add_argument(
        "--nne",
        type=int,
        default=256,
        help="Number of electron-density grid points.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes used for the independent density "
            "calculations. Default: 1 (serial)."
        ),
    )

    parser.add_argument(
        "--metallicities",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        help="List of metallicities Z/Zsun.",
    )

    parser.add_argument(
        "--min-abund-solar",
        type=float,
        default=1.0e-12,
        help=(
            "Minimum abundance threshold for the solar/all-elements calculation. "
            "Small values include more elements."
        ),
    )

    parser.add_argument(
        "--min-abund-hhe",
        type=float,
        default=1.0e-2,
        help=(
            "Minimum abundance threshold for the H/He-only calculation. "
            "1e-2 usually keeps H and He but excludes metals."
        ),
    )

    parser.add_argument(
        "--no-continuum",
        action="store_true",
        help="Disable continuum cooling. By default continuum is included.",
    )

    parser.add_argument(
        "--clip-negative-metal-cooling",
        action="store_true",
        help=(
            "Clip Lambda_solar - Lambda_HHe to be non-negative. "
            "Useful only to remove tiny numerical subtraction artifacts."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    return parser.parse_args()


def check_environment(xuvtop_arg=None):
    global ch

    # Prefer an explicit CLI value, then the environment, then the database
    # bundled alongside this repository.
    xuvtop = Path(
        xuvtop_arg or os.environ.get("XUVTOP", DEFAULT_XUVTOP)
    ).expanduser().resolve()

    if not xuvtop.is_dir():
        raise RuntimeError(f"CHIANTI database directory does not exist: {xuvtop}")

    # ChiantiPy reads XUVTOP during import, so set it before importing the
    # package rather than only validating it after import.
    os.environ["XUVTOP"] = str(xuvtop)

    if ch is None:
        try:
            import ChiantiPy.core as ch_module
        except ImportError as exc:
            raise RuntimeError(
                "Could not import ChiantiPy. Install it with:\n\n"
                "    pip install ChiantiPy\n"
            ) from exc
        ch = ch_module

    return str(xuvtop)


def _initialize_cooling_worker(xuvtop):
    """Configure CHIANTI inside a spawned worker process."""
    global _WORKER_XUVTOP
    _WORKER_XUVTOP = check_environment(xuvtop)


def _get_radloss_rate(radloss_object):
    """
    Extract the radiative-loss coefficient from a ChiantiPy radLoss object.

    In common ChiantiPy versions this is:

        radloss_object.RadLoss["rate"]

    This helper makes the error message clearer if the API changes.
    """
    if not hasattr(radloss_object, "RadLoss"):
        raise RuntimeError(
            "ChiantiPy radLoss object does not have attribute RadLoss. "
            "Your ChiantiPy version may have a different API."
        )

    radloss_dict = radloss_object.RadLoss

    if "rate" not in radloss_dict:
        raise RuntimeError(
            "Could not find key 'rate' in radLoss.RadLoss. Available keys are: "
            f"{list(radloss_dict.keys())}"
        )

    return np.asarray(radloss_dict["rate"], dtype=float)


def compute_cooling_vs_T_for_density(
    temperatures,
    electron_density,
    abundance,
    min_abund,
    do_continuum=True,
):
    """
    Compute Lambda(T, ne) for one density and all temperatures.

    Parameters
    ----------
    temperatures : array
        Temperature grid in K.

    electron_density : float
        Electron density in cm^-3.

    abundance : str
        CHIANTI abundance file name.

    min_abund : float
        Minimum abundance threshold used by CHIANTI.

    do_continuum : bool
        Include continuum cooling.

    Returns
    -------
    rate : ndarray
        Cooling coefficient array with same length as temperatures.
    """

    # ChiantiPy uses doContinuum as int/bool depending on version.
    rl = ch.radLoss(
        temperatures,
        electron_density,
        abundance=abundance,
        minAbund=min_abund,
        doContinuum=do_continuum,
        verbose=False,
    )

    rate = _get_radloss_rate(rl)

    if rate.shape[0] != temperatures.shape[0]:
        raise RuntimeError(
            "Unexpected CHIANTI output shape. "
            f"Expected length {temperatures.shape[0]}, got shape {rate.shape}."
        )

    return rate


def _compute_density_column(task):
    (
        index,
        electron_density,
        temperatures,
        abundance,
        min_abund,
        do_continuum,
    ) = task

    print(
        f"  started density {index + 1:4d}: ne = {electron_density:.6e} cm^-3",
        flush=True,
    )

    rate = compute_cooling_vs_T_for_density(
        temperatures=temperatures,
        electron_density=electron_density,
        abundance=abundance,
        min_abund=min_abund,
        do_continuum=do_continuum,
    )

    return index, electron_density, rate


def compute_cooling_grid(
    temperatures,
    electron_densities,
    abundance,
    min_abund,
    do_continuum=True,
    label="",
    workers=1,
    xuvtop=None,
):
    """
    Compute Lambda(T, ne) on a 2D grid.

    Returns
    -------
    cooling : ndarray
        Shape is (nT, nne).
    """

    nT = len(temperatures)
    nne = len(electron_densities)

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers > 1 and xuvtop is None:
        raise ValueError("xuvtop is required when workers is greater than 1")

    cooling = np.zeros((nT, nne), dtype=float)

    print()
    print(f"Computing {label} cooling grid")
    print(f"  abundance  = {abundance}")
    print(f"  minAbund   = {min_abund:g}")
    print(f"  continuum  = {do_continuum}")
    print(f"  nT         = {nT}")
    print(f"  nne        = {nne}")
    print(f"  workers    = {workers}")
    print()

    start = time.time()

    if workers == 1:
        for j, ne in enumerate(electron_densities):
            rate = compute_cooling_vs_T_for_density(
                temperatures=temperatures,
                electron_density=ne,
                abundance=abundance,
                min_abund=min_abund,
                do_continuum=do_continuum,
            )
            cooling[:, j] = rate
            print(
                f"  density {j + 1:4d}/{nne:4d}: "
                f"ne = {ne:.6e} cm^-3, "
                f"total = {time.time() - start:.2f} s"
            )
    else:
        tasks = (
            (
                j,
                ne,
                temperatures,
                abundance,
                min_abund,
                do_continuum,
            )
            for j, ne in enumerate(electron_densities)
        )
        # Use spawn explicitly. It prevents CHIANTI state imported by the
        # parent process from being inherited by workers, which can deadlock
        # CHIANTI calculations when fork is used.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_cooling_worker,
            initargs=(xuvtop,),
        ) as executor:
            futures = {
                executor.submit(_compute_density_column, task): task[0]
                for task in tasks
            }
            print(f"  submitted {len(futures)} density calculations", flush=True)
            for completed, future in enumerate(as_completed(futures), start=1):
                j, ne, rate = future.result()
                cooling[:, j] = rate
                print(
                    f"  density column {j + 1:4d}/{nne:4d}: "
                    f"ne = {ne:.6e} cm^-3, "
                    f"completed = {completed:4d}/{nne:4d}, "
                    f"total = {time.time() - start:.2f} s",
                    flush=True,
                )

    return cooling


def build_metallicity_table(
    cooling_solar,
    cooling_hhe,
    metallicities,
    clip_negative_metal_cooling=False,
):
    """
    Build Lambda(Z, T, ne) from solar and H/He cooling.

    Parameters
    ----------
    cooling_solar : ndarray
        Shape (nT, nne).

    cooling_hhe : ndarray
        Shape (nT, nne).

    metallicities : ndarray
        Z/Zsun values.

    clip_negative_metal_cooling : bool
        If True, set tiny negative metal contribution to zero.

    Returns
    -------
    cooling_table : ndarray
        Shape (nZ, nT, nne).
    """

    metal_cooling_solar = cooling_solar - cooling_hhe

    if clip_negative_metal_cooling:
        metal_cooling_solar = np.maximum(metal_cooling_solar, 0.0)

    nZ = len(metallicities)
    nT, nne = cooling_solar.shape

    cooling_table = np.zeros((nZ, nT, nne), dtype=float)

    for k, z in enumerate(metallicities):
        cooling_table[k, :, :] = cooling_hhe + z * metal_cooling_solar

    return cooling_table, metal_cooling_solar


def write_hdf5(
    filename,
    temperatures,
    electron_densities,
    metallicities,
    cooling_table,
    cooling_solar,
    cooling_hhe,
    cooling_metals_solar,
    args,
    xuvtop,
):
    """
    Write table and metadata to HDF5.
    """

    with h5py.File(filename, "w") as f:
        f.create_dataset("temperature_K", data=temperatures)
        f.create_dataset("log10_temperature_K", data=np.log10(temperatures))

        f.create_dataset("electron_density_cm-3", data=electron_densities)
        f.create_dataset("log10_electron_density_cm-3", data=np.log10(electron_densities))

        f.create_dataset("metallicity_Zsun", data=metallicities)

        f.create_dataset(
            "cooling_erg_cm3_s",
            data=cooling_table,
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "cooling_solar_erg_cm3_s",
            data=cooling_solar,
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "cooling_HHe_erg_cm3_s",
            data=cooling_hhe,
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "cooling_metals_solar_erg_cm3_s",
            data=cooling_metals_solar,
            compression="gzip",
            compression_opts=4,
        )

        f.attrs["description"] = (
            "CHIANTI optically-thin radiative cooling table. "
            "Main dataset cooling_erg_cm3_s has shape "
            "(metallicity, temperature, electron_density)."
        )

        f.attrs["cooling_usage"] = "volumetric cooling rate = ne * nH * Lambda"
        f.attrs["cooling_units"] = "erg cm^3 s^-1"
        f.attrs["temperature_units"] = "K"
        f.attrs["electron_density_units"] = "cm^-3"
        f.attrs["metallicity_units"] = "Z/Zsun"

        f.attrs["abundance"] = args.abundance
        f.attrs["xuvtop"] = xuvtop
        f.attrs["min_abund_solar"] = args.min_abund_solar
        f.attrs["min_abund_hhe"] = args.min_abund_hhe
        f.attrs["do_continuum"] = not args.no_continuum
        f.attrs["metallicity_scaling"] = (
            "Lambda(Z) = Lambda_HHe + (Z/Zsun) * "
            "(Lambda_solar - Lambda_HHe)"
        )
        f.attrs["axis_order"] = "cooling_erg_cm3_s[metallicity, temperature, electron_density]"


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print()
        print(f"ERROR: output file already exists: {args.output}")
        print("Use --overwrite to replace it.")
        print()
        sys.exit(1)

    xuvtop = check_environment(args.xuvtop)

    temperatures = np.logspace(args.logT_min, args.logT_max, args.nT)
    electron_densities = np.logspace(args.logne_min, args.logne_max, args.nne)
    metallicities = np.asarray(args.metallicities, dtype=float)

    do_continuum = not args.no_continuum

    print()
    print("CHIANTI cooling-table generation")
    print("================================")
    print(f"XUVTOP                  = {xuvtop}")
    print(f"Output file             = {args.output}")
    print(f"Abundance               = {args.abundance}")
    print(f"log10(T/K) range         = {args.logT_min} to {args.logT_max}")
    print(f"log10(ne/cm^-3) range    = {args.logne_min} to {args.logne_max}")
    print(f"Metallicities Z/Zsun    = {metallicities}")
    print(f"Include continuum       = {do_continuum}")
    print(f"Worker processes        = {args.workers}")
    print()

    cooling_solar = compute_cooling_grid(
        temperatures=temperatures,
        electron_densities=electron_densities,
        abundance=args.abundance,
        min_abund=args.min_abund_solar,
        do_continuum=do_continuum,
        label="solar/all-elements",
        workers=args.workers,
        xuvtop=xuvtop,
    )

    cooling_hhe = compute_cooling_grid(
        temperatures=temperatures,
        electron_densities=electron_densities,
        abundance=args.abundance,
        min_abund=args.min_abund_hhe,
        do_continuum=do_continuum,
        label="H/He-only",
        workers=args.workers,
        xuvtop=xuvtop,
    )

    cooling_table, cooling_metals_solar = build_metallicity_table(
        cooling_solar=cooling_solar,
        cooling_hhe=cooling_hhe,
        metallicities=metallicities,
        clip_negative_metal_cooling=args.clip_negative_metal_cooling,
    )

    write_hdf5(
        filename=args.output,
        temperatures=temperatures,
        electron_densities=electron_densities,
        metallicities=metallicities,
        cooling_table=cooling_table,
        cooling_solar=cooling_solar,
        cooling_hhe=cooling_hhe,
        cooling_metals_solar=cooling_metals_solar,
        args=args,
        xuvtop=xuvtop,
    )

    print()
    print("Done.")
    print(f"Wrote: {args.output}")
    print()
    print("Main dataset:")
    print("  cooling_erg_cm3_s[metallicity, temperature, electron_density]")
    print()
    print("Use as:")
    print("  volumetric_cooling_rate = ne * nH * Lambda")
    print()


if __name__ == "__main__":
    main()
