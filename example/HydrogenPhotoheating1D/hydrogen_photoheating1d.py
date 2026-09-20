# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Optically thin hydrogen photoheating and recombination parcel.

An initially neutral pure-hydrogen parcel with fixed total density is exposed
to a spatially uniform ionizing radiation field. The radiation is treated as
optically thin, so the photon density is fixed while the source is on and set
to zero when the source switches off. The run writes HDF5 snapshots, reloads
them, and plots the thermal and ionization history from those outputs.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "radhydropy-cache"
mplconfig_dir = Path(tempfile.gettempdir()) / "radhydropy-matplotlib"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from example import example_utils as eu

import radhydropy.io as rio
from example.HydrogenPhotoheating1D.tools import et

DEFAULT_CONFIG = Path(__file__).resolve().with_name("hydrogen_photoheating1d.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    initial_condition = config["initial_condition"]
    exampleparams = config["example"]
    output = config["par"]["output"]
    eu.clean_previous_outputs(config)

    reference = et.reference_values(
        exampleparams["photon_flux"],
        initial_condition["hydrogen_number_density"],
        exampleparams["excess_photoionization_energy"],
        exampleparams["sigma_gamma"],
        exampleparams["thermal_equilibrium_timescale"],
    )

    ric = et.build_initial_condition(config)
    ic_filename = config["par"]["simulation"]["initial_condition_filename"]
    ric.write(ic_filename)
    sim = rio.loadhdf5(config, ic_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    et.RunHydrogenPhotoheating(
        sim,
        exampleparams["source_switch_time"],
        reference["photon_number_density_cgs_cm3_unyt"],
        outputtime=0,
    )

    outputfiles = et.output_files(
        output["directory"],
        output["filename_prefix"],
    )
    history = et.load_history_from_outputs(
        outputfiles,
        config,
    )

    figure_filename = Path(output["directory"]) / exampleparams["plot_filename"]
    et.save_history_plot(history, str(figure_filename), reference)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the optically thin hydrogen photoheating example.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML file containing nested par, initial_condition, and example sections.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
