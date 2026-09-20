# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Fixed-radiation hydrogen photoionization box.

The gas starts neutral at ``T = 2e4 K`` and ``nH = 1 cm^-3``. A fixed,
spatially uniform photon number density photoionizes the gas while the
radiation-field evolution and thermal source update are disabled. The run
stops once the gas is 99 percent ionized, writes HDF5 snapshots, reloads them,
and plots the neutral-fraction evolution against the analytic fixed-field
solution.
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
from example.HydrogenPhotoionization1D.tools import et

DEFAULT_CONFIG = Path(__file__).resolve().with_name("hydrogen_photoionization1d.yaml")


def RunHydrogenPhotoionization(sim, target_neutral_fraction, outputtime=0):  # noqa: N802
    """Run the fixed-field photoionization example until neutral fraction falls."""
    return sim.RunAll(
        outputtime=outputtime,
        mode="hydro_sources",
        stop_condition=lambda runner: et.mean_neutral_fraction(runner) <= target_neutral_fraction,
    )


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    config["initial_condition"]
    exampleparams = config["example"]
    output = config["par"]["output"]
    eu.clean_previous_outputs(config)
    ric = et.build_initial_condition(config)
    ic_filename = config["par"]["simulation"]["initial_condition_filename"]
    ric.write(ic_filename)
    sim = rio.loadhdf5(config, ic_filename)
    RunHydrogenPhotoionization(
        sim,
        exampleparams["target_neutral_fraction"],
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
    et.save_history_plot(
        history,
        str(figure_filename),
        config,
        exampleparams["target_neutral_fraction"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed-radiation hydrogen photoionization example.",
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
