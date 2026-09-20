# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "radhydropy-matplotlib"),
)
import matplotlib as mpl

mpl.use("Agg")
import example_utils as eu
import matplotlib.pyplot as plt

import tools as et

DEFAULT_CONFIG = Path(__file__).resolve().with_name("Inflow1d.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    output_time_filename = config["par"]["output"].get("time_list_filename")
    if output_time_filename:
        config["par"]["output"]["time_list_filename"] = (
            Path(config_filename).resolve().parent / output_time_filename
        )
    output = config["par"]["output"]
    eu.clean_previous_outputs(config)
    code_units_obj = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])

    config["_code_units"] = code_units_obj
    ric = et.build_initial_condition(config)
    ric.write(
        config["par"]["simulation"]["initial_condition_filename"],
        validate=True,
    )
    mainrun = Rsim(config["par"])
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    outputfiles = sorted(
        Path(output["directory"]).glob(f"{output['filename_prefix']}_*.hdf5"),
    )
    color_cycle = iter(plt.rcParams["axes.prop_cycle"])
    for outfilename in outputfiles:
        et.plot_snapshot(
            str(outfilename),
            config,
            ls="none",
            marker="o",
            mfc="none",
            markevery=1,
            color=next(color_cycle)["color"],
        )
    figure_filename = Path(output["directory"]) / config["example"]["plot_filename"]
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 1D inflow example.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML file with nested runtime and initial-condition settings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
