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
    os.path.join(tempfile.gettempdir(), "radhydropy-matplotlib"),
)
import matplotlib as mpl

mpl.use("Agg")
import example_utils as eu
import matplotlib.pyplot as plt

from example.Outflow1d import tools as et

DEFAULT_CONFIG = Path(__file__).resolve().with_name("Outflow1d.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    config["initial_condition"]
    exampleparams = config["example"]
    eu.clean_previous_outputs(config)
    code_units_obj = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])

    config["_code_units"] = code_units_obj
    ric = et.build_initial_condition(config)
    ric.write(config["par"]["simulation"]["initial_condition_filename"], validate=True)
    mainrun = Rsim(config["par"])
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    color_cycle = iter(plt.rcParams["axes.prop_cycle"])
    for outindex in exampleparams["output_indices"]:
        outfilename = os.path.join(
            config["par"]["output"]["directory"],
            config["par"]["output"]["filename_prefix"] + "_%03d" % outindex + ".hdf5",
        )
        et.plot_snapshot(
            outfilename,
            config,
            ls="none",
            marker="o",
            mfc="none",
            markevery=exampleparams["plot"]["markevery"],
            color=next(color_cycle)["color"],
        )
    figure_filename = os.path.join(
        config["par"]["output"]["directory"],
        exampleparams["plot"]["filename"],
    )
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 1D outflow example.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML file with nested runtime and initial-condition settings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
