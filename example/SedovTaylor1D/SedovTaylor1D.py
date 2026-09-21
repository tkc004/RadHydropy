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
import matplotlib.pyplot as plt

from example import example_utils as eu
from example.SedovTaylor1D import tools as et

et.set_plot_style()

DEFAULT_CONFIG = Path(__file__).resolve().with_name("SedovTaylor1D.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    exampleparams = config["example"]
    output = config["par"]["output"]
    eu.clean_previous_outputs(config)
    code_units_obj = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])

    config["_code_units"] = code_units_obj
    ric = et.build_initial_condition(config)
    ric.write(config["par"]["simulation"]["initial_condition_filename"])
    mainrun = Rsim(config["par"])
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    color_cycle = iter(plt.rcParams["axes.prop_cycle"])
    for outindex in exampleparams["output_indices"]:
        outfilename = Path(output["directory"]) / (
            output["filename_prefix"] + "_%03d" % outindex + ".hdf5"
        )
        et.plot_snapshot(
            outfilename,
            config,
            ls="none",
            marker="o",
            mfc="none",
            markevery=1,
            color=next(color_cycle)["color"],
        )
        # plt.ylim(ymax=2.0)
    figure_filename = Path(output["directory"]) / exampleparams["plot_filename"]
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 1D Sedov-Taylor example.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help='YAML file with config["par"] and initial_condition.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
