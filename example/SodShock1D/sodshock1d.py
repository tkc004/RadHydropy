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
from example import example_utils as eu
import matplotlib.pyplot as plt

from example.SodShock1D.tools import et

DEFAULT_CONFIG = Path(__file__).resolve().with_name("sodshock1d.yaml")


def main(config_filename=DEFAULT_CONFIG, riemann_solver=None):
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)
    config["initial_condition"]
    exampleparams = config["example"]
    if riemann_solver is not None:
        config["par"]["hydrodynamics"]["riemann_solver"] = riemann_solver
    output = config["par"]["output"]
    eu.clean_previous_outputs(config)
    code_units_obj = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])

    config["_code_units"] = code_units_obj
    writer = et.build_initial_condition(config)
    writer.write(config["par"]["simulation"]["initial_condition_filename"])
    mainrun = Rsim(config["par"])
    mainrun.RunAll()
    outindex = exampleparams["output_index"]
    outfilename = (
        Path(output["directory"]) / output["filename_prefix"] + "_%03d" % outindex + ".hdf5"
    )
    et.plot_snapshot(
        outfilename,
        config,
        ls="none",
        marker="o",
        mfc="none",
        markevery=5,
    )
    figure_filename = Path(output["directory"]) / exampleparams["plot_filename"]
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Sod shock example.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML file with par_config and initial_condition.",
    )
    parser.add_argument("--riemann-solver", choices=("Rusanov", "HLLC"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, riemann_solver=args.riemann_solver)
