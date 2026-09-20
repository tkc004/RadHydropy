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

from radhydropy.gravity import Gravity
from radhydropy.units import CodeUnits

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "radhydropy-matplotlib"),
)
import matplotlib as mpl

mpl.use("Agg")
from example import example_utils as eu
import matplotlib.pyplot as plt

import radhydropy.io as rio
from example.BallisticInfallSphericalPointMass1D import tools as et

DEFAULT_CONFIG = (
    Path(__file__)
    .resolve()
    .with_name(
        "ballistic_infall_spherical_point_mass1d.yaml",
    )
)


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    nested = eu.load_nested_example_config(config_filename)

    initial_condition = nested["initial_condition"]
    eu.clean_previous_outputs(nested)
    code_units_obj = CodeUnits.from_mapping(nested["par"]["units"]["CodeUnits"])

    nested["_code_units"] = code_units_obj
    ric = et.build_initial_condition(nested)
    ric.write(nested["par"]["simulation"]["initial_condition_filename"])

    mainrun = rio.loadhdf5(
        nested,
        nested["par"]["simulation"]["initial_condition_filename"],
    )
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()
    mainrun.par.gravity = Gravity(
        externalgravity=True,
        acceleration=et.point_mass_acceleration(
            initial_condition["point_mass"],
            code_unit_system=code_units_obj,
        ),
        code_units=code_units_obj,
    )
    mainrun.Run(mode="hydro")

    final_outfile = Path(nested["par"]["output"]["directory"]) / (
        nested["par"]["output"]["filename_prefix"] + "_001.hdf5"
    )
    et.plot_snapshot(
        final_outfile,
        nested,
        ls="none",
        marker="o",
        mfc="none",
        markevery=1,
        color="C0",
    )
    figure_filename = (
        Path(nested["par"]["output"]["directory"]) / "BallisticInfallSphericalPointMass1D.jpg"
    )
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a spherical ballistic-infall gravity check example.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML file with par_config and initial_condition.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
