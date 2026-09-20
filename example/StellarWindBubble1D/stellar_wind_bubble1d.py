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

from example.StellarWindBubble1D import tools as et

DEFAULT_CONFIG = (
    Path(__file__)
    .resolve()
    .with_name(
        "stellar_wind_bubble1d.yaml",
    )
)


def load_snapshots(config, max_outputs=10, start_index=1):
    """Load example outputs into lightweight snapshot wrappers."""
    snapshots = []
    for outindex in range(start_index, max_outputs):
        outfilename = Path(config["par"]["output"]["directory"]) / (
            config["par"]["output"]["filename_prefix"] + "_%03d" % outindex + ".hdf5"
        )
        snapshots.append(et.load_output_state(outfilename, config))
    return snapshots


def main(config_filename=DEFAULT_CONFIG, *, plot_only=False):
    et.set_plot_style()
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    example_config = config["example"]
    output_config = config["par"]["output"]
    if not plot_only:
        eu.clean_previous_outputs(config)

    if not plot_only:
        code_units_obj = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
        config["_code_units"] = code_units_obj
        initial_condition = et.build_initial_condition(config)
        initial_condition.write(
            config["par"]["simulation"]["initial_condition_filename"],
            validate=True,
        )
        mainrun = Rsim(config["par"])
        mainrun.RunAll(outputtime=0)

    snapshots = load_snapshots(config)
    figure_prefix = example_config.get("figure_prefix", "StellarWindBubble1D")

    profile_figure = et.make_profile_figure(snapshots, config)
    profile_figure_filename = Path(output_config["directory"]) / f"{figure_prefix}_profiles.jpg"
    profile_figure.savefig(profile_figure_filename, dpi=200)
    plt.close(profile_figure)

    radius_figure = et.make_radius_figure(snapshots, config)
    radius_figure_filename = Path(output_config["directory"]) / f"{figure_prefix}_radius.jpg"
    radius_figure.savefig(radius_figure_filename, dpi=200)
    plt.close(radius_figure)

    velocity_figure = et.make_velocity_figure(snapshots, config)
    if velocity_figure is not None:
        velocity_figure_filename = (
            Path(output_config["directory"]) / f"{figure_prefix}_velocity.jpg"
        )
        velocity_figure.savefig(velocity_figure_filename, dpi=200)
        plt.close(velocity_figure)

    pressure_figure = et.make_pressure_figure(snapshots, config)
    if pressure_figure is not None:
        pressure_figure_filename = (
            Path(output_config["directory"]) / f"{figure_prefix}_pressure.jpg"
        )
        pressure_figure.savefig(pressure_figure_filename, dpi=200)
        plt.close(pressure_figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the spherical stellar-wind bubble example.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Nested YAML configuration.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the hydro run and rebuild the figure from existing outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, plot_only=args.plot_only)
