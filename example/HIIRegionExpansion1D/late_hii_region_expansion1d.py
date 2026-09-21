# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Late phase isothermal H II region expansion in spherical 1D.

This example is from STARBENCH: The D-type expansion of an H II region
https://arxiv.org/abs/1507.05621v1

(Late phase of the expansion: note the neutral gas is at 10^3 K, not 10^2 K
as in the early phase example.)

This example follows the hydrodynamic expansion of a central photoionized
region around a source at the origin. The gas is pure hydrogen, spherical,
and evolved with hydrodynamics plus hydrogen photo-chemistry. The neutral and
ionized media are both treated with a simplified isothermal closure:

* neutral gas: ``T = 10^3 K``;
* ionized gas: ``T = 10^4 K``.

The example is YAML-driven, writes HDF5 snapshots, reloads those snapshots,
and plots the ionization-front history and density profiles from the saved outputs.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = Path(tempfile.gettempdir()) / "radhydropy-cache"
mplconfig_dir = Path(tempfile.gettempdir()) / "radhydropy-matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

import unyt

import radhydropy.io as rio
from example import example_utils as eu
from example.HIIRegionExpansion1D import tools as et
from radhydropy.rsim import Rsim

DEFAULT_CONFIG = Path(__file__).resolve().with_name("late_hii_region_expansion1d.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    loaded_config = eu.load_nested_example_config(config_filename)
    par = loaded_config["par"]
    initial_condition = loaded_config["initial_condition"]
    exampleparams = loaded_config["example"]
    config = loaded_config
    output = par["output"]
    eu.clean_previous_outputs(config)

    Path(output["directory"]).mkdir(parents=True, exist_ok=True)
    Path(output["directory"]).mkdir(parents=True, exist_ok=True)

    et.write_initial_condition(config)

    sim = Rsim(config["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    et.apply_piecewise_isothermal_state(sim, config)
    et.print_startup_diagnostics(sim, config, initial_condition)

    output_specs = exampleparams["output_snapshots"]
    step_backend = et.make_logging_step_backend(sim, config, max_logged_steps=5)
    sim.Run(
        outputtime=0,
        mode="hydro_sources",
        step_backend=step_backend,
    )
    outputfilenames = et.output_files(output["directory"], output["filename_prefix"])

    history = et.load_history_from_outputs(outputfilenames, config)

    figure_stem = "LateHIIRegionExpansion1D"
    if par["radiation"].get("temporal_scheme") == "c2ray":
        figure_stem += "_C2Ray"
    figure_filename = Path(output["directory"]) / f"{figure_stem}_IFront.jpg"
    et.save_front_plot(history, config, figure_filename)

    density_figure_filenames = []
    for label, snapshot in et.load_labeled_density_snapshots(
        outputfilenames,
        config,
        output_specs,
    ):
        density_figure_filename = Path(output["directory"]) / (
            f"{figure_stem}_Density_{label}Myr.jpg"
        )
        et.save_density_profile_plot(snapshot, config, density_figure_filename)
        density_figure_filenames.append(density_figure_filename)

    initial_condition["comparison_time"].to_value(unyt.Myr)
    et.front_radius_at_time(
        history,
        initial_condition["comparison_time"],
    ).to_value(unyt.pc)
    et.spitzer_radius(
        initial_condition["comparison_time"],
        config,
    ).to_value(unyt.pc)
    et.hosokawa_inutsuka_radius(
        initial_condition["comparison_time"],
        config,
    ).to_value(unyt.pc)
    et.stagnation_radius(config).to_value(unyt.pc)

    for density_figure_filename in density_figure_filenames:
        pass
    for _outputfilename in outputfilenames:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the late HII region expansion example.",
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
