# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Static Stromgren sphere at constant temperature.

This benchmark keeps the gas density and temperature fixed. A central source
emits ionizing photons at a constant rate, the long-characteristic
radiative-transfer update supplies ``n_gamma``, and the hydrogen neutral
fraction is advanced with the implicit chemistry solver. Hydrodynamics,
heating, and cooling are disabled.
"""  # noqa: CPY001

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

cache_dir = os.path.join(tempfile.gettempdir(), "radhydropy-cache")
mplconfig_dir = os.path.join(tempfile.gettempdir(), "radhydropy-matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
os.environ.setdefault("MPLCONFIGDIR", mplconfig_dir)

import example_utils as eu  # noqa: E402

import radhydropy.io as rio  # noqa: E402
import tools as et  # noqa: E402
from radhydropy.rsim import Rsim  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().with_name("static_stromgren_sphere1d.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    nested = eu.load_nested_example_config(config_filename)
    config = nested
    config["initial_condition"]
    config.get("example", {})
    eu.clean_previous_outputs(config)
    Path(nested["par"]["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    Path(nested["par"]["output"]["directory"]).mkdir(parents=True, exist_ok=True)

    et.write_initial_condition(config)

    sim = Rsim(nested["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    front_history = sim.EvolveStaticThermochemistry(
        nested["par"]["simulation"]["final_time"],
        nested["par"]["timestep"]["chemistry_timestep"],
    )
    front_history = dict(front_history)
    front_history["time_proper_Myr"] = front_history.pop("time_Myr")
    front_history["front_radius_proper_kpc"] = front_history.pop("front_radius_kpc")

    output_filename = (
        Path(nested["par"]["output"]["directory"])
        / f"{nested['par']['output']['filename_prefix']}_000.hdf5"
    )
    rio.writehdf5(sim, output_filename)

    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    config["_output_par"] = out_par
    figure_filename = Path(nested["par"]["output"]["directory"]) / "StaticStromgrenSphere1D.jpg"
    front_figure_filename = (
        Path(nested["par"]["output"]["directory"]) / "StaticStromgrenSphere1D_IFront.jpg"
    )
    budget_figure_filename = (
        Path(nested["par"]["output"]["directory"]) / "StaticStromgrenSphere1D_PhotonBudget.jpg"
    )

    et.save_plot(out_mesh, out_fluid, config, figure_filename)
    et.save_front_history_plot(front_history, config, front_figure_filename)
    et.save_photon_budget_plot(
        front_history,
        budget_figure_filename,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the static Stromgren sphere example.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML file containing nested runtime and initial-condition settings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
