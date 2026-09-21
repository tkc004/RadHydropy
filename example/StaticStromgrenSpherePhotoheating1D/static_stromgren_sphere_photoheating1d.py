# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Static Stromgren sphere with photoheating.

This repeats the static Stromgren sphere benchmark, but lets the hydrogen
source update heat and cool the gas. Hydrodynamic motion is disabled: density
is fixed and only radiative transfer, chemistry, and thermal source terms are
advanced. The example is configured from YAML, writes HDF5 snapshots, reloads
the final snapshot, and plots from the saved output rather than live state.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
static_stromgren_dir = Path(__file__).resolve().parents[1] / "StaticStromgrenSphere1D"
if str(static_stromgren_dir) not in sys.path:
    sys.path.append(str(static_stromgren_dir))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = Path(tempfile.gettempdir()) / "radhydropy-cache"
mplconfig_dir = Path(tempfile.gettempdir()) / "radhydropy-matplotlib"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

import radhydropy.io as rio
from example import example_utils as eu
from example.StaticStromgrenSpherePhotoheating1D import tools as et
from radhydropy.rsim import Rsim

DEFAULT_CONFIG = (
    Path(__file__)
    .resolve()
    .with_name(
        "static_stromgren_sphere_photoheating1d.yaml",
    )
)


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    nested = eu.load_nested_example_config(config_filename)
    config = nested
    config["initial_condition"]
    example = config.get("example", {})
    eu.clean_previous_outputs(config)
    config_dir = Path(config_filename).resolve().parent
    for key in (
        "temperature_reference_filename",
        "neutral_fraction_reference_filename",
    ):
        if key in example:
            value = Path(example[key])
            if not value.is_absolute():
                example[key] = str(config_dir / value)

    Path(nested["par"]["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    Path(nested["par"]["output"]["directory"]).mkdir(parents=True, exist_ok=True)

    et.write_initial_condition(config)

    mainrun = Rsim(nested["par"])
    rio.readhdf5(
        mainrun.par,
        mainrun.mesh,
        mainrun.fluid,
        mainrun.par.simulation.initial_condition_filename,
    )
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()

    history = mainrun.EvolveStaticThermochemistry(
        nested["par"]["simulation"]["final_time"],
        nested["par"]["timestep"]["evolution_timestep"],
        include_thermal_history=True,
        reference_time=example["reference_time"],
    )
    et.normalize_static_history(history)

    output_filename = (
        Path(nested["par"]["output"]["directory"])
        / f"{nested['par']['output']['filename_prefix']}_000.hdf5"
    )
    rio.writehdf5(mainrun, output_filename)

    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    config["_output_par"] = out_par
    figure_name = "StaticStromgrenSpherePhotoheating1D.jpg"
    if nested["par"]["radiation"].get("radiative_transfer_temporal_scheme") == "c2ray":
        figure_name = "StaticStromgrenSpherePhotoheating1D_C2Ray.jpg"
    figure_filename = Path(nested["par"]["output"]["directory"]) / figure_name
    et.save_plot(out_mesh, out_fluid, history, config, figure_filename)

    if nested["par"]["thermochemistry"].get("hydrogen_alpha_B") is None:
        pass
    else:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the static Stromgren sphere photoheating example.",
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
