# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Hydrostatic gas in a 1e8 Msun NFW dark-matter halo."""

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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

cache_dir = Path(tempfile.gettempdir()) / "radhydropy-cache"
mplconfig_dir = Path(tempfile.gettempdir()) / "radhydropy-matplotlib"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

import example_utils as eu

import radhydropy.io as rio
from example.NFWHydrostaticEquilibrium1D import tools as et
from radhydropy.gravity import Gravity, nfw_potential
from radhydropy.units import CodeUnits

DEFAULT_CONFIG = (
    Path(__file__)
    .resolve()
    .with_name(
        "nfw_hydrostatic_equilibrium1d.yaml",
    )
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config["par"]
    initial_condition = config["initial_condition"]
    eu.clean_previous_outputs(config)
    code_units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    halo = et.nfw_halo_parameters(
        initial_condition["halo_mass"],
        initial_condition["concentration"],
        initial_condition["redshift"],
        initial_condition["overdensity"],
        initial_condition["h0"],
    )
    temperature_proper_unyt = et.virial_temperature(halo, initial_condition["mu"])

    config["_code_units"] = code_units
    initial_state = et.build_initial_condition(config)
    initial_state.write(par["simulation"]["initial_condition_filename"], validate=True)

    sim = rio.loadhdf5(config, par["simulation"]["initial_condition_filename"])
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.geometry_state.x_proper_code,
            halo["rho_scale_cgs_g_cm3_unyt"],
            halo["radius_scale_proper_kpc_unyt"],
            code_units=code_units,
        ),
        coordinate=sim.mesh.geometry_state.x_proper_code.copy(),
        code_units=code_units,
    )
    sim.Run(mode="hydro")

    final_outfile = Path(par["output"]["directory"]) / (
        par["output"]["filename_prefix"] + "_001.hdf5"
    )
    if not os.path.exists(final_outfile):
        raise FileNotFoundError(f"Expected evolved snapshot at {final_outfile}")
    figure_filename = Path(par["output"]["directory"]) / "NFWHydrostaticEquilibrium1D.jpg"
    et.read_and_plot(
        final_outfile,
        config,
        halo,
        temperature_proper_unyt,
        figure_filename,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the NFW hydrostatic-equilibrium example.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
