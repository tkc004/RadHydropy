# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Adiabatic accretion shock benchmark for a 1e12 Msun NFW halo."""

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
import unyt

import radhydropy.io as rio
from example.NFWVirialShockAdiabatic1D import tools as et
from radhydropy.gravity import Gravity, nfw_potential
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits

DEFAULT_CONFIG = (
    Path(__file__)
    .resolve()
    .with_name(
        "nfw_virial_shock_adiabatic1d.yaml",
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

    config["_code_units"] = code_units
    initial_state = et.build_initial_condition(config)
    initial_state.write(par["simulation"]["initial_condition_filename"], validate=True)

    sim = Rsim(config["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
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

    output_files = [
        Path(par["output"]["directory"]) / name
        for name in sorted(os.listdir(par["output"]["directory"]))
        if name.startswith(par["output"]["filename_prefix"] + "_") and name.endswith(".hdf5")
    ]
    figure_filename = Path(par["output"]["directory"]) / "NFWVirialShockAdiabatic1D.jpg"
    rows = et.rankine_hugoniot_diagnostics(
        output_files,
        config,
        halo,
    )
    report_filename = (
        Path(par["output"]["directory"]) / "NFWVirialShockAdiabatic1D_RankineHugoniot.txt"
    )
    et.plot_snapshots(output_files, config, halo, figure_filename)
    et.write_rankine_hugoniot_report(rows, report_filename)

    halo["radius_virial_proper_kpc_unyt"].to_value(unyt.kpc)
    halo["vel_virial_proper_km_s_unyt"].to_value(unyt.km / unyt.s)
    et.virial_temperature(
        halo,
        initial_condition["mu"],
    ).to_value(unyt.K)
    for _row in rows:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the adiabatic NFW virial-shock benchmark.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
