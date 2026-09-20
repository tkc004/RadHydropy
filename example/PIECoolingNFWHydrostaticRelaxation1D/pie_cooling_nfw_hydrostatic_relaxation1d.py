# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""HM12 PIE relaxation of a hydrostatic atmosphere in a fixed NFW halo."""

import argparse
import sys
from pathlib import Path

import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import example_utils as eu

import radhydropy.io as rio
from example.PIECoolingNFWHydrostaticRelaxation1D import tools as et
from radhydropy.gravity import Gravity, nfw_potential
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits

DEFAULT_CONFIG = EXAMPLE_DIR / "pie_cooling_nfw_hydrostatic_relaxation1d.yaml"


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    config = eu.load_nested_example_config(config_filename)
    par = config["par"]
    initial_mapping = config["initial_condition"]
    thermochemistry = par["thermochemistry"]
    table_filename = str(
        (config_filename.parent / thermochemistry["metal_pie_table_filename"]).resolve(),
    )
    thermochemistry["metal_pie_table_filename"] = table_filename
    eu.clean_previous_outputs(config)
    Path(par["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    code_units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    halo = et.nfw_halo_parameters(
        initial_mapping["halo_mass"],
        initial_mapping["concentration"],
        initial_mapping["redshift"],
        initial_mapping["overdensity"],
        initial_mapping["h0"],
    )
    temperature_virial_unyt = et.virial_temperature(halo, initial_mapping["mu"])
    config["_code_units"] = code_units
    initial = et.build_initial_condition(config)
    initial.write(par["simulation"]["initial_condition_filename"], validate=True)
    sim = rio.loadhdf5(config, par["simulation"]["initial_condition_filename"])
    sim.par.metal_pie_table = MetalPIETable(
        par["thermochemistry"]["metal_pie_table_filename"],
    )
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    nghost = int(par["mesh"]["ghost_cells"])
    interior = slice(nghost, -nghost if nghost else None)
    rho_proper_max = float(np.max(np.asarray(sim.fluid.rho_radarray.value[interior])))
    floor = thermochemistry["cooling_temperature_floor"].to_value(unyt.K)
    runaway_factor = float(thermochemistry.get("runaway_density_factor", 100.0))

    def stop_on_runaway(runner):
        rho_proper_code = np.asarray(runner.fluid.rho_radarray.value[interior])
        temperature_state = np.asarray(runner.fluid.temp_radarray.value[interior])
        runaway = np.max(rho_proper_code) >= runaway_factor * rho_proper_max
        # Do not terminate because a tenuous outer cell reaches the imposed
        # floor.  The relevant runaway is central loss of pressure support.
        ncentral = max(8, int(0.1 * temperature_state.size))
        floor_reached = np.min(temperature_state[:ncentral]) <= 1.01 * floor
        return bool(runaway or floor_reached)

    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.geometry_state.x_proper_code,
            halo["rho_scale_cgs_g_cm3_unyt"],
            halo["radius_scale_proper_kpc_unyt"],
            code_units=sim.par.units.CodeUnits,
        ),
        coordinate=sim.mesh.geometry_state.x_proper_code.copy(),
        code_units=sim.par.units.CodeUnits,
    )
    sim.Run(mode="hydro_sources", stop_condition=stop_on_runaway)
    all_outputs = sorted(
        Path(par["output"]["directory"]).glob(f"{par['output']['filename_prefix']}_*.hdf5"),
    )
    scheduled_times = [
        float(value)
        for value in Path(par["output"]["time_list_filename"]).read_text().splitlines()[1:]
    ]
    outputs = all_outputs[: len(scheduled_times)]
    if len(outputs) < 2:  # noqa: PLR2004
        raise RuntimeError("expected at least two saved snapshots")
    results = [et.analyze_snapshot(name, config, halo, temperature_virial_unyt) for name in outputs]
    for result, scheduled_time in zip(results, scheduled_times, strict=False):
        result["time_proper_Myr"] = scheduled_time
    result_stem = par["simulation"]["name"]
    report = EXAMPLE_DIR / f"{result_stem}_Report.txt"
    figure = EXAMPLE_DIR / f"{result_stem}.jpg"
    et.write_report(results, report, floor)
    et.plot_results(results, halo, figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
