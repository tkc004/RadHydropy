# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Analytic pressure-supported central-core hydrostatic test."""  # noqa: CPY001

import argparse
import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from example_utils import load_nested_example_config  # noqa: E402

import radhydropy.io as rio  # noqa: E402
import tools as et  # noqa: E402
from radhydropy.gravity import Gravity, point_mass_potential  # noqa: E402
from radhydropy.units import CodeUnits, quantity_to_value  # noqa: E402

DEFAULT_CONFIG = Path(__file__).with_name("hydrostatic_core_spherical1d.yaml")


def run(config_filename=DEFAULT_CONFIG):
    config = load_nested_example_config(config_filename)
    initial_condition = config["initial_condition"]
    units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    initial = et.build_initial_condition(config)
    output_dir = Path(config["par"]["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    initial.write(output_dir / "InitialCondition.hdf5", validate=True)

    config["par"]["simulation"] = {
        **config["par"]["simulation"],
        "initial_condition_filename": str(output_dir / "InitialCondition.hdf5"),
    }
    config["par"]["output"] = {
        **config["par"]["output"],
        "directory": str(output_dir),
    }
    sim = rio.loadhdf5(config, config["par"]["simulation"]["initial_condition_filename"])
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=point_mass_potential(
            sim.mesh.geometry_state.x_proper_code,
            initial_condition["point_mass"],
            code_units=units,
        ),
        coordinate=sim.mesh.geometry_state.x_proper_code.copy(),
        code_units=units,
    )

    step_times = []
    original_step = sim.Step

    def step_backend(**kwargs):
        start = time.perf_counter()
        result = original_step(**kwargs)
        step_times.append((float(result["dt"]), time.perf_counter() - start))
        return result

    sim.Run(mode="hydro", step_backend=step_backend)

    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    radius_proper_code = np.asarray(sim.mesh.geometry_state.x_proper_code[first:last], dtype=float)
    rho_values = np.asarray(sim.fluid.rho_radarray.value, dtype=float)
    temp_values = np.asarray(sim.fluid.temp_radarray.value, dtype=float)
    mu_values = np.asarray(sim.fluid.mu, dtype=float)
    if mu_values.size != int(config["par"]["mesh"]["grid_cells"]):
        mu_values = mu_values[first:last]
    rho_proper_code = (
        rho_values[first:last]
        if rho_values.size != int(config["par"]["mesh"]["grid_cells"])
        else rho_values
    )
    pressure_proper_code = np.asarray(
        sim.fluid.eos.pressure(rho_values, temp_values, mu_values),
        dtype=float,
    )
    analytic_rho_proper_code = et.analytic_density_code(radius_proper_code, config)
    core_radius_proper_code = quantity_to_value(
        config["par"]["gravity"]["radius_core_proper"],
        units.length_unit,
    )
    halo = radius_proper_code >= core_radius_proper_code
    relative_error = np.abs(rho_proper_code - analytic_rho_proper_code) / np.maximum(
        analytic_rho_proper_code,
        1.0e-300,
    )
    core_cells = radius_proper_code < core_radius_proper_code
    core_last = np.flatnonzero(core_cells)[-1]
    pressure_mismatch = abs(
        float(pressure_proper_code[core_last]) - float(pressure_proper_code[core_last + 1]),
    ) / max(float(pressure_proper_code[core_last + 1]), 1.0e-300)
    max_halo_error = float(np.max(relative_error[halo]))
    float(np.mean([item[0] for item in step_times]))

    figure = output_dir / "HydrostaticCoreSpherical1D.jpg"
    radius_proper_pc = radius_proper_code * float(units.length_in_cgs) / 3.085677581e18
    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(radius_proper_pc, rho_proper_code, label="simulation")
    plt.loglog(radius_proper_pc, analytic_rho_proper_code, "--", label="analytic")
    plt.axvline(
        core_radius_proper_code * float(units.length_in_cgs) / 3.085677581e18,
        color="0.4",
        ls=":",
        label="core radius",
    )
    plt.xlabel("radius [pc]")
    plt.ylabel("density [code units]")
    plt.title("Pressure-supported core: spherical point-mass atmosphere")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=200)
    plt.close()
    return max_halo_error, pressure_mismatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
