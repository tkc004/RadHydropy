"""Analytic pressure-supported central-core hydrostatic test."""

import argparse
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from example_utils import load_nested_example_config
from radhydropy.gravity import Gravity, point_mass_potential
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name("hydrostatic_core_spherical1d.yaml")


def run(config_filename=DEFAULT_CONFIG):
    config = load_nested_example_config(config_filename)
    par = config["par"]
    initial_condition = config["initial_condition"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    initial = et.InitialCondition(config, units)
    output_dir = Path(par["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, output_dir / "InitialCondition.hdf5")

    runtime = {**par, "simulation": {**par["simulation"],
        "initial_condition_filename": str(output_dir / "InitialCondition.hdf5")},
        "output": {**par["output"], "directory": str(output_dir),
                   "savedir": str(output_dir)}}
    sim = Rsim(runtime)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=point_mass_potential(
            sim.mesh.coordinate,
            initial_condition["point_mass"],
            code_units=units,
        ),
        coordinate=sim.mesh.coordinate.copy(),
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

    first = int(sim.par.noghost)
    last = first + int(sim.par.nogrid)
    radius = np.asarray(sim.mesh.coordinate[first:last], dtype=float)
    density = np.asarray(sim.fluid.rho_code[first:last], dtype=float)
    analytic = et.analytic_density_code(radius, config, units)
    core_radius = float(np.asarray(sim.par.gas_core_radius))
    halo = radius >= core_radius
    relative_error = np.abs(density - analytic) / np.maximum(analytic, 1.0e-300)
    core_cells = radius < core_radius
    core_last = np.flatnonzero(core_cells)[-1]
    pressure_mismatch = abs(
        float(sim.fluid.pre_code[first + core_last])
        - float(sim.fluid.pre_code[first + core_last + 1])
    ) / max(float(sim.fluid.pre_code[first + core_last + 1]), 1.0e-300)
    max_halo_error = float(np.max(relative_error[halo]))
    mean_step = float(np.mean([item[0] for item in step_times]))
    print("maximum halo density relative error: %.6e" % max_halo_error)
    print("core/halo pressure relative mismatch: %.6e" % pressure_mismatch)
    print("hydro steps: %d" % len(step_times))
    print("mean timestep: %.6e" % mean_step)

    figure = output_dir / "HydrostaticCoreSpherical1D.jpg"
    radius_pc = radius * float(units.length_in_cgs) / 3.085677581e18
    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(radius_pc, density, label="simulation")
    plt.loglog(radius_pc, analytic, "--", label="analytic")
    plt.axvline(core_radius * float(units.length_in_cgs) / 3.085677581e18,
                color="0.4", ls=":", label="core radius")
    plt.xlabel("radius [pc]")
    plt.ylabel("density [code units]")
    plt.title("Pressure-supported core: spherical point-mass atmosphere")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=200)
    plt.close()
    print("figure = %s" % figure)
    return max_halo_error, pressure_mismatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
