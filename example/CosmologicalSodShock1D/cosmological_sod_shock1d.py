"""Sod shock tube in an expanding Einstein--de Sitter background."""

import argparse
import copy
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example" / "SodShock1D"))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
from cosmological_initial_condition import (
    build_initial_condition as build_cosmological_initial_condition,
)
from sodshock_analytic import shocktubecal, shocktubeanalyticgraph


DEFAULT_CONFIG = Path(__file__).with_name("cosmological_sod_shock1d.yaml")


class State:
    pass


def _read_profile(filename, code_unit_system):
    par, mesh, fluid = State(), State(), State()
    par.CodeUnits = code_unit_system
    par.code_unit_system = State()
    par.code_unit_system.CodeUnits = code_unit_system
    par.simulation = State()
    par.simulation.coordinate_system = "cartesian"
    par.mesh = State()
    par.mesh.grid_cells = 800
    par.mesh.ghost_cells = 2
    rio.readhdf5(par, mesh, fluid, filename)
    first = int(getattr(par, "noghost", 2))
    count = int(par.mesh.grid_cells)
    return (
        0.5 * np.asarray(
            mesh.boundary_comoving_code[first:first + count + 1], dtype=float
        )[:-1] + 0.5 * np.asarray(
            mesh.boundary_comoving_code[first:first + count + 1], dtype=float
        )[1:],
        np.asarray(fluid.rho_comoving_code[first:first + count], dtype=float),
        np.asarray(fluid.temp_supercomoving_code[first:first + count], dtype=float),
        float(np.sum(np.asarray(fluid.Mass_code[first:first + count], dtype=float))),
        float(np.sum(np.asarray(fluid.Energy_code[first:first + count], dtype=float))),
    )


def run(config_filename=DEFAULT_CONFIG, riemann_solver=None, dual_energy=None):
    config = eu.load_nested_example_config(config_filename)
    case_config = copy.deepcopy(config)
    initial_condition = config["initial_condition"]
    if riemann_solver is not None:
        case_config["par"]["hydrodynamics"]["riemann_solver"] = riemann_solver
    if dual_energy is not None:
        case_config["par"]["hydrodynamics"]["dual_energy"] = dual_energy
    output = case_config["par"]["output"]
    output_dir = Path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(config)
    units = CodeUnits.from_mapping(case_config["par"]["units"]["CodeUnits"])
    gravity = case_config["par"].get("gravity", {})
    if gravity.get("cosmology_type") in ("lambda_cdm", "LambdaCDM", "lcdm"):
        code_cosmology = LambdaCDM.from_code_units(
            units,
            t_ref=float(gravity["cosmology_t_ref"]),
            a_ref=float(gravity["cosmology_a_ref"]),
            omega_m=float(gravity["cosmology_omega_m"]),
            omega_lambda=float(gravity["cosmology_omega_lambda"]),
            hubble_ref=gravity.get("cosmology_hubble_ref"),
        )
    else:
        code_cosmology = EinsteinDeSitter.from_code_units(
            units,
            t_ref=float(gravity.get("cosmology_t_ref", 1.0)),
            a_ref=float(gravity.get("cosmology_a_ref", 1.0)),
        )
    case_config["_code_cosmology"] = code_cosmology
    case_config["_initial_tau_supercomoving_code"] = 0.0
    boxsize_code = float(initial_condition["box_size_comoving"].to_value(units.length_unit))
    case_config["_boundary_start_code"] = -boxsize_code / int(
        case_config["par"]["mesh"]["grid_cells"]
    )
    initial = build_cosmological_initial_condition(case_config)
    ic_filename = output_dir / "InitialCondition.hdf5"
    rio.writehdf5(initial, ic_filename)
    case_config["par"]["simulation"]["initial_condition_filename"] = str(ic_filename)
    sim = Rsim(case_config["par"])
    sim.par.set_cosmology_model(initial.par.cosmology)
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    initial_tau = np.asarray(initial.par.tau_supercomoving_code, dtype=float)
    sim.par.tau_supercomoving_code = initial_tau.copy()
    sim.par.simulation.tau_supercomoving_code = initial_tau.copy()
    sim.fluid.SetFluidTime(initial_tau)
    sim.par.set_cosmology_model(initial.par.cosmology)
    sim.Run(outputtime=0)
    outputs = sorted(output_dir.glob("Output_*.hdf5"))
    # The fixed-cadence callback can stop just before the final target time;
    # write the exact final state so the analytic comparison uses the same
    # time as the simulation.
    sim.fluid.SetTemperature()
    rio.write_numbered_hdf5(sim, len(outputs))
    outputs = sorted(output_dir.glob("Output_*.hdf5"))
    profiles = [_read_profile(filename, units) for filename in outputs]
    initial_mass, initial_energy = profiles[0][3:5]
    final_mass, final_energy = profiles[-1][3:5]
    if not np.isclose(final_mass, initial_mass, rtol=2.0e-10):
        raise RuntimeError("cosmological Sod mass is not conserved")
    if not np.isclose(final_energy, initial_energy, rtol=2.0e-10):
        raise RuntimeError("cosmological Sod supercomoving energy is not conserved")
    if not np.max(profiles[-1][2]) > np.max(profiles[0][2]):
        raise RuntimeError("cosmological Sod shock did not heat the gas")

    radius, density, temperature, _, _ = profiles[-1]
    gamma = float(case_config["par"]["hydrodynamics"]["gamma"])
    pressure_factor = unyt.kb.to_value(unyt.erg / unyt.K) / unyt.mp.to_value(unyt.g)
    pressure_left = float(initial_condition["rho_left_proper"]) * float(
        initial_condition["temperature_left_proper"].to_value("K")
    ) * pressure_factor
    pressure_right = float(initial_condition["rho_right_proper"]) * float(
        initial_condition["temperature_right_proper"].to_value("K")
    ) * pressure_factor
    rho2, rho3, pressure2, velocity2, velocity_tail, velocity_shock, _ = shocktubecal(
        gamma,
        float(initial_condition["rho_right_proper"]),
        float(initial_condition["rho_left_proper"]),
        pressure_right,
        pressure_left,
    )
    final_tau = float(np.asarray(sim.fluid.tau_supercomoving_code, dtype=float))
    print(f"final supercomoving time = {final_tau:.8g}")
    rho_exact, pressure_exact, _ = shocktubeanalyticgraph(
        gamma,
        float(initial_condition["rho_right_proper"]),
        rho2,
        rho3,
        float(initial_condition["rho_left_proper"]),
        pressure_right,
        pressure2,
        pressure_left,
        velocity2,
        velocity_tail,
        velocity_shock,
        final_tau,
        radius,
        0.5 * float(initial_condition["box_size_comoving"].to_value(units.length_unit)),
    )
    interface = 0.5 * float(initial_condition["box_size_comoving"].to_value(units.length_unit))
    central = (radius > interface - 2.0) & (radius < interface + 2.0)
    density_l1 = float(np.mean(np.abs(density[central] - rho_exact[central])))
    if density_l1 > 0.04:
        raise RuntimeError(
            f"cosmological Sod density profile misses exact solution: L1={density_l1:.6g}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for index in np.unique(np.linspace(0, len(profiles) - 1, 5).astype(int)):
        radius, density, temperature, _, _ = profiles[index]
        axes[0].plot(radius, density, label=f"output {index:03d}")
        axes[1].plot(radius, temperature, label=f"output {index:03d}")
    axes[0].set_ylabel("comoving density")
    axes[1].set_ylabel("supercomoving temperature")
    axes[1].set_xlabel("comoving coordinate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    exact_temperature = pressure_exact / np.maximum(rho_exact, 1.0e-30) / pressure_factor
    axes[0].plot(radius[central], rho_exact[central], "k--", lw=1.2, label="exact final")
    axes[1].plot(radius[central], exact_temperature[central], "k--", lw=1.2)
    axes[0].set_xlim(interface - 2.0, interface + 2.0)
    fig.suptitle("Cosmological Sod shock tube")
    fig.tight_layout()
    figure = Path(output["savedir"]) / "CosmologicalSodShock1D.jpg"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"mass relative error = {(final_mass - initial_mass) / initial_mass:.6e}")
    print(f"energy relative error = {(final_energy - initial_energy) / initial_energy:.6e}")
    print(f"final density L1 error = {density_l1:.6e}")
    print(f"scale factor at final time = {sim.par.cosmology.scale_factor_from_supercomoving(float(sim.fluid.tau_supercomoving_code)):.8g}")
    print(f"figure = {figure}")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--riemann-solver", choices=("Rusanov", "HLLC"))
    parser.add_argument("--dual-energy", action="store_true")
    args = parser.parse_args()
    run(args.config, args.riemann_solver, args.dual_energy)
