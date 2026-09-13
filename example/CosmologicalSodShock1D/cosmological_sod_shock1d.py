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
from radhydropy.cosmology_context import CosmologyContext
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu
from tools import shocktubecal, shocktubeanalyticgraph


DEFAULT_CONFIG = Path(__file__).with_name("cosmological_sod_shock1d.yaml")


class _RuntimeCosmology:
    """Expose the live cosmology while retaining the serialized model API."""

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)


def _read_profile(filename, config):
    """Read one cosmological snapshot through the canonical typed runtime."""
    sim = rio.loadhdf5(config, filename)
    first = int(sim.par.mesh.ghost_cells)
    count = int(sim.par.mesh.grid_cells)
    boundary_comoving_radarray = sim.mesh.boundary_radarray
    rho_comoving_radarray = sim.fluid.rho_radarray
    temp_supercomoving_radarray = sim.fluid.temp_radarray
    boundary_comoving_code = np.asarray(
        boundary_comoving_radarray.to(config["_code_units"].length_unit).value,
        dtype=float,
    )
    rho_comoving_code = np.asarray(
        rho_comoving_radarray.to(config["_code_units"].density_unit).value,
        dtype=float,
    )
    temp_supercomoving_code = np.asarray(
        temp_supercomoving_radarray.to(
            config["_code_units"].temperature_unit
        ).value,
        dtype=float,
    )
    return (
        0.5 * boundary_comoving_code[first:first + count + 1][:-1]
        + 0.5 * boundary_comoving_code[first:first + count + 1][1:],
        rho_comoving_code[first:first + count],
        temp_supercomoving_code[first:first + count],
        float(np.sum(np.asarray(sim.fluid.Mass_code[first:first + count], dtype=float))),
        float(np.sum(np.asarray(sim.fluid.Energy_code[first:first + count], dtype=float))),
    )


def _build_initial_condition(config, units):
    """Build the Sod IC through the representation-aware writer boundary."""
    par_config = config["par"]
    initial_condition = config["initial_condition"]
    grid_cells = int(par_config["mesh"]["grid_cells"])
    box_size_comoving_unyt = initial_condition["box_size_comoving"]
    boundary_comoving_unyt = np.linspace(
        0.0,
        float(box_size_comoving_unyt.to_value(units.length_unit)),
        grid_cells + 1,
    ) * units.length_unit
    cell_centers_comoving_unyt = 0.5 * (
        boundary_comoving_unyt[:-1] + boundary_comoving_unyt[1:]
    )
    left = cell_centers_comoving_unyt < 0.5 * box_size_comoving_unyt

    writer = InitialConditionWriter(
        ic_config=config["initial_condition"],
        par_config=par_config,
        code_units=units,
        cosmology_context=CosmologyContext(
            gamma=float(par_config["hydrodynamics"]["gamma"]),
            cosmology=config["_code_cosmology"].type_name,
            scale_factor=1.0,
            hubble_parameter_km_s_Mpc=0.0,
        ),
    )
    writer.box_size = writer.radquantity(box_size_comoving_unyt)
    writer.mesh.boundary_radarray = writer.radarray(
        boundary_comoving_unyt, representation="comoving"
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.where(
            left,
            initial_condition["rho_left_proper"],
            initial_condition["rho_right_proper"],
        ),
        representation="proper",
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(grid_cells) * units.velocity_unit,
        representation="proper",
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.where(
            left,
            initial_condition["temperature_left_proper"],
            initial_condition["temperature_right_proper"],
        ),
        representation="proper",
    )
    writer.simulation.par.tau_supercomoving_code = np.array([0.0])
    writer.simulation.par.simulation.tau_supercomoving_code = np.array([0.0])
    writer.simulation.fluid.mu = np.full(
        grid_cells, float(initial_condition["mu"])
    )
    return writer


def _analytic_solution(config, units, radius_comoving_code, final_tau):
    """Evaluate the analytic Sod solution in the simulation code units."""
    initial_condition = config["initial_condition"]
    gamma = float(config["par"]["hydrodynamics"]["gamma"])
    pressure_factor = unyt.kb.to_value(unyt.erg / unyt.K) / unyt.mp.to_value(unyt.g)
    rho_left_proper_code = float(
        initial_condition["rho_left_proper"].to_value(units.density_unit)
    )
    rho_right_proper_code = float(
        initial_condition["rho_right_proper"].to_value(units.density_unit)
    )
    pressure_left = rho_left_proper_code * float(
        initial_condition["temperature_left_proper"].to_value("K")
    ) * pressure_factor
    pressure_right = rho_right_proper_code * float(
        initial_condition["temperature_right_proper"].to_value("K")
    ) * pressure_factor
    rho2, rho3, pressure2, velocity2, velocity_tail, velocity_shock, _ = shocktubecal(
        gamma,
        rho_right_proper_code,
        rho_left_proper_code,
        pressure_right,
        pressure_left,
    )
    rho_exact, pressure_exact, _ = shocktubeanalyticgraph(
        gamma,
        rho_right_proper_code,
        rho2,
        rho3,
        rho_left_proper_code,
        pressure_right,
        pressure2,
        pressure_left,
        velocity2,
        velocity_tail,
        velocity_shock,
        final_tau,
        radius_comoving_code,
        0.5 * float(
            initial_condition["box_size_comoving"].to_value(units.length_unit)
        ),
    )
    interface = 0.5 * float(
        initial_condition["box_size_comoving"].to_value(units.length_unit)
    )
    return rho_exact, pressure_exact, pressure_factor, interface


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
    cosmology_config = case_config["par"].get("cosmology", {})
    if cosmology_config.get("cosmology_type") == "lambda_cdm":
        code_cosmology = LambdaCDM.from_code_units(
            units,
            t_ref=quantity_to_value(cosmology_config["cosmology_t_ref"], units.time_unit),
            a_ref=float(cosmology_config["cosmology_a_ref"]),
            omega_m=float(cosmology_config["cosmology_omega_m"]),
            omega_lambda=float(cosmology_config["cosmology_omega_lambda"]),
            hubble_ref=cosmology_config.get("cosmology_hubble_ref"),
        )
    else:
        code_cosmology = EinsteinDeSitter.from_code_units(
            units,
            t_ref=quantity_to_value(cosmology_config.get("cosmology_t_ref", {"value": 1.0, "unit": "s"}), units.time_unit),
            a_ref=float(cosmology_config.get("cosmology_a_ref", 1.0)),
        )
    case_config["_code_units"] = units
    case_config["_code_cosmology"] = code_cosmology
    ic_filename = output_dir / "InitialCondition.hdf5"
    writer = _build_initial_condition(case_config, units)
    writer.write(ic_filename)
    initial = writer.simulation
    case_config["par"]["simulation"]["initial_condition_filename"] = str(ic_filename)
    sim = rio.loadhdf5(case_config, ic_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    initial_tau = np.asarray(initial.par.tau_supercomoving_code, dtype=float)
    sim.par.tau_supercomoving_code = initial_tau.copy()
    sim.par.simulation.tau_supercomoving_code = initial_tau.copy()
    sim.fluid.SetFluidTime(initial_tau)
    if not (
        np.allclose(sim.par.tau_supercomoving_code, initial_tau)
        and np.allclose(sim.par.simulation.tau_supercomoving_code, initial_tau)
        and np.allclose(np.asarray(sim.fluid.tau_supercomoving_code, dtype=float), initial_tau)
    ):
        raise RuntimeError("cosmological startup clocks disagree after SetInitFluid")
    sim.par.cosmology = _RuntimeCosmology(code_cosmology)
    sim.Run(outputtime=0)
    outputs = sorted(output_dir.glob("Output_*.hdf5"))
    # The fixed-cadence callback can stop just before the final target time;
    # write the exact final state so the analytic comparison uses the same
    # time as the simulation.
    sim.fluid.SetTemperature()
    rio.write_numbered_hdf5(sim, len(outputs))
    outputs = sorted(output_dir.glob("Output_*.hdf5"))
    profiles = [_read_profile(filename, case_config) for filename in outputs]
    initial_mass, initial_energy = profiles[0][3:5]
    final_mass, final_energy = profiles[-1][3:5]
    if not np.isclose(final_mass, initial_mass, rtol=2.0e-10):
        raise RuntimeError("cosmological Sod mass is not conserved")
    if not np.isclose(final_energy, initial_energy, rtol=2.0e-10):
        raise RuntimeError("cosmological Sod supercomoving energy is not conserved")
    if not np.max(profiles[-1][2]) > np.max(profiles[0][2]):
        raise RuntimeError("cosmological Sod shock did not heat the gas")

    radius_comoving_code, rho_comoving_code, temp_supercomoving_code, _, _ = profiles[-1]
    final_tau = float(np.asarray(sim.fluid.tau_supercomoving_code, dtype=float))
    print(f"final supercomoving time = {final_tau:.8g}")
    rho_exact, pressure_exact, pressure_factor, interface = _analytic_solution(
        case_config,
        units,
        radius_comoving_code,
        final_tau,
    )
    central = (radius_comoving_code > interface - 2.0) & (radius_comoving_code < interface + 2.0)
    density_l1 = float(np.mean(np.abs(rho_comoving_code[central] - rho_exact[central])))
    if density_l1 > 0.04:
        raise RuntimeError(
            f"cosmological Sod density profile misses exact solution: L1={density_l1:.6g}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for index in np.unique(np.linspace(0, len(profiles) - 1, 5).astype(int)):
        radius_comoving_code, rho_comoving_code, temp_supercomoving_code, _, _ = profiles[index]
        axes[0].plot(radius_comoving_code, rho_comoving_code, label=f"output {index:03d}")
        axes[1].plot(radius_comoving_code, temp_supercomoving_code, label=f"output {index:03d}")
    axes[0].set_ylabel("comoving density")
    axes[1].set_ylabel("supercomoving temperature")
    axes[1].set_xlabel("comoving coordinate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    exact_temperature = pressure_exact / np.maximum(rho_exact, 1.0e-30) / pressure_factor
    axes[0].plot(radius_comoving_code[central], rho_exact[central], "k--", lw=1.2, label="exact final")
    axes[1].plot(radius_comoving_code[central], exact_temperature[central], "k--", lw=1.2)
    axes[0].set_xlim(interface - 2.0, interface + 2.0)
    fig.suptitle("Cosmological Sod shock tube")
    fig.tight_layout()
    figure = Path(output["directory"]) / "CosmologicalSodShock1D.jpg"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"mass relative error = {(final_mass - initial_mass) / initial_mass:.6e}")
    print(f"energy relative error = {(final_energy - initial_energy) / initial_energy:.6e}")
    print(f"final density L1 error = {density_l1:.6e}")
    print(f"scale factor at final time = {sim.par.cosmology.model.scale_factor_from_supercomoving(float(sim.fluid.tau_supercomoving_code)):.8g}")
    print(f"figure = {figure}")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--riemann-solver", choices=("Rusanov", "HLLC"))
    parser.add_argument("--dual-energy", action="store_true")
    args = parser.parse_args()
    run(args.config, args.riemann_solver, args.dual_energy)
