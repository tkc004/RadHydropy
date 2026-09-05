"""Sod shock tube in an expanding Einstein--de Sitter background."""

import argparse
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
from sodshock_analytic import shocktubecal, shocktubeanalyticgraph


DEFAULT_CONFIG = Path(__file__).with_name("cosmological_sod_shock1d.yaml")


class State:
    pass


def make_initial_condition(ic, units, par):
    state = State()
    state.par = State()
    state.mesh = State()
    state.fluid = State()
    state.par.CodeUnits = units
    state.par.units = State()
    state.par.units.CodeUnits = units
    state.par.unit_system = units.unit_system
    state.par.nogrid = int(par["mesh"]["grid_cells"])
    state.par.mesh = State()
    state.par.mesh.grid_cells = int(par["mesh"]["grid_cells"])
    state.par.mesh.ghost_cells = 0
    state.par.hydrodynamics = State()
    state.par.hydrodynamics.gamma = float(par["hydrodynamics"]["gamma"])
    state.par.simulation = State()
    state.par.simulation.current_time = np.asarray([0.0]) * units.time_unit
    state.par.simulation.box_size = np.asarray([float(ic["boxsize"].to_value(units.length_unit))]) * units.length_unit
    state.par.simulation.coordinate_system = "cartesian"
    state.par.coordsys = "cartesian"
    boxsize = float(ic["boxsize"].to_value(units.length_unit))
    state.par.boxsize = np.asarray([boxsize]) * units.length_unit
    state.par.time = np.asarray([0.0]) * units.time_unit
    state.par.cosmological_expansion = True
    state.par.supercomoving_coordinates = True
    state.par.coordinate_frame = "comoving"
    state.par.time_coordinate = "supercomoving"
    state.par.density_representation = "comoving"
    state.par.temperature_representation = "supercomoving"
    state.par.velocity_representation = "supercomoving_peculiar"
    gravity = par.get("gravity", {})
    if gravity.get("cosmology_type") in ("lambda_cdm", "LambdaCDM", "lcdm"):
        state.par.cosmology = LambdaCDM.from_code_units(
            units,
            t_ref=float(gravity["cosmology_t_ref"]),
            a_ref=float(gravity["cosmology_a_ref"]),
            omega_m=float(gravity["cosmology_omega_m"]),
            omega_lambda=float(gravity["cosmology_omega_lambda"]),
            hubble_ref=gravity.get("cosmology_hubble_ref"),
        )
    else:
        state.par.cosmology = EinsteinDeSitter.from_code_units(
            units, t_ref=1.0, a_ref=1.0
        )
    dx = boxsize / state.par.nogrid
    boundary = np.linspace(-dx, boxsize + dx, state.par.nogrid + 1)
    coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    state.mesh.boundary = boundary * units.length_unit
    state.mesh.coordinate = coordinate * units.length_unit
    state.mesh.xdelta = np.full(state.par.nogrid, dx) * units.length_unit
    state.mesh.area = np.ones(state.par.nogrid + 0) * units.area_unit
    state.mesh.vol = np.full(state.par.nogrid, dx) * units.volume_unit
    left = coordinate < 0.5 * boxsize
    state.fluid.rho_code = np.where(left, float(ic["rho_left"]), float(ic["rho_right"]))
    state.fluid.temp_code = np.where(
        left,
        float(ic["temp_left"].to_value("K")),
        float(ic["temp_right"].to_value("K")),
    )
    state.fluid.vel_code = np.zeros(state.par.nogrid)
    state.fluid.mu = np.full(state.par.nogrid, float(ic["mu"]))
    return state


def _read_profile(filename, units):
    par, mesh, fluid = State(), State(), State()
    par.CodeUnits = units
    par.units = State()
    par.units.CodeUnits = units
    par.simulation = State()
    par.simulation.coordinate_system = "cartesian"
    par.mesh = State()
    par.mesh.grid_cells = 800
    par.mesh.ghost_cells = 2
    rio.readhdf5(par, mesh, fluid, filename)
    first = int(getattr(par, "noghost", 2))
    count = int(par.nogrid)
    return (
        0.5 * np.asarray(
            mesh.boundary[first:first + count + 1], dtype=float
        )[:-1] + 0.5 * np.asarray(
            mesh.boundary[first:first + count + 1], dtype=float
        )[1:],
        np.asarray(fluid.rho_code[first:first + count], dtype=float),
        np.asarray(fluid.temp_code[first:first + count], dtype=float),
        float(np.sum(np.asarray(fluid.Mass_code[first:first + count], dtype=float))),
        float(np.sum(np.asarray(fluid.Energy_code[first:first + count], dtype=float))),
    )


def run(config_filename=DEFAULT_CONFIG, riemann_solver=None, dual_energy=None):
    config = eu.load_nested_example_config(config_filename)
    runtime = config["par"]
    icparams = config["initial_condition"]
    if riemann_solver is not None:
        runtime = {**runtime, "hydrodynamics": {**runtime["hydrodynamics"], "riemann_solver": riemann_solver}}
    if dual_energy is not None:
        runtime = {**runtime, "hydrodynamics": {**runtime["hydrodynamics"], "dual_energy": dual_energy}}
    output = runtime["output"]
    output_dir = Path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(output)
    units = CodeUnits.from_mapping(runtime["units"]["CodeUnits"])
    initial = make_initial_condition(icparams, units, runtime)
    ic_filename = output_dir / "InitialCondition.hdf5"
    rio.writehdf5(initial, ic_filename)
    runtime = {key: (dict(value) if isinstance(value, dict) else value)
               for key, value in runtime.items()}
    runtime["simulation"] = {**runtime["simulation"], "initial_condition_filename": str(ic_filename)}
    sim = Rsim(runtime)
    sim.par.set_cosmology_model(initial.par.cosmology)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
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
    gamma = float(runtime["hydrodynamics"]["gamma"])
    pressure_factor = unyt.kb.to_value(unyt.erg / unyt.K) / unyt.mp.to_value(unyt.g)
    pressure_left = float(icparams["rho_left"]) * float(
        icparams["temp_left"].to_value("K")
    ) * pressure_factor
    pressure_right = float(icparams["rho_right"]) * float(
        icparams["temp_right"].to_value("K")
    ) * pressure_factor
    rho2, rho3, pressure2, velocity2, velocity_tail, velocity_shock, _ = shocktubecal(
        gamma,
        float(icparams["rho_right"]),
        float(icparams["rho_left"]),
        pressure_right,
        pressure_left,
    )
    final_tau = float(np.asarray(sim.fluid.time_code, dtype=float))
    print(f"final supercomoving time = {final_tau:.8g}")
    rho_exact, pressure_exact, _ = shocktubeanalyticgraph(
        gamma,
        float(icparams["rho_right"]),
        rho2,
        rho3,
        float(icparams["rho_left"]),
        pressure_right,
        pressure2,
        pressure_left,
        velocity2,
        velocity_tail,
        velocity_shock,
        final_tau,
        radius,
        0.5 * float(icparams["boxsize"].to_value(units.length_unit)),
    )
    interface = 0.5 * float(icparams["boxsize"].to_value(units.length_unit))
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
    print(f"scale factor at final time = {sim.par.cosmology.scale_factor_from_supercomoving(float(sim.fluid.time_code)):.8g}")
    print(f"figure = {figure}")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--riemann-solver", choices=("Rusanov", "HLLC"))
    parser.add_argument("--dual-energy", action="store_true")
    args = parser.parse_args()
    run(args.config, args.riemann_solver, args.dual_energy)
