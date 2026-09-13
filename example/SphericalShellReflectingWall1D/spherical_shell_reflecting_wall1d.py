"""Cold spherical shell falling onto a finite reflecting wall.

This is deliberately a short, order-zero HLLC validation problem.  Unlike a
spherical mesh that touches ``r=0``, the inner face has finite area, so the
reflected Riemann problem is retained and can be diagnosed directly.
"""

from pathlib import Path
import argparse
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

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu

DEFAULT_CONFIG = Path(__file__).with_name("spherical_shell_reflecting_wall1d.yaml")


class InnerWallSolver(Solver):
    """Reflect at the finite inner face and copy the state at the outer face."""

    def SetBoundary(self, mesh, fluid, par):
        self.ApplyHydrostaticCore(mesh, fluid, par)
        first = int(par.mesh.ghost_cells)
        last = first + int(par.mesh.grid_cells) - 1
        ng = int(par.mesh.ghost_cells)
        left = self._boundary_state(
            fluid, slice(first, first + ng), negate_velocity=True, reverse=True
        )
        self._copy_boundary_state(fluid, slice(0, ng), left)
        right = self._boundary_state(fluid, last)
        self._copy_boundary_state(fluid, slice(last + 1, last + 1 + ng), right)


def make_initial_condition(config):
    ic = config["initial_condition"]
    code_unit_system = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    grid_cells = int(ic["grid_cells"])
    writer = InitialConditionWriter(
        par_config=config["par"], code_units=code_unit_system,
        ic_config=ic,
    )
    writer.simulation.par.simulation.coordinate_system = ic["coordinate_system"]
    radius_inner_proper_code = quantity_to_value(
        ic["radius_inner_proper"], code_unit_system.length_unit
    )
    radius_outer_proper_code = quantity_to_value(
        ic["radius_outer_proper"], code_unit_system.length_unit
    )
    boundary_proper_code = np.linspace(
        radius_inner_proper_code, radius_outer_proper_code, grid_cells + 1
    )
    radius_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    shell_inner_proper_code = quantity_to_value(
        ic["shell_inner"], code_unit_system.length_unit
    )
    shell_outer_proper_code = quantity_to_value(
        ic["shell_outer"], code_unit_system.length_unit
    )
    shell = (radius_proper_code >= shell_inner_proper_code) & (
        radius_proper_code <= shell_outer_proper_code
    )
    density_shell_proper_unyt = ic["rho_shell_proper"]
    temperature_proper_unyt = np.where(
        shell, ic["temperature_proper"], 0.0 * unyt.K
    )
    velocity_proper_unyt = np.where(
        shell, ic["vel_proper"], 0.0 * ic["vel_proper"].units
    )
    writer.box_size = writer.radquantity(ic["radius_outer_proper"])
    writer.mesh.boundary_radarray = writer.radarray(
        boundary_proper_code * code_unit_system.length_unit
    )
    writer.mesh.x_radarray = writer.radarray(
        radius_proper_code * code_unit_system.length_unit
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.where(shell, 1.0, 0.0) * density_shell_proper_unyt
    )
    writer.fluid.temp_radarray = writer.radarray(temperature_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(velocity_proper_unyt)
    writer.simulation.fluid.mu = np.full(
        grid_cells, float(ic["mean_molecular_weight"])
    )
    return writer


def _profile(sim):
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    r = np.asarray(sim.mesh.x_proper_code[first:last], dtype=float)
    rho_proper_code = np.asarray(sim.fluid.rho_proper_code[first:last], dtype=float)
    vel_proper_code = np.asarray(sim.fluid.vel_proper_code[first:last], dtype=float)
    pre_proper_code = np.asarray(sim.fluid.pre_proper_code[first:last], dtype=float)
    temp_proper_code = np.asarray(sim.fluid.temp_proper_code[first:last], dtype=float)
    entropy = np.full_like(pre_proper_code, np.nan)
    active = rho_proper_code > 0.0
    entropy[active] = (
        pre_proper_code[active]
        / rho_proper_code[active] ** float(sim.par.hydrodynamics.gamma)
    )
    return (
        r,
        rho_proper_code,
        vel_proper_code,
        pre_proper_code,
        temp_proper_code,
        entropy,
    )


def run(config_filename=DEFAULT_CONFIG, riemann_solver=None):
    config = eu.load_nested_example_config(config_filename)
    initial_condition = config['initial_condition']
    if riemann_solver is not None:
        config["par"]["hydrodynamics"]["riemann_solver"] = riemann_solver
        config["par"]["output"]["directory"] = Path(config["par"]["output"]["directory"]).with_name(
            Path(config["par"]["output"]["directory"]).name + "_" + riemann_solver
        )
        config["par"]["output"]["directory"] = config["par"]["output"]["directory"]
    output_directory = Path(config["par"]["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    initial = make_initial_condition(config)
    initial.write(
        config["par"]["simulation"]["initial_condition_filename"], validate=False
    )
    sim = Rsim(config["par"])
    sim.solver = InnerWallSolver()
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    first = int(sim.par.mesh.ghost_cells)
    wall_face = first
    snapshots, fluxes = [], []
    code_unit_system = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    target_proper_code = quantity_to_value(
        config["par"]["simulation"]["final_time"], code_unit_system.time_unit
    )
    output_cadence_proper_code = quantity_to_value(
        config["par"]["output"]["cadence"], code_unit_system.time_unit
    )
    next_output = 0.0
    while float(sim.fluid.time_proper_code) < target_proper_code:
        sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
        sim.solver.SetConserved(sim.mesh, sim.fluid)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, sim.par)
        dt = sim.GetStepTime(final_time=target_proper_code)
        sim.solver.SetInterFaceFlux(
            sim.mesh, sim.fluid, config["par"]['boundary']['condition'],
            method=config["par"]["hydrodynamics"]["riemann_solver"], order=0,
        )
        fluxes.append([float(sim.fluid.time_proper_code), float(sim.fluid.Mass_code.flux[wall_face]), float(sim.fluid.Mom_code.flux[wall_face]), float(sim.fluid.Energy_code.flux[wall_face])])
        sim.Step(dt=dt, mode="hydro")
        time_proper_code = float(sim.fluid.time_proper_code)
        if time_proper_code >= next_output - 1.0e-12:
            snapshots.append((time_proper_code,) + _profile(sim))
            next_output += output_cadence_proper_code

    snapshots.append((float(sim.fluid.time_proper_code),) + _profile(sim))
    final = snapshots[-1]
    radius_proper_code, rho_proper_code, vel_proper_code, pre_proper_code, temp_proper_code, entropy_dimensionless = final[1:]
    active = rho_proper_code > float(config["par"]["hydrodynamics"].get("cfl_density_floor", 0.0))
    hot = active & (
        temp_proper_code > 10.0 * quantity_to_value(
            initial_condition["temperature_proper"], code_unit_system.temperature_unit
        )
    )
    if not np.any(hot):
        raise RuntimeError("finite reflecting wall did not produce post-shock heating")
    data = {"time_proper_code": np.array([s[0] for s in snapshots]), "radius_proper_code": radius_proper_code, "rho_proper_code": np.array([s[2] for s in snapshots]), "vel_proper_code": np.array([s[3] for s in snapshots]), "pre_proper_code": np.array([s[4] for s in snapshots]), "temp_proper_code": np.array([s[5] for s in snapshots]), "entropy_dimensionless": np.array([s[6] for s in snapshots]), "flux_proper_code": np.asarray(fluxes)}
    np.savez(output_directory / "SphericalShellReflectingWall1D_diagnostics.npz", **data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    radius_proper_kpc = radius_proper_code / 3.085677581e21
    for i in np.unique(np.linspace(0, len(snapshots) - 1, min(5, len(snapshots))).astype(int)):
        label = f"{snapshots[i][0] / 1.0e6 / 365.25 / 86400.0:.1f} Myr"
        axes[0, 0].plot(radius_proper_kpc, snapshots[i][2], label=label)
        axes[0, 1].plot(radius_proper_kpc, snapshots[i][3])
        axes[1, 0].plot(radius_proper_kpc, snapshots[i][4])
        axes[1, 1].plot(radius_proper_kpc, snapshots[i][5])
    axes[0, 0].set_ylabel("density")
    axes[0, 1].set_ylabel("velocity")
    axes[1, 0].set_ylabel("pressure")
    axes[1, 1].set_ylabel("temperature")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("radius [kpc]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Cold spherical shell onto a finite reflecting wall "
        f"({config['par']['hydrodynamics']['riemann_solver']}, order 0)"
    )
    fig.tight_layout()
    figure = output_directory / "SphericalShellReflectingWall1D.jpg"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"wall post-shock pressure max = {np.max(pre_proper_code[hot]):.6e}")
    print(f"wall post-shock temperature max = {np.max(temp_proper_code[hot]):.6e}")
    print(f"figure = {figure}")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--riemann-solver", choices=("HLLC", "Rusanov"))
    args = parser.parse_args()
    run(args.config, args.riemann_solver)
