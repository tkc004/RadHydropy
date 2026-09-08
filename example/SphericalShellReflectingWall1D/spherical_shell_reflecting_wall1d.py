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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.eos import EOS
from radhydropy.runtime_fields import (
    FluidRuntimeState, MeshGeometryState, PROPER_RUNTIME_FIELDS,
)
from radhydropy.units import CodeUnits
import example_utils as eu

DEFAULT_CONFIG = Path(__file__).with_name("spherical_shell_reflecting_wall1d.yaml")


class State:
    pass


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


def make_initial_condition(ic, code_unit_system):
    state = State()
    state.par, state.mesh, state.fluid = State(), State(), State()
    state.par.code_unit_system = type('Units', (), {'CodeUnits': code_unit_system})()
    state.par.units = state.par.code_unit_system
    state.par.unit_system = code_unit_system.unit_system
    state.par.simulation = type('Simulation', (), {})()
    state.par.nogrid = int(ic["grid_cells"])
    state.par.mesh = type('MeshParameters', (), {'ghost_cells': 0, 'grid_cells': state.par.nogrid})()
    state.par.coordsys = "spherical"
    state.par.boxsize = np.asarray([float(ic["outer_radius"].to_value(code_unit_system.length_unit))]) * code_unit_system.length_unit
    state.par.time_proper_code = np.asarray([0.0]) * code_unit_system.time_unit
    state.par.simulation.time_proper_code = state.par.time_proper_code
    state.par.simulation.coordinate_system = 'spherical'
    state.par.simulation.box_size = state.par.boxsize
    rmin = float(ic["inner_radius"].to_value(code_unit_system.length_unit))
    rmax = float(ic["outer_radius"].to_value(code_unit_system.length_unit))
    boundary_proper_code = np.linspace(rmin, rmax, state.par.nogrid + 1)
    boundary_proper_code_unyt = boundary_proper_code * code_unit_system.length_unit
    x_proper_code_unyt = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1]) * code_unit_system.length_unit
    width_proper_code_unyt = np.diff(boundary_proper_code) * code_unit_system.length_unit
    area_proper_code_unyt = 4.0 * np.pi * boundary_proper_code[:-1] ** 2 * code_unit_system.area_unit
    volume_proper_code_unyt = 4.0 * np.pi / 3.0 * (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3) * code_unit_system.volume_unit
    radius_proper_code = np.asarray(x_proper_code_unyt.to_value(code_unit_system.length_unit))
    shell = (radius_proper_code >= float(ic["shell_inner"].to_value(code_unit_system.length_unit))) & (radius_proper_code <= float(ic["shell_outer"].to_value(code_unit_system.length_unit)))
    state.fluid.rho_proper_code = np.where(shell, float(ic["shell_density"]), 0.0)
    state.fluid.temp_proper_code = np.where(shell, float(ic["temperature"].to_value("K")), 0.0)
    state.fluid.vel_proper_code = np.where(shell, float(ic["velocity"].to_value(code_unit_system.velocity_unit)), 0.0)
    state.fluid.mu = np.full(state.par.nogrid, float(ic["mean_molecular_weight"]))
    boundary_proper_code = np.asarray(boundary_proper_code_unyt.to_value(code_unit_system.length_unit), dtype=float)
    x_proper_code = np.asarray(x_proper_code_unyt.to_value(code_unit_system.length_unit), dtype=float)
    width_proper_code = np.asarray(width_proper_code_unyt.to_value(code_unit_system.length_unit), dtype=float)
    area_proper_code = np.asarray(area_proper_code_unyt.to_value(code_unit_system.area_unit), dtype=float)
    volume_proper_code = np.asarray(volume_proper_code_unyt.to_value(code_unit_system.volume_unit), dtype=float)
    state.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=x_proper_code,
        boundary_proper_code=boundary_proper_code,
        width_proper_code=width_proper_code,
        area_proper_code=area_proper_code,
        volume_proper_code=volume_proper_code,
    )
    state.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    state.fluid.time_proper_code = 0.0
    state.fluid.eos = EOS('polytropic', 5.0 / 3.0, code_unit_system)
    state.fluid.pre_proper_code = state.fluid.eos.pressure(
        state.fluid.rho_proper_code,
        state.fluid.temp_proper_code,
        state.fluid.mu,
    )
    state.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=state.fluid.rho_proper_code,
        vel_proper_code=state.fluid.vel_proper_code,
        pre_proper_code=state.fluid.pre_proper_code,
        temp_proper_code=state.fluid.temp_proper_code,
        time_proper_code=state.fluid.time_proper_code,
        mu_dimensionless=state.fluid.mu,
    )
    return state


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
    exampleparams = config['example']
    if riemann_solver is not None:
        config["par"]["hydrodynamics"]["riemann_solver"] = riemann_solver
        config["par"]["output"]["directory"] = Path(config["par"]["output"]["directory"]).with_name(
            Path(config["par"]["output"]["directory"]).name + "_" + riemann_solver
        )
        config["par"]["output"]["savedir"] = config["par"]["output"]["directory"]
    outdir = Path(config["par"]["output"]["directory"])
    outdir.mkdir(parents=True, exist_ok=True)
    units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    initial = make_initial_condition(initial_condition, units)
    rio.writehdf5(initial, config["par"]["simulation"]["initial_condition_filename"])
    sim = Rsim(config["par"])
    sim.solver = InnerWallSolver()
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    first = int(sim.par.mesh.ghost_cells)
    wall_face = first
    snapshots, fluxes = [], []
    target = float(par_config["simulation"]["final_time"].to_value("s"))
    output_dt = float(par_config["output"]["cadence"].to_value("s"))
    next_output = 0.0
    while float(sim.fluid.time_proper_code) < target:
        sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
        sim.solver.SetConserved(sim.mesh, sim.fluid)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, sim.par)
        dt = sim.GetStepTime(final_time=target)
        sim.solver.SetInterFaceFlux(
            sim.mesh, sim.fluid, par_config['boundary']['condition'],
            method=par_config["hydrodynamics"]["riemann_solver"], order=0,
        )
        fluxes.append([float(sim.fluid.time_proper_code), float(sim.fluid.Mass_code.flux[wall_face]), float(sim.fluid.Mom_code.flux[wall_face]), float(sim.fluid.Energy_code.flux[wall_face])])
        sim.Step(dt=dt, mode="hydro")
        time_proper_code = float(sim.fluid.time_proper_code)
        if time_proper_code >= next_output - 1.0e-12:
            snapshots.append((time_proper_code,) + _profile(sim))
            next_output += output_dt

    snapshots.append((float(sim.fluid.time_proper_code),) + _profile(sim))
    final = snapshots[-1]
    r, rho, vel, pre, temp, entropy = final[1:]
    active = rho > float(par_config["hydrodynamics"].get("cfl_density_floor", 0.0))
    hot = active & (temp > 10.0 * float(initial_condition["temperature"].to_value("K")))
    if not np.any(hot):
        raise RuntimeError("finite reflecting wall did not produce post-shock heating")
    data = {"time": np.array([s[0] for s in snapshots]), "radius": r, "rho": np.array([s[2] for s in snapshots]), "velocity": np.array([s[3] for s in snapshots]), "pressure": np.array([s[4] for s in snapshots]), "temperature": np.array([s[5] for s in snapshots]), "entropy": np.array([s[6] for s in snapshots]), "wall_flux": np.asarray(fluxes)}
    np.savez(outdir / "SphericalShellReflectingWall1D_diagnostics.npz", **data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    radius_kpc = r / 3.085677581e21
    for i in np.unique(np.linspace(0, len(snapshots) - 1, min(5, len(snapshots))).astype(int)):
        label = f"{snapshots[i][0] / 1.0e6 / 365.25 / 86400.0:.1f} Myr"
        axes[0, 0].plot(radius_kpc, snapshots[i][2], label=label)
        axes[0, 1].plot(radius_kpc, snapshots[i][3])
        axes[1, 0].plot(radius_kpc, snapshots[i][4])
        axes[1, 1].plot(radius_kpc, snapshots[i][5])
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
        f"({par_config['hydrodynamics']['riemann_solver']}, order 0)"
    )
    fig.tight_layout()
    figure = outdir / "SphericalShellReflectingWall1D.jpg"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"wall post-shock pressure max = {np.max(pre[hot]):.6e}")
    print(f"wall post-shock temperature max = {np.max(temp[hot]):.6e}")
    print(f"figure = {figure}")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--riemann-solver", choices=("HLLC", "Rusanov"))
    args = parser.parse_args()
    run(args.config, args.riemann_solver)
