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
sys.path.insert(0, str(PROJECT_ROOT))

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
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


def make_initial_condition(ic, units):
    state = State()
    state.par, state.mesh, state.fluid = State(), State(), State()
    state.par.units = type('Units', (), {'CodeUnits': units})()
    state.par.unit_system = units.unit_system
    state.par.simulation = type('Simulation', (), {})()
    state.par.nogrid = int(ic["grid_cells"])
    state.par.mesh = type('MeshParameters', (), {'ghost_cells': 0, 'grid_cells': state.par.nogrid})()
    state.par.coordsys = "spherical"
    state.par.boxsize = np.asarray([float(ic["outer_radius"].to_value(units.length_unit))]) * units.length_unit
    state.par.time = np.asarray([0.0]) * units.time_unit
    state.par.simulation.current_time = state.par.time
    state.par.simulation.coordinate_system = 'spherical'
    state.par.simulation.box_size = state.par.boxsize
    rmin = float(ic["inner_radius"].to_value(units.length_unit))
    rmax = float(ic["outer_radius"].to_value(units.length_unit))
    boundary = np.linspace(rmin, rmax, state.par.nogrid + 1)
    state.mesh.boundary = boundary * units.length_unit
    state.mesh.coordinate = 0.5 * (boundary[1:] + boundary[:-1]) * units.length_unit
    state.mesh.xdelta = np.diff(boundary) * units.length_unit
    state.mesh.area = 4.0 * np.pi * boundary[:-1] ** 2 * units.area_unit
    state.mesh.vol = 4.0 * np.pi / 3.0 * (boundary[1:] ** 3 - boundary[:-1] ** 3) * units.volume_unit
    radius = np.asarray(state.mesh.coordinate.to_value(units.length_unit))
    shell = (radius >= float(ic["shell_inner"].to_value(units.length_unit))) & (radius <= float(ic["shell_outer"].to_value(units.length_unit)))
    state.fluid.rho_code = np.where(shell, float(ic["shell_density"]), 0.0)
    state.fluid.temp_code = np.where(shell, float(ic["temperature"].to_value("K")), 0.0)
    state.fluid.vel_code = np.where(shell, float(ic["velocity"].to_value(units.velocity_unit)), 0.0)
    state.fluid.mu = np.full(state.par.nogrid, float(ic["mean_molecular_weight"]))
    return state


def _profile(sim):
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    r = np.asarray(sim.mesh.coordinate[first:last], dtype=float)
    rho_code = np.asarray(sim.fluid.rho_code[first:last], dtype=float)
    vel_code = np.asarray(sim.fluid.vel_code[first:last], dtype=float)
    pre = np.asarray(sim.fluid.pre_code[first:last], dtype=float)
    temp_code = np.asarray(sim.fluid.temp_code[first:last], dtype=float)
    entropy = np.full_like(pre, np.nan)
    active = rho > 0.0
    entropy[active] = pre[active] / rho[active] ** float(sim.par.hydrodynamics.gamma)
    return r, rho, vel, pre, temp, entropy


def run(config_filename=DEFAULT_CONFIG, riemann_solver=None):
    config = eu.load_nested_example_config(config_filename)
    runparams, icparams = config['par'], config['initial_condition']
    exampleparams = config['example']
    if riemann_solver is not None:
        runparams["hydrodynamics"]["riemann_solver"] = riemann_solver
        runparams["output"]["directory"] = Path(runparams["output"]["directory"]).with_name(
            Path(runparams["output"]["directory"]).name + "_" + riemann_solver
        )
        runparams["output"]["savedir"] = runparams["output"]["directory"]
    outdir = Path(runparams["output"]["directory"])
    outdir.mkdir(parents=True, exist_ok=True)
    units = CodeUnits.from_mapping(runparams["units"]["CodeUnits"])
    initial = make_initial_condition(icparams, units)
    rio.writehdf5(initial, runparams["simulation"]["initial_condition_filename"])
    sim = Rsim(runparams)
    sim.solver = InnerWallSolver()
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    first = int(sim.par.mesh.ghost_cells)
    wall_face = first
    snapshots, fluxes = [], []
    target = float(runparams["simulation"]["final_time"].to_value("s"))
    output_dt = float(runparams["output"]["cadence"].to_value("s"))
    next_output = 0.0
    while float(sim.fluid.time_code) < target:
        sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
        sim.solver.SetConserved(sim.mesh, sim.fluid)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, sim.par)
        dt = sim.GetStepTime(final_time=target)
        sim.solver.SetInterFaceFlux(
            sim.mesh, sim.fluid, runparams['boundary']['condition'],
            method=runparams["hydrodynamics"]["riemann_solver"], order=0,
        )
        fluxes.append([float(sim.fluid.time_code), float(sim.fluid.Mass_code.flux[wall_face]), float(sim.fluid.Mom_code.flux[wall_face]), float(sim.fluid.Energy_code.flux[wall_face])])
        sim.Step(dt=dt, mode="hydro")
        time_code = float(sim.fluid.time_code)
        if time_code >= next_output - 1.0e-12:
            snapshots.append((time_code,) + _profile(sim))
            next_output += output_dt

    snapshots.append((float(sim.fluid.time_code),) + _profile(sim))
    final = snapshots[-1]
    r, rho, vel, pre, temp, entropy = final[1:]
    active = rho > float(runparams["hydrodynamics"].get("cfl_density_floor", 0.0))
    hot = active & (temp > 10.0 * float(icparams["temperature"].to_value("K")))
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
        f"({runparams['hydrodynamics']['riemann_solver']}, order 0)"
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
