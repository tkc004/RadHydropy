"""Spherical converging-flow shock benchmark.

This is a controlled, gravity-free test of the spherical Euler/Riemann path.
Uniform gas starts with inward radial velocity and reflects at the origin and
outer wall.  Once the inward flow reaches the origin, a converging shock must
convert kinetic energy into thermal energy while conserving total energy.
"""

import argparse
from pathlib import Path
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
from radhydropy.example_config import load_example_parameters
from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu


DEFAULT_CONFIG = Path(__file__).with_name("spherical_converging_shock1d.yaml")


class InitialCondition:
    pass


def make_initial_condition(ic, units):
    state = InitialCondition()
    state.par = InitialCondition()
    state.mesh = InitialCondition()
    state.fluid = InitialCondition()
    state.par.CodeUnits = units
    state.par.unit_system = units.unit_system
    state.par.nogrid = int(ic["nogrid"])
    state.par.coordsys = "spherical"
    state.par.boxsize = np.asarray([float(ic["boxsize"].to_value("cm"))])
    state.par.time = np.asarray([0.0])
    faces = np.linspace(0.0, state.par.boxsize[0], state.par.nogrid + 1)
    state.mesh.boundary = faces
    state.mesh.coordinate = 0.5 * (faces[1:] + faces[:-1])
    state.mesh.area = 4.0 * np.pi * faces[:-1] ** 2
    state.mesh.vol = 4.0 * np.pi / 3.0 * np.diff(faces ** 3)
    state.fluid.rho = np.full(state.par.nogrid, float(ic["rhoini"].to_value("g/cm**3")))
    state.fluid.temp = np.full(state.par.nogrid, float(ic["tempini"].to_value("K")))
    state.fluid.vel = np.full(state.par.nogrid, float(ic["vini"].to_value("cm/s")))
    state.fluid.mu = np.full(state.par.nogrid, float(ic["muini"]))
    return state


def _read_profile(filename, units):
    par = InitialCondition()
    mesh = InitialCondition()
    fluid = InitialCondition()
    par.CodeUnits = units
    rio.readhdf5(par, mesh, fluid, filename)
    first = int(getattr(par, "noghost", 2))
    count = int(getattr(par, "nogrid"))
    boundary = np.asarray(mesh.boundary, dtype=float)
    coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    volume = 4.0 * np.pi / 3.0 * (boundary[1:] ** 3 - boundary[:-1] ** 3)
    rho = np.asarray(fluid.rho[first:first + count], dtype=float)
    velocity = np.asarray(fluid.vel[first:first + count], dtype=float)
    temp = np.asarray(fluid.temp[first:first + count], dtype=float)
    mu = np.asarray(fluid.mu[first:first + count], dtype=float)
    eos = EOS("polytropic", gamma=1.4, code_units=units)
    pressure = eos.pressure(rho, temp, mu)
    mass = float(
        np.sum(np.asarray(fluid.Mass[first:first + count], dtype=float))
        if hasattr(fluid, "Mass")
        else np.sum(rho * volume[first:first + count])
    )
    energy = float(
        np.sum(np.asarray(fluid.Energy[first:first + count], dtype=float))
        if hasattr(fluid, "Energy")
        else np.sum(eos.total_energy_density(rho, velocity, pressure) * volume[first:first + count])
    )
    thermal = float(np.sum(eos.thermal_energy_density(pressure) * volume[first:first + count]))
    return (
        coordinate[first:first + count],
        rho,
        velocity,
        temp,
        mass,
        energy,
        thermal,
    )


def run(config_filename=DEFAULT_CONFIG, riemann_solver=None, dual_energy=None):
    runparams, icparams = load_example_parameters(config_filename, Path.cwd())
    if riemann_solver is not None:
        runparams["riemann_solver"] = riemann_solver
    if dual_energy is not None:
        runparams["dual_energy"] = dual_energy
    eu.clean_previous_outputs(runparams)
    units = CodeUnits.from_mapping(runparams["CodeUnits"])
    initial = make_initial_condition(icparams, units)
    rio.writehdf5(initial, runparams["ICfilename"])

    sim = Rsim(runparams)
    sim.RunAll(outputtime=0)
    outputs = sorted(Path(runparams["outdir"]).glob("Output_*.hdf5"))
    if len(outputs) < 2:
        raise RuntimeError("spherical converging benchmark produced too few outputs")

    profiles = [_read_profile(filename, units) for filename in outputs]
    initial_mass, initial_energy = profiles[0][4:6]
    final_mass, final_energy = profiles[-1][4:6]
    initial_temperature = profiles[0][3]
    final_temperature = profiles[-1][3]
    thermal_energy = []
    for profile in profiles:
        thermal_energy.append(profile[6])
    thermal_energy = np.asarray(thermal_energy)
    if not thermal_energy[-1] > thermal_energy[0]:
        raise RuntimeError("converging flow did not increase thermal energy")
    if not np.max(final_temperature) > 5.0 * np.max(initial_temperature):
        raise RuntimeError("converging flow did not produce a resolved central shock")
    if not np.isclose(final_mass, initial_mass, rtol=2.0e-6):
        raise RuntimeError("spherical reflecting benchmark lost mass")
    if not np.isclose(final_energy, initial_energy, rtol=2.0e-5, atol=2.0e-10):
        raise RuntimeError("spherical reflecting benchmark lost total energy")

    figure = Path(runparams["savedir"]) / "SphericalConvergingShock1D.jpg"
    selected = np.unique(np.linspace(0, len(profiles) - 1, min(6, len(profiles))).astype(int))
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for index in selected:
        radius, rho, velocity, temp, _, _, _ = profiles[index]
        axes[0].plot(radius, rho, label=f"output {index:03d}")
        axes[1].plot(radius, temp, label=f"output {index:03d}")
    axes[0].set_ylabel("density [code units]")
    axes[1].set_ylabel("temperature [code units]")
    axes[1].set_xlabel("radius [code length]")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("Spherical converging-flow shock benchmark")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"figure = {figure}")
    print(f"mass relative error = {(final_mass - initial_mass) / initial_mass:.6e}")
    print(f"energy relative error = {(final_energy - initial_energy) / initial_energy:.6e}")
    print(f"thermal energy increase = {thermal_energy[-1] / thermal_energy[0]:.6e}")
    print(f"central temperature amplification = {np.max(final_temperature) / np.max(initial_temperature):.6e}")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--riemann-solver", choices=("Rusanov", "HLLC"))
    parser.add_argument("--dual-energy", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        riemann_solver=args.riemann_solver,
        dual_energy=args.dual_energy,
    )
