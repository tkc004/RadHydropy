"""Noh spherical implosion benchmark for spherical pressure work.

Uniform cold gas moves inward in a spherical domain and reflects at the
origin.  The converging flow produces a central shock and converts kinetic
energy into thermal energy.  The runner repeats the problem at several
resolutions and compares final radial profiles.
"""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu


DEFAULT_CONFIG = HERE / "noh_spherical_implosion1d.yaml"


class State:
    pass


def make_initial_condition(ic, units):
    state = State()
    state.par, state.mesh, state.fluid = State(), State(), State()
    state.par.units = type('Units', (), {'CodeUnits': units})()
    state.par.unit_system = units.unit_system
    state.par.simulation = type('Simulation', (), {})()
    state.par.mesh = type('MeshParameters', (), {'ghost_cells': 0, 'grid_cells': int(ic['grid_cells'])})()
    state.par.nogrid = int(ic["grid_cells"])
    state.par.coordsys = "spherical"
    rmax = float(ic["box_size"].to_value(units.length_unit))
    state.par.boxsize = np.asarray([rmax]) * units.length_unit
    state.par.time = np.asarray([0.0]) * units.time_unit
    state.par.simulation.current_time = state.par.time
    state.par.simulation.coordinate_system = 'spherical'
    state.par.simulation.box_size = state.par.boxsize
    boundary = np.linspace(0.0, rmax, state.par.nogrid + 1)
    state.mesh.boundary = boundary * units.length_unit
    state.mesh.coordinate = 0.5 * (boundary[1:] + boundary[:-1]) * units.length_unit
    state.mesh.xdelta = np.diff(boundary) * units.length_unit
    state.mesh.area = 4.0 * np.pi * boundary[:-1] ** 2 * units.area_unit
    state.mesh.vol = (
        4.0 * np.pi / 3.0 * np.diff(boundary**3) * units.volume_unit
    )
    state.fluid.rho_code = np.full(
        state.par.nogrid, float(ic["initial_density"].to_value("g/cm**3"))
    )
    state.fluid.vel_code = np.full(
        state.par.nogrid, float(ic["velocity"].to_value(units.velocity_unit))
    )
    state.fluid.temp_code = np.full(
        state.par.nogrid, float(ic["temperature"].to_value("K"))
    )
    state.fluid.mu = np.full(state.par.nogrid, float(ic["mean_molecular_weight"]))
    return state


def read_profile(filename, units, gamma):
    par, mesh, fluid = State(), State(), State()
    par.CodeUnits = units
    par.simulation = type("Simulation", (), {"coordinate_system": "spherical"})()
    par.mesh = type("MeshParameters", (), {"grid_cells": None, "ghost_cells": 0})()
    rio.readhdf5(par, mesh, fluid, filename)
    first = int(getattr(par, "noghost", 2))
    last = first + int(par.nogrid)
    boundary = np.asarray(mesh.boundary, dtype=float)
    radius = 0.5 * (boundary[1:] + boundary[:-1])
    volume = 4.0 * np.pi / 3.0 * np.diff(boundary**3)
    rho_code = np.asarray(fluid.rho_code, dtype=float)[first:last]
    velocity = np.asarray(fluid.vel_code, dtype=float)[first:last]
    temperature = np.asarray(fluid.temp_code, dtype=float)[first:last]
    mu = np.asarray(fluid.mu, dtype=float)[first:last]
    eos = EOS("polytropic", gamma=gamma, code_units=units)
    pressure = np.asarray(eos.pressure(rho, temperature, mu), dtype=float)
    kinetic = 0.5 * rho * velocity**2 * volume[first:last]
    thermal = np.asarray(eos.thermal_energy_density(pressure), dtype=float) * volume[first:last]
    return {
        "radius": radius[first:last],
        "rho": rho,
        "velocity": velocity,
        "temperature": temperature,
        "pressure": pressure,
        "kinetic": float(np.sum(kinetic)),
        "thermal": float(np.sum(thermal)),
        "time": float(np.asarray(par.time).flat[0]),
    }


def run(config_filename=DEFAULT_CONFIG, dual_energy=None):
    config = eu.load_nested_example_config(config_filename)
    base_runparams, base_icparams = config['par'], config['initial_condition']
    exampleparams = config['example']
    resolutions = [int(value) for value in exampleparams.get("resolutions", [256])]
    if dual_energy is not None:
        base_runparams["hydrodynamics"]["dual_energy"] = bool(dual_energy)
        if not dual_energy:
            base_runparams["output"]["savedir"] = str(
                Path(base_runparams["output"]["savedir"]).with_name(
                    Path(base_runparams["output"]["savedir"]).name + "_no_dual_energy"
                )
            )
            base_runparams["output"]["directory"] = base_runparams["output"]["savedir"]
    root = Path(base_runparams["output"]["savedir"])
    root.mkdir(parents=True, exist_ok=True)
    units = CodeUnits.from_mapping(base_runparams["units"]["CodeUnits"])
    all_profiles = {}

    for resolution in resolutions:
        runparams = dict(base_runparams)
        icparams = dict(base_icparams)
        output = root / f"resolution_{resolution}"
        output.mkdir(parents=True, exist_ok=True)
        runparams["output"]["directory"] = str(output)
        runparams["output"]["savedir"] = str(output)
        runparams["simulation"]["initial_condition_filename"] = str(output / "InitialCondition.hdf5")
        icparams["grid_cells"] = resolution
        runparams["mesh"]["grid_cells"] = resolution
        initial = make_initial_condition(icparams, units)
        rio.writehdf5(
            initial, runparams["simulation"]["initial_condition_filename"]
        )

        sim = Rsim(runparams)
        sim.RunAll(outputtime=0)
        snapshots = sorted(output.glob("Output_*.hdf5"))
        if len(snapshots) < 2:
            raise RuntimeError(f"Noh resolution {resolution} produced too few outputs")
        profiles = [read_profile(filename, units, runparams["hydrodynamics"]["gamma"]) for filename in snapshots]
        all_profiles[resolution] = profiles
        initial_profile, final_profile = profiles[0], profiles[-1]
        if not final_profile["thermal"] > initial_profile["thermal"]:
            raise RuntimeError(f"Noh resolution {resolution} did not heat")
        if not np.max(final_profile["temperature"]) > 10.0 * np.max(initial_profile["temperature"]):
            raise RuntimeError(f"Noh resolution {resolution} did not form a hot central shock")
        print(
            "resolution=%d thermal_ratio=%.6e Tmax=%.6e total_energy=(%.6e, %.6e)"
            % (
                resolution,
                final_profile["thermal"] / initial_profile["thermal"],
                np.max(final_profile["temperature"]),
                initial_profile["kinetic"] + initial_profile["thermal"],
                final_profile["kinetic"] + final_profile["thermal"],
            )
        )

    selected = sorted(all_profiles)
    final = {resolution: all_profiles[resolution][-1] for resolution in selected}
    rmax = float(base_icparams["box_size"].to_value(units.length_unit))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for resolution in selected:
        profile = final[resolution]
        radius = profile["radius"] / rmax
        axes[0, 0].plot(radius, profile["rho"], label=f"N={resolution}")
        axes[0, 1].plot(radius, profile["temperature"], label=f"N={resolution}")
        axes[1, 0].plot(radius, profile["velocity"])
        axes[1, 1].plot(radius, profile["pressure"])
    axes[0, 0].set_ylabel(r"density [$\mathrm{g\,cm^{-3}}$]")
    axes[0, 1].set_ylabel("temperature [K]")
    axes[1, 0].set_ylabel(r"velocity [$\mathrm{cm\,s^{-1}}$]")
    axes[1, 1].set_ylabel(r"pressure [$\mathrm{erg\,cm^{-3}}$]")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel(r"radius / $R_{\rm max}$")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Noh spherical implosion: final radial-profile convergence")
    fig.tight_layout()
    figure = root / "NohSphericalImplosion1D_Profiles.jpg"
    fig.savefig(figure, dpi=200)
    plt.close(fig)

    reference = final[selected[-1]]
    convergence = []
    for resolution in selected[:-1]:
        profile = final[resolution]
        reference_rho = np.interp(profile["radius"], reference["radius"], reference["rho"])
        convergence.append(
            [resolution,
             np.mean(np.abs(profile["rho"] - reference_rho))
             / max(np.mean(np.abs(reference_rho)), 1.0e-300)]
        )
    convergence = np.asarray(convergence, dtype=float)
    np.savez(
        root / "NohSphericalImplosion1D_Convergence.npz",
        resolutions=np.asarray(selected),
        convergence=convergence,
    )
    if len(convergence):
        fig, axis = plt.subplots(figsize=(6, 4.5))
        axis.loglog(convergence[:, 0], convergence[:, 1], "o-")
        axis.set_xlabel("resolution (number of cells)")
        axis.set_ylabel("density-profile L1 error vs. finest run")
        axis.set_title("Noh radial-profile convergence")
        axis.grid(alpha=0.3, which="both")
        fig.tight_layout()
        convergence_figure = root / "NohSphericalImplosion1D_Convergence.jpg"
        fig.savefig(convergence_figure, dpi=200)
        plt.close(fig)
    else:
        convergence_figure = None
    print(f"profile figure = {figure}")
    print(f"convergence figure = {convergence_figure}")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--without-dual-energy",
        action="store_true",
        help="disable the independent InternalEnergy evolution",
    )
    args = parser.parse_args()
    run(args.config, dual_energy=False if args.without_dual_energy else None)
