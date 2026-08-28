"""Noh spherical implosion benchmark for spherical pressure work.

Uniform cold gas moves inward in a spherical domain and reflects at the
origin.  The converging flow produces a central shock and converts kinetic
energy into thermal energy.  The runner repeats the problem at several
resolutions and compares final radial profiles.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "noh_spherical_implosion1d.yaml"


class State:
    pass


def make_initial_condition(ic, units):
    state = State()
    state.par, state.mesh, state.fluid = State(), State(), State()
    state.par.CodeUnits = units
    state.par.unit_system = units.unit_system
    state.par.nogrid = int(ic["nogrid"])
    state.par.coordsys = "spherical"
    rmax = float(ic["rmax"].to_value(units.length_unit))
    state.par.boxsize = np.asarray([rmax]) * units.length_unit
    state.par.time = np.asarray([0.0]) * units.time_unit
    boundary = np.linspace(0.0, rmax, state.par.nogrid + 1)
    state.mesh.boundary = boundary * units.length_unit
    state.mesh.coordinate = 0.5 * (boundary[1:] + boundary[:-1]) * units.length_unit
    state.mesh.xdelta = np.diff(boundary) * units.length_unit
    state.mesh.area = 4.0 * np.pi * boundary[:-1] ** 2 * units.area_unit
    state.mesh.vol = (
        4.0 * np.pi / 3.0 * np.diff(boundary**3) * units.volume_unit
    )
    state.fluid.rho = np.full(
        state.par.nogrid, float(ic["rhoini"].to_value("g/cm**3"))
    )
    state.fluid.vel = np.full(
        state.par.nogrid, float(ic["vini"].to_value(units.velocity_unit))
    )
    state.fluid.temp = np.full(
        state.par.nogrid, float(ic["tempini"].to_value("K"))
    )
    state.fluid.mu = np.full(state.par.nogrid, float(ic["muini"]))
    return state


def read_profile(filename, units, gamma):
    par, mesh, fluid = State(), State(), State()
    par.CodeUnits = units
    rio.readhdf5(par, mesh, fluid, filename)
    first = int(getattr(par, "noghost", 2))
    last = first + int(par.nogrid)
    boundary = np.asarray(mesh.boundary, dtype=float)
    radius = 0.5 * (boundary[1:] + boundary[:-1])
    volume = 4.0 * np.pi / 3.0 * np.diff(boundary**3)
    rho = np.asarray(fluid.rho, dtype=float)[first:last]
    velocity = np.asarray(fluid.vel, dtype=float)[first:last]
    temperature = np.asarray(fluid.temp, dtype=float)[first:last]
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
    base_runparams, base_icparams = load_example_parameters(config_filename, Path.cwd())
    resolutions = [int(value) for value in base_runparams.pop("resolutions", [256])]
    if dual_energy is not None:
        base_runparams["dual_energy"] = bool(dual_energy)
        if not dual_energy:
            base_runparams["savedir"] = str(
                Path(base_runparams["savedir"]).with_name(
                    Path(base_runparams["savedir"]).name + "_no_dual_energy"
                )
            )
            base_runparams["outdir"] = base_runparams["savedir"]
    root = Path(base_runparams["savedir"])
    root.mkdir(parents=True, exist_ok=True)
    units = CodeUnits.from_mapping(base_runparams["CodeUnits"])
    all_profiles = {}

    for resolution in resolutions:
        runparams = dict(base_runparams)
        icparams = dict(base_icparams)
        output = root / f"resolution_{resolution}"
        output.mkdir(parents=True, exist_ok=True)
        runparams["outdir"] = str(output)
        runparams["savedir"] = str(output)
        runparams["ICfilename"] = str(output / "InitialCondition.hdf5")
        icparams["nogrid"] = resolution
        initial = make_initial_condition(icparams, units)
        rio.writehdf5(initial, runparams["ICfilename"])

        sim = Rsim(runparams)
        sim.RunAll(outputtime=0)
        snapshots = sorted(output.glob("Output_*.hdf5"))
        if len(snapshots) < 2:
            raise RuntimeError(f"Noh resolution {resolution} produced too few outputs")
        profiles = [read_profile(filename, units, runparams["gamma"]) for filename in snapshots]
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
    rmax = float(base_icparams["rmax"].to_value(units.length_unit))
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
