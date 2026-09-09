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
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
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
from basic_hydro_utils import make_initial_condition as make_canonical_initial_condition


DEFAULT_CONFIG = HERE / "noh_spherical_implosion1d.yaml"


def make_initial_condition(config):
    code_unit_system = CodeUnits.from_mapping(
        config['par']['units']['CodeUnits']
    )
    ic = config['initial_condition']
    n = int(ic["grid_cells"])
    box_size_proper_code = float(
        ic["box_size_proper"].to_value(code_unit_system.length_unit)
    )
    boundary_proper_code = np.linspace(0.0, box_size_proper_code, n + 1)
    return make_canonical_initial_condition(
        config,
        boundary_proper_code=boundary_proper_code,
        rho_proper_code=np.full(n, float(ic["rho_proper"].to_value("g/cm**3"))),
        vel_proper_code=np.full(n, float(ic["vel_proper"].to_value(code_unit_system.velocity_unit))),
        temp_proper_code=np.full(n, float(ic["temperature_proper"].to_value("K"))),
        mu_dimensionless=np.full(n, float(ic["mean_molecular_weight"])),
    )


def read_profile(filename, config):
    code_unit_system = CodeUnits.from_mapping(
        config['par']['units']['CodeUnits']
    )
    sim = Rsim(config['par'])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    boundary_proper_code = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    coordinate_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(boundary_proper_code**3)
    rho_proper_code = np.asarray(sim.fluid.rho_proper_code, dtype=float)[first:last]
    vel_proper_code = np.asarray(sim.fluid.vel_proper_code, dtype=float)[first:last]
    temp_proper_code = np.asarray(sim.fluid.temp_proper_code, dtype=float)[first:last]
    mu_dimensionless = np.asarray(sim.fluid.mu, dtype=float)[first:last]
    eos = EOS(
        "polytropic",
        gamma=float(config['par']['hydrodynamics']['gamma']),
        code_units=code_unit_system,
    )
    pre_proper_code = np.asarray(eos.pressure(rho_proper_code, temp_proper_code, mu_dimensionless), dtype=float)
    kinetic_energy_proper_code = 0.5 * rho_proper_code * vel_proper_code**2 * volume_proper_code[first:last]
    thermal_energy_proper_code = np.asarray(eos.thermal_energy_density(pre_proper_code), dtype=float) * volume_proper_code[first:last]
    return {
        "radius_proper_code": coordinate_proper_code[first:last],
        "rho_proper_code": rho_proper_code,
        "vel_proper_code": vel_proper_code,
        "temp_proper_code": temp_proper_code,
        "pre_proper_code": pre_proper_code,
        "kinetic_energy_proper_code": float(np.sum(kinetic_energy_proper_code)),
        "thermal_energy_proper_code": float(np.sum(thermal_energy_proper_code)),
        "time_proper_code": float(np.asarray(sim.par.time_proper_code).flat[0]),
    }


def run(config_filename=DEFAULT_CONFIG, dual_energy=None):
    config = eu.load_nested_example_config(config_filename)
    base_par_config, base_initial_condition = config['par'], config['initial_condition']
    exampleparams = config['example']
    resolutions = [int(value) for value in exampleparams.get("resolutions", [256])]
    if dual_energy is not None:
        base_par_config["hydrodynamics"]["dual_energy"] = bool(dual_energy)
        if not dual_energy:
            base_par_config["output"]["directory"] = str(
                Path(base_par_config["output"]["directory"]).with_name(
                    Path(base_par_config["output"]["directory"]).name + "_no_dual_energy"
                )
            )
            base_par_config["output"]["directory"] = base_par_config["output"]["directory"]
    root = Path(base_par_config["output"]["directory"])
    root.mkdir(parents=True, exist_ok=True)
    units = CodeUnits.from_mapping(base_par_config["units"]["CodeUnits"])
    all_profiles = {}

    for resolution in resolutions:
        par_config = dict(base_par_config)
        initial_condition = dict(base_initial_condition)
        output = root / f"resolution_{resolution}"
        output.mkdir(parents=True, exist_ok=True)
        par_config["output"]["directory"] = str(output)
        par_config["output"]["directory"] = str(output)
        par_config["simulation"]["initial_condition_filename"] = str(output / "InitialCondition.hdf5")
        initial_condition["grid_cells"] = resolution
        par_config["mesh"]["grid_cells"] = resolution
        resolution_config = dict(config)
        resolution_config["par"] = par_config
        resolution_config["initial_condition"] = initial_condition
        initial = make_initial_condition(resolution_config)
        rio.writehdf5(
            initial, par_config["simulation"]["initial_condition_filename"]
        )

        sim = Rsim(resolution_config["par"])
        sim.RunAll(outputtime=0)
        snapshots = sorted(output.glob("Output_*.hdf5"))
        if len(snapshots) < 2:
            raise RuntimeError(f"Noh resolution {resolution} produced too few outputs")
        profiles = [read_profile(filename, resolution_config) for filename in snapshots]
        all_profiles[resolution] = profiles
        initial_profile, final_profile = profiles[0], profiles[-1]
        if not final_profile["thermal_energy_proper_code"] > initial_profile["thermal_energy_proper_code"]:
            raise RuntimeError(f"Noh resolution {resolution} did not heat")
        if not np.max(final_profile["temp_proper_code"]) > 10.0 * np.max(initial_profile["temp_proper_code"]):
            raise RuntimeError(f"Noh resolution {resolution} did not form a hot central shock")
        print(
            "resolution=%d thermal_ratio=%.6e Tmax=%.6e total_energy=(%.6e, %.6e)"
            % (
                resolution,
                final_profile["thermal_energy_proper_code"] / initial_profile["thermal_energy_proper_code"],
                np.max(final_profile["temp_proper_code"]),
                initial_profile["kinetic_energy_proper_code"] + initial_profile["thermal_energy_proper_code"],
                final_profile["kinetic_energy_proper_code"] + final_profile["thermal_energy_proper_code"],
            )
        )

    selected = sorted(all_profiles)
    final = {resolution: all_profiles[resolution][-1] for resolution in selected}
    box_size_proper_code = float(
        base_initial_condition["box_size_proper"].to_value(units.length_unit)
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for resolution in selected:
        profile = final[resolution]
        radius_proper_code = profile["radius_proper_code"] / box_size_proper_code
        axes[0, 0].plot(radius_proper_code, profile["rho_proper_code"], label=f"N={resolution}")
        axes[0, 1].plot(radius_proper_code, profile["temp_proper_code"], label=f"N={resolution}")
        axes[1, 0].plot(radius_proper_code, profile["vel_proper_code"])
        axes[1, 1].plot(radius_proper_code, profile["pre_proper_code"])
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
        reference_rho = np.interp(
            profile["radius_proper_code"],
            reference["radius_proper_code"],
            reference["rho_proper_code"],
        )
        convergence.append(
            [resolution,
             np.mean(np.abs(profile["rho_proper_code"] - reference_rho))
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
