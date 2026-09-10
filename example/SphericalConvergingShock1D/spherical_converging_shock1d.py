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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name("spherical_converging_shock1d.yaml")


def _read_profile(filename, config):
    """Read a proper-code snapshot into the configured typed runtime state."""
    result = Rsim(config["par"])
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    code_unit_system = result.par.units.CodeUnits
    mesh = result.mesh
    fluid = result.fluid
    first = 2
    count = len(mesh.boundary_proper_code) - 1 - 2 * first
    boundary_proper_code = np.asarray(mesh.boundary_proper_code, dtype=float)
    coordinate_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    volume_proper_code = 4.0 * np.pi / 3.0 * (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    rho_proper_code = np.asarray(fluid.rho_proper_code[first:first + count], dtype=float)
    velocity_proper_code = np.asarray(fluid.vel_proper_code[first:first + count], dtype=float)
    temp_proper_code = np.asarray(fluid.temp_proper_code[first:first + count], dtype=float)
    mu = np.asarray(fluid.mu[first:first + count], dtype=float)
    eos = EOS("polytropic", gamma=1.4, code_units=code_unit_system)
    pressure_proper_code = eos.pressure(
        rho_proper_code, temp_proper_code, mu
    )
    total_mass_code = float(
        np.sum(np.asarray(fluid.Mass_code[first:first + count], dtype=float))
        if hasattr(fluid, "Mass_code")
        else np.sum(rho_proper_code * volume_proper_code[first:first + count])
    )
    total_energy_proper_code = float(
        np.sum(np.asarray(fluid.Energy_code[first:first + count], dtype=float))
        if hasattr(fluid, "Energy_code")
        else np.sum(
            eos.total_energy_density(
                rho_proper_code, velocity_proper_code, pressure_proper_code
            ) * volume_proper_code[first:first + count]
        )
    )
    thermal = float(
        np.sum(
            eos.thermal_energy_density(pressure_proper_code)
            * volume_proper_code[first:first + count]
        )
    )
    return (
        coordinate_proper_code[first:first + count],
        rho_proper_code,
        velocity_proper_code,
        temp_proper_code,
        total_mass_code,
        total_energy_proper_code,
        thermal,
    )


def run(config_filename=DEFAULT_CONFIG, riemann_solver=None, dual_energy=None):
    config = eu.load_nested_example_config(config_filename)

    if riemann_solver is not None:
        config["par"]["hydrodynamics"]["riemann_solver"] = riemann_solver
    if dual_energy is not None:
        config["par"]["hydrodynamics"]["dual_energy"] = dual_energy
    output = config["par"]['output']
    eu.clean_previous_outputs(config)
    units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    config['_code_units'] = units
    initial = et.build_initial_condition(config)
    rio.writehdf5(initial, config["par"]["simulation"]["initial_condition_filename"])

    sim = Rsim(config["par"])
    sim.RunAll(outputtime=0)
    outputs = sorted(Path(output["directory"]).glob("Output_*.hdf5"))
    if len(outputs) < 2:
        raise RuntimeError("spherical converging benchmark produced too few outputs")

    profiles = [_read_profile(filename, config) for filename in outputs]
    # Keep the reference plot focused on the first 13 snapshots.  This also
    # makes an interrupted run with those snapshots available reproducible
    # without including a later, potentially incomplete tail.
    profiles = profiles[:13]
    initial_mass, initial_energy = profiles[0][4:6]
    final_mass, final_energy = profiles[-1][4:6]
    temperature_proper_code = profiles[0][3]
    final_temperature_proper_code = profiles[-1][3]
    thermal_energy = []
    for profile in profiles:
        thermal_energy.append(profile[6])
    thermal_energy = np.asarray(thermal_energy)
    if not thermal_energy[-1] > thermal_energy[0]:
        raise RuntimeError("converging flow did not increase thermal energy")
    if not np.max(final_temperature_proper_code) > 5.0 * np.max(temperature_proper_code):
        raise RuntimeError("converging flow did not produce a resolved central shock")
    if not np.isclose(final_mass, initial_mass, rtol=2.0e-6):
        raise RuntimeError("spherical reflecting benchmark lost mass")
    if not np.isclose(final_energy, initial_energy, rtol=2.0e-5, atol=2.0e-10):
        raise RuntimeError("spherical reflecting benchmark lost total energy")

    figure = Path(output["directory"]) / "SphericalConvergingShock1D.jpg"
    selected = np.unique(np.linspace(0, len(profiles) - 1, min(6, len(profiles))).astype(int))
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for index in selected:
        radius_proper_code, rho_proper_code, velocity_proper_code, temp_proper_code, _, _, _ = profiles[index]
        axes[0].plot(radius_proper_code, rho_proper_code, label=f"output {index:03d}")
        axes[1].plot(radius_proper_code, temp_proper_code, label=f"output {index:03d}")
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
    print(
        "central temperature amplification = "
        f"{np.max(final_temperature_proper_code) / np.max(temperature_proper_code):.6e}"
    )
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--riemann-solver", choices=("Rusanov", "HLLC"))
    parser.add_argument("--dual-energy", action="store_true", default=None)
    args = parser.parse_args()
    run(
        args.config,
        riemann_solver=args.riemann_solver,
        dual_energy=args.dual_energy,
    )
