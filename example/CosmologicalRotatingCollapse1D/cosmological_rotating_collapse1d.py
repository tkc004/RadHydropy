"""Compare rotating and nonrotating spherical collapse in an EdS universe.

This is a one-dimensional spherical centrifugal-barrier benchmark.  It is not
a multidimensional disk-formation calculation: each shell carries its own
conserved signed specific angular momentum.
"""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/radhydropy-matplotlib")
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu


DEFAULT_CONFIG = Path(__file__).with_name("cosmological_rotating_collapse1d.yaml")


def spherical_centers(boundary):
    return 0.75 * (
        boundary[1:]**4 - boundary[:-1]**4
    ) / (boundary[1:]**3 - boundary[:-1]**3)


def integrate_shell_reference(initial, cosmology, scale_factors):
    """Integrate pressureless physical shell orbits for comparison only."""
    radius = np.asarray(initial.mesh.coordinate, dtype=float)
    mass = np.cumsum(
        np.asarray(initial.fluid.rho_code, dtype=float)
        * np.asarray(initial.mesh.vol, dtype=float)
    )
    j = np.asarray(initial.fluid.specific_angular_momentum_code, dtype=float)
    cosmic_time_initial = float(
        cosmology.cosmic_time_from_supercomoving(initial.par.time)
    )
    scale_initial = float(cosmology.scale_factor(cosmic_time_initial))
    hubble_initial = float(cosmology.hubble(cosmic_time_initial))
    initial_physical_radius = scale_initial * radius
    initial_physical_velocity = (
        hubble_initial * initial_physical_radius
        + np.asarray(initial.fluid.vel_code, dtype=float) / scale_initial
    )
    requested_times = cosmology.t_ref * np.asarray(scale_factors, dtype=float)**1.5
    cosmic_times = np.unique(requested_times)
    reference = np.empty((len(cosmic_times), len(radius)), dtype=float)
    for shell, shell_mass in enumerate(mass):
        def rhs(time, state):
            shell_radius, shell_velocity = state
            if shell_radius <= 0.0:
                return shell_velocity, 0.0
            acceleration = (
                -cosmology.gravitational_constant * shell_mass / shell_radius**2
                + j[shell]**2 / shell_radius**3
            )
            return shell_velocity, acceleration

        solution = solve_ivp(
            rhs,
            (cosmic_times[0], cosmic_times[-1]),
            (initial_physical_radius[shell], initial_physical_velocity[shell]),
            t_eval=cosmic_times,
            rtol=1.0e-8, atol=1.0e-10,
            max_step=max((cosmic_times[-1] - cosmic_times[0]) / 32.0, 1.0e-8),
        )
        if not solution.success:
            raise RuntimeError(
                "shell ODE failed for shell %d: %s" % (shell, solution.message)
            )
        reference[:, shell] = solution.y[0]
    reference /= np.asarray(cosmology.scale_factor(cosmic_times), dtype=float)[:, None]
    if len(cosmic_times) == len(requested_times):
        return reference
    return np.column_stack([
        np.interp(requested_times, cosmic_times, reference[:, shell])
        for shell in range(len(radius))
    ])


def integrate_shell_density_reference(initial, cosmology, scale_factors):
    """Return conservative Eulerian density from pressureless shell ODEs."""
    boundary = np.asarray(initial.mesh.boundary, dtype=float)
    radius = np.asarray(initial.mesh.coordinate, dtype=float)
    volume = np.asarray(initial.mesh.vol, dtype=float)
    shell_mass = np.asarray(initial.fluid.rho_code, dtype=float) * volume
    edge_mass = np.concatenate(([0.0], np.cumsum(shell_mass)))
    j = np.asarray(initial.fluid.specific_angular_momentum_code, dtype=float)
    edge_j = np.interp(
        edge_mass,
        np.concatenate(([0.0], np.cumsum(shell_mass))),
        np.concatenate(([0.0], j)),
    )
    cosmic_time_initial = float(
        cosmology.cosmic_time_from_supercomoving(initial.par.time)
    )
    scale_initial = float(cosmology.scale_factor(cosmic_time_initial))
    hubble_initial = float(cosmology.hubble(cosmic_time_initial))
    initial_physical_boundary = scale_initial * boundary
    initial_edge_velocity = np.interp(
        boundary, radius, np.asarray(initial.fluid.vel_code, dtype=float),
        left=0.0, right=float(np.asarray(initial.fluid.vel_code, dtype=float)[-1]),
    )
    initial_physical_velocity = (
        hubble_initial * initial_physical_boundary
        + initial_edge_velocity / scale_initial
    )
    requested_times = cosmology.t_ref * np.asarray(scale_factors, dtype=float)**1.5
    cosmic_times = np.unique(requested_times)
    physical_edges = np.empty((len(cosmic_times), len(boundary)), dtype=float)
    for edge, enclosed_mass in enumerate(edge_mass):
        def rhs(time, state):
            shell_radius, shell_velocity = state
            if shell_radius <= 0.0 or enclosed_mass <= 0.0:
                return shell_velocity, 0.0
            acceleration = (
                -cosmology.gravitational_constant * enclosed_mass / shell_radius**2
                + edge_j[edge]**2 / shell_radius**3
            )
            return shell_velocity, acceleration

        solution = solve_ivp(
            rhs,
            (cosmic_times[0], cosmic_times[-1]),
            (initial_physical_boundary[edge], initial_physical_velocity[edge]),
            t_eval=cosmic_times,
            rtol=1.0e-8, atol=1.0e-10,
            max_step=max((cosmic_times[-1] - cosmic_times[0]) / 32.0, 1.0e-8),
        )
        if not solution.success:
            raise RuntimeError(
                "density shell ODE failed at edge %d: %s"
                % (edge, solution.message)
            )
        physical_edges[:, edge] = solution.y[0]

    reference_unique = np.empty((len(cosmic_times), len(volume)), dtype=float)
    for time_index, cosmic_time in enumerate(cosmic_times):
        current_scale = float(cosmology.scale_factor(cosmic_time))
        comoving_edges = physical_edges[time_index] / current_scale
        if np.any(np.diff(comoving_edges) <= 0.0):
            raise RuntimeError("pressureless reference shells crossed")
        shell_volume = 4.0 * np.pi / 3.0 * (
            comoving_edges[1:]**3 - comoving_edges[:-1]**3
        )
        target_volume = volume
        deposited_mass = np.zeros(len(volume), dtype=float)
        for shell in range(len(shell_mass)):
            shell_inner, shell_outer = comoving_edges[shell:shell + 2]
            shell_volume_value = shell_volume[shell]
            for cell in range(len(volume)):
                overlap_inner = max(shell_inner, boundary[cell])
                overlap_outer = min(shell_outer, boundary[cell + 1])
                if overlap_outer > overlap_inner:
                    overlap_volume = 4.0 * np.pi / 3.0 * (
                        overlap_outer**3 - overlap_inner**3
                    )
                    deposited_mass[cell] += (
                        shell_mass[shell] * overlap_volume / shell_volume_value
                    )
        reference_unique[time_index] = deposited_mass / target_volume
    if len(cosmic_times) == len(requested_times):
        return reference_unique
    reference = np.empty((len(requested_times), len(volume)), dtype=float)
    for cell in range(len(volume)):
        reference[:, cell] = np.interp(
            requested_times, cosmic_times, reference_unique[:, cell]
        )
    return reference


def enclosed_radii(boundary, mass_density, volume, target_mass):
    cumulative = np.concatenate(([0.0], np.cumsum(
        np.asarray(mass_density, dtype=float) * np.asarray(volume, dtype=float)
    )))
    return np.interp(
        np.asarray(target_mass, dtype=float), cumulative,
        np.asarray(boundary, dtype=float),
    )


class InitialCondition:
    def __init__(self, initial_condition, par, rotation_factor, units, cosmology):
        count = int(par["mesh"]["grid_cells"])
        icparams = initial_condition
        cosmic_time = float(icparams["cosmic_time"])
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        hubble = float(cosmology.hubble(cosmic_time))
        tau = float(cosmology.supercomoving_time(cosmic_time))
        boundary = np.linspace(
            float(icparams["rmin"]), float(icparams["rmax"]), count + 1
        )
        radius = spherical_centers(boundary)
        volume = 4.0 * np.pi / 3.0 * (
            boundary[1:]**3 - boundary[:-1]**3
        )
        rho_background = float(cosmology.background_density(cosmic_time))
        overdensity = float(icparams["overdensity"])
        inside = radius < float(icparams["top_hat_radius"])
        rho_physical = rho_background * (1.0 + overdensity * inside)
        rho_comoving = rho_physical * scale_factor**3
        enclosed_mass = np.cumsum(rho_comoving * volume)
        physical_radius = scale_factor * radius
        specific_j = rotation_factor * np.sqrt(
            cosmology.gravitational_constant
            * enclosed_mass * physical_radius
        )
        temperature = quantity_to_value(
            icparams["tempini"], units.temperature_unit
        ) * scale_factor**2

        self.par = SimpleNamespace(
            CodeUnits=units,
            unit_system=units.unit_system,
            nogrid=count,
            noghost=int(par["mesh"]["ghost_cells"]),
            coordsys="spherical",
            time=np.array(tau),
            boxsize=np.array([float(icparams["rmax"])]),
            cosmological_expansion=True,
            supercomoving_coordinates=True,
            cosmological_gravity=True,
            selfgravity=True,
            externalgravity=False,
            cosmology=cosmology,
            cosmology_type=cosmology.type_name,
            cosmology_t_ref=cosmology.t_ref,
            cosmology_a_ref=cosmology.a_ref,
            coordinate_frame="comoving",
            time_coordinate="supercomoving",
            velocity_representation="supercomoving_peculiar",
            density_representation="comoving",
            pressure_representation="supercomoving",
            temperature_representation="supercomoving",
            gas_angular_momentum=True,
            gas_rotational_energy=True,
        )
        self.par.units = SimpleNamespace(CodeUnits=units)
        self.par.simulation = SimpleNamespace(
            current_time=np.array(tau), box_size=np.array([float(icparams["rmax"])]),
            coordinate_system="spherical",
        )
        self.par.mesh = SimpleNamespace(grid_cells=count, ghost_cells=0)
        self.par.hydrodynamics = SimpleNamespace(gamma=float(par["hydrodynamics"]["gamma"]))
        self.mesh = SimpleNamespace(
            boundary=boundary,
            coordinate=radius,
            area=4.0 * np.pi * boundary[:-1]**2,
            vol=volume,
        )
        self.fluid = SimpleNamespace(
            rho_code=rho_comoving,
            vel_code=-(scale_factor**2 * hubble * overdensity / 3.0) * radius,
            temp_code=np.full(count, temperature),
            mu=np.full(count, float(icparams["muini"])),
            specific_angular_momentum_code=specific_j,
        )


def run_case(base_par, initial_condition, label, rotation_factor, units, cosmology):
    output_dir = ROOT / base_par["output"]["directory"] / label
    output_dir.mkdir(parents=True, exist_ok=True)
    par = copy.deepcopy(base_par)
    par["simulation"] = dict(par["simulation"])
    par["simulation"]["initial_condition_filename"] = str(output_dir / "InitialCondition.hdf5")
    par["output"] = dict(par["output"])
    par["output"].update(directory=str(output_dir), savedir=str(output_dir), filename_prefix="Output")
    initial = InitialCondition(
        initial_condition, par, rotation_factor, units, cosmology
    )
    rio.writehdf5(initial, par["simulation"]["initial_condition_filename"])
    sim = Rsim(par)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.fluid.SetFluidTime(sim.par.time)
    sim.SetInitFluid()
    sim.par.cosmology = cosmology
    active = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    target_mass = np.cumsum(
        np.asarray(initial.fluid.rho_code, dtype=float)
        * np.asarray(initial.mesh.vol, dtype=float)
    )
    history = {
        "a": [], "maximum_density": [], "support": [],
        "density_profiles": [], "j_profiles": [], "total_j": [],
        "shell_radius": [],
    }

    def record(state):
        tau = float(np.asarray(state.fluid.time_code).flat[0])
        a = float(state.par.cosmology.scale_factor_from_supercomoving(tau))
        radius = np.abs(np.asarray(state.mesh.coordinate[active], dtype=float))
        j = np.asarray(state.fluid.specific_angular_momentum_code[active], dtype=float)
        enclosed = np.cumsum(
            np.asarray(state.fluid.Mass_code[active], dtype=float)
        )
        gravity = state.par.cosmology.gravitational_constant * enclosed
        valid = (radius > 0.0) & (gravity > 0.0)
        support = np.zeros_like(radius)
        support[valid] = j[valid]**2 / (gravity[valid] * radius[valid])
        history["a"].append(a)
        history["maximum_density"].append(
            float(np.max(np.asarray(state.fluid.rho_code[active], dtype=float)))
        )
        history["support"].append(float(np.max(support)))
        history["density_profiles"].append(
            np.asarray(state.fluid.rho_code[active], dtype=float).copy()
        )
        history["j_profiles"].append(j.copy())
        history["total_j"].append(
            float(np.sum(np.asarray(state.fluid.AngularMomentum_code[active], dtype=float)))
        )
        history["shell_radius"].append(enclosed_radii(
            state.mesh.boundary[active.start:active.stop + 1],
            state.fluid.rho_code[active], state.mesh.vol[active], target_mass,
        ))

    record(sim)
    final_tau = float(cosmology.supercomoving_time(float(base_par["simulation"]["final_time"])))
    sim.Evolve(final_time=final_tau, mode="hydro", history_callback=record)
    scale_factors = np.asarray(history["a"], dtype=float)
    reference_shell_radius = integrate_shell_reference(
        initial, cosmology, scale_factors
    )
    reference_density = integrate_shell_density_reference(
        initial, cosmology, scale_factors
    )
    final_filename = output_dir / "Output_final.hdf5"
    sim.fluid.SetTemperature()
    rio.writehdf5(sim, final_filename)
    np.savez(
        output_dir / "history.npz",
        a=np.asarray(history["a"], dtype=float),
        density=np.asarray(history["density_profiles"], dtype=float),
        specific_angular_momentum=np.asarray(history["j_profiles"], dtype=float),
        total_angular_momentum=np.asarray(history["total_j"], dtype=float),
        maximum_density=np.asarray(history["maximum_density"], dtype=float),
        support=np.asarray(history["support"], dtype=float),
        radius=np.asarray(sim.mesh.coordinate[active], dtype=float),
        shell_radius=np.asarray(history["shell_radius"], dtype=float),
        reference_shell_radius=reference_shell_radius,
        reference_density=reference_density,
    )
    return label, sim, history, output_dir


def main(config_filename=DEFAULT_CONFIG, nogrid_override=None,
         output_root_override=None, cfl_override=None,
         positivity_override=None):
    config = eu.load_nested_example_config(config_filename)
    runtime = config["par"]
    par = copy.deepcopy(config["par"])
    icparams = config["initial_condition"]
    if nogrid_override is not None:
        par["mesh"] = {**par["mesh"], "grid_cells": int(nogrid_override)}
    if output_root_override is not None:
        par["output"] = {**par["output"], "directory": str(output_root_override), "savedir": str(output_root_override)}
    if cfl_override is not None:
        par["hydrodynamics"] = {**par["hydrodynamics"], "CFL": float(cfl_override)}
    if positivity_override is not None:
        par["hydrodynamics"] = {**par["hydrodynamics"], "positivity_preserving": bool(positivity_override)}
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(par["gravity"]["cosmology_t_ref"]),
        a_ref=float(par["gravity"]["cosmology_a_ref"]),
    )
    cases = [
        ("nonrotating", 0.0),
        ("moderate", float(icparams["moderate_rotation_factor"])),
        ("high", float(icparams["high_rotation_factor"])),
    ]
    results = [
        run_case(par, icparams, label, factor, units, cosmology)
        for label, factor in cases
    ]
    by_label = {label: (sim, history, directory) for label, sim, history, directory in results}
    final_density = {
        label: history["maximum_density"][-1]
        for label, (_, history, _) in by_label.items()
    }
    final_support = {
        label: history["support"][-1]
        for label, (_, history, _) in by_label.items()
    }
    if not (
        final_density["nonrotating"]
        >= final_density["moderate"]
        >= final_density["high"]
    ):
        raise RuntimeError("rotation did not monotonically suppress collapse")
    if not (
        final_support["nonrotating"]
        <= final_support["moderate"]
        <= final_support["high"]
    ):
        raise RuntimeError("centrifugal support is not ordered by rotation")

    for label, (_, history, _) in by_label.items():
        total_j = np.asarray(history["total_j"], dtype=float)
        scale = max(1.0, abs(total_j[0]))
        if np.max(np.abs(total_j - total_j[0])) / scale > 1.0e-10:
            raise RuntimeError("total angular momentum is not conserved for %s" % label)

    saved_histories = {
        label: np.load(directory / "history.npz")
        for label, (_, _, directory) in by_label.items()
    }
    output_root = ROOT / par["output"]["savedir"]
    figure = output_root / "CosmologicalRotatingCollapse1D.jpg"
    plt.figure(figsize=(7, 4))
    for label in ("nonrotating", "moderate", "high"):
        data = saved_histories[label]
        line, = plt.plot(data["a"], data["maximum_density"], label=label)
        plt.plot(
            data["a"], np.max(data["reference_density"], axis=1),
            "--", color=line.get_color(),
            label="%s ODE" % label,
        )
    plt.xlabel("scale factor $a$")
    plt.ylabel("maximum comoving gas density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=200)
    plt.close()

    density_comparison_figure = (
        output_root
        / "CosmologicalRotatingCollapse1D_density_comparison.jpg"
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6), sharex="col", sharey="row",
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    for column, label in enumerate(("nonrotating", "moderate", "high")):
        axis = axes[0, column]
        difference_axis = axes[1, column]
        data = saved_histories[label]
        axis.plot(
            data["radius"], data["density"][0],
            ":", color="black", linewidth=1.5, label="initial",
        )
        axis.plot(
            data["radius"], data["density"][-1],
            label="simulation", linewidth=2.0,
        )
        axis.plot(
            data["radius"], data["reference_density"][-1],
            "--", label="pressureless ODE", linewidth=1.5,
        )
        axis.set_title(label)
        axis.grid(alpha=0.25)
        relative_difference = (
            data["density"][-1] - data["reference_density"][-1]
        ) / np.maximum(data["reference_density"][-1], 1.0e-300)
        difference_axis.plot(
            data["radius"], relative_difference,
            color="tab:purple", linewidth=1.5,
        )
        difference_axis.axhline(0.0, color="black", linewidth=0.8)
        difference_axis.set_xlabel("comoving radius $x$")
        difference_axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("comoving gas density")
    axes[1, 0].set_ylabel("relative difference")
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(density_comparison_figure, dpi=200)
    plt.close(fig)

    total_angular_figure = (
        output_root
        / "CosmologicalRotatingCollapse1D_total_angular_momentum.jpg"
    )
    plt.figure(figsize=(7, 4))
    for label in ("nonrotating", "moderate", "high"):
        data = saved_histories[label]
        plt.plot(data["a"], data["total_angular_momentum"], label=label)
    plt.xlabel("scale factor $a$")
    plt.ylabel("total gas angular momentum $\\sum J$")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(total_angular_figure, dpi=200)
    plt.close()

    shell_figure = (
        output_root
        / "CosmologicalRotatingCollapse1D_shell_ode.jpg"
    )
    shell_count = len(saved_histories["high"]["radius"])
    shell_indices = (
        int(0.2 * (shell_count - 1)),
        int(0.5 * (shell_count - 1)),
        int(0.8 * (shell_count - 1)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, label in zip(axes, ("nonrotating", "moderate", "high")):
        data = saved_histories[label]
        for shell in shell_indices:
            axis.plot(
                data["a"], data["shell_radius"][:, shell],
                label="simulation shell %d" % shell,
            )
            axis.plot(
                data["a"], data["reference_shell_radius"][:, shell],
                "--", label="ODE shell %d" % shell,
            )
        axis.set_title(label)
        axis.set_xlabel("scale factor $a$")
    axes[0].set_ylabel("comoving shell radius $x$")
    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(shell_figure, dpi=200)
    plt.close(fig)

    density_figure = output_root / "CosmologicalRotatingCollapse1D_density.jpg"
    angular_figure = output_root / "CosmologicalRotatingCollapse1D_angular_momentum.jpg"
    fig = plt.figure(figsize=(13, 3.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.08))
    plot_axes = [fig.add_subplot(grid[0, 0])]
    plot_axes.extend(fig.add_subplot(grid[0, index], sharey=plot_axes[0]) for index in (1, 2))
    colorbar_axis = fig.add_subplot(grid[0, 3])
    for axis, label in zip(plot_axes, ("nonrotating", "moderate", "high")):
        data = saved_histories[label]
        image = axis.imshow(
            np.log10(np.maximum(data["density"], 1.0e-300)),
            origin="lower", aspect="auto",
            extent=(data["radius"][0], data["radius"][-1], data["a"][0], data["a"][-1]),
        )
        axis.set_title(label)
        axis.set_xlabel("comoving radius $x$")
    plot_axes[0].set_ylabel("scale factor $a$")
    fig.colorbar(image, cax=colorbar_axis, label="$\\log_{10} \\rho_{\\rm sc}$")
    fig.savefig(density_figure, dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(13, 3.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.08))
    plot_axes = [fig.add_subplot(grid[0, 0])]
    plot_axes.extend(fig.add_subplot(grid[0, index], sharey=plot_axes[0]) for index in (1, 2))
    colorbar_axis = fig.add_subplot(grid[0, 3])
    maximum_j = max(
        np.max(np.abs(data["specific_angular_momentum"]), initial=0.0)
        for data in saved_histories.values()
    )
    for axis, label in zip(plot_axes, ("nonrotating", "moderate", "high")):
        data = saved_histories[label]
        image = axis.imshow(
            data["specific_angular_momentum"],
            origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-maximum_j, vmax=maximum_j,
            extent=(data["radius"][0], data["radius"][-1], data["a"][0], data["a"][-1]),
        )
        axis.set_title(label)
        axis.set_xlabel("comoving radius $x$")
    plot_axes[0].set_ylabel("scale factor $a$")
    fig.colorbar(image, cax=colorbar_axis, label="$j=x v_{\\phi,\\rm sc}$")
    fig.savefig(angular_figure, dpi=200)
    plt.close(fig)
    print("cosmological rotating collapse comparison passed")
    for label in ("nonrotating", "moderate", "high"):
        print(
            "%s: final max density %.8g, centrifugal support %.8g"
            % (label, final_density[label], final_support[label])
        )
    print("figure = %s" % figure)
    print("density comparison figure = %s" % density_comparison_figure)
    print("total angular-momentum figure = %s" % total_angular_figure)
    print("shell ODE figure = %s" % shell_figure)
    print("density evolution figure = %s" % density_figure)
    print("angular-momentum evolution figure = %s" % angular_figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--nogrid", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--positivity-preserving", action="store_true",
                        default=None)
    args = parser.parse_args()
    main(
        args.config,
        nogrid_override=args.nogrid,
        output_root_override=args.output_root,
        cfl_override=args.cfl,
        positivity_override=args.positivity_preserving,
    )
