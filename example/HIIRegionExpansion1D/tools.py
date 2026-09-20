"""Utilities for the early isothermal H II region expansion example."""

import glob
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
from basic_hydro_utils import make_initial_condition

import radhydropy.io as rio
import radhydropy.thermo_networks.hydrogen as rth
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, code_quantity_to_cgs


def build_initial_condition(config):
    """Build the H II initial state from direct nested configuration groups."""
    initial = config["initial_condition"]
    code_units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    grid_cells = int(config["par"]["mesh"]["grid_cells"])
    boundary_proper_cgs_cm_unyt = (
        np.linspace(
            0.0,
            initial["box_size_proper"].to_value(unyt.cm),
            grid_cells + 1,
        )
        * unyt.cm
    )
    boundary_proper_code = boundary_proper_cgs_cm_unyt.to_value(code_units.length_unit)
    density_proper_code = (
        np.ones(grid_cells)
        * initial["rho_proper"]
        .to(
            code_units.density_unit,
        )
        .value
    )
    temperature_proper_code = (
        np.ones(grid_cells)
        * initial["temperature_neutral_proper"].to(code_units.temperature_unit).value
    )
    sim = make_initial_condition(
        config,
        boundary_proper_code=boundary_proper_code,
        rho_proper_code=density_proper_code,
        vel_proper_code=np.zeros(grid_cells),
        temp_proper_code=temperature_proper_code,
        mu_dimensionless=np.ones(grid_cells),
    )
    sim.fluid.xHI = np.ones(grid_cells)
    radiation = config["par"]["radiation"]
    sim.fluid.ngamma_code = np.full(
        grid_cells,
        radiation.get("hydrogen_ngamma_initial", 0.0 / unyt.cm**3)
        .to(
            code_units.number_density_unit,
        )
        .value,
    )
    sim.fluid._refresh_runtime_state()
    return sim.par, sim.mesh, sim.fluid, sim.solver


def write_initial_condition(config):
    """Build the raw IC state and write it to ``ICfilename``."""
    par, mesh, fluid, solver = build_initial_condition(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    icfilename = config["par"]["simulation"]["initial_condition_filename"]
    Path(icfilename).unlink(missing_ok=True)
    rio.writehdf5(sim, icfilename)


def load_output_state(outputfilename, config):
    snapshot = Rsim(config["par"])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, outputfilename)
    par, mesh, fluid = snapshot.par, snapshot.mesh, snapshot.fluid
    # ``readhdf5`` restores the saved boundary and fluid state, but it does not
    # recompute the derived mesh geometry. Rebuild those cached geometric
    # fields from the loaded boundary so post-processing uses the snapshot's
    # actual coordinates instead of the constructor-time placeholders.
    boundary_proper_code = np.asarray(mesh.boundary_proper_code, dtype=float)
    if par.simulation.coordinate_system == "cartesian":
        mesh.width_proper_code = boundary_proper_code[1:] - boundary_proper_code[:-1]
        mesh.coordinate_inverse_proper_code = 1.0 / mesh.width_proper_code
        mesh.x_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
        mesh.area_proper_code = (
            np.ones(len(mesh.width_proper_code))
            * par.mesh.area_proper.to(
                code_units_obj.area_unit,
            ).value
        )
        mesh.volume_proper_code = mesh.width_proper_code * mesh.area_proper_code
    elif par.simulation.coordinate_system == "spherical":
        mesh.width_proper_code = boundary_proper_code[1:] - boundary_proper_code[:-1]
        mesh.coordinate_inverse_proper_code = 1.0 / mesh.width_proper_code
        mesh.area_proper_code = (boundary_proper_code[:-1] ** 2) * 4.0 * np.pi
        mesh.volume_proper_code = (
            np.absolute(boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
            * 4.0
            * np.pi
            / 3.0
        )
        volume_difference_proper_code = (
            boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
        )
        mesh.x_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
        nonzero_volume_difference = volume_difference_proper_code != 0.0
        mesh.x_proper_code[nonzero_volume_difference] = (
            0.75
            * (
                boundary_proper_code[1:][nonzero_volume_difference] ** 4
                - boundary_proper_code[:-1][nonzero_volume_difference] ** 4
            )
            / volume_difference_proper_code[nonzero_volume_difference]
        )
        if np.any((boundary_proper_code[:-1] < 0.0) & (boundary_proper_code[1:] > 0.0)):
            crossing = np.where(
                (boundary_proper_code[:-1] < 0.0) & (boundary_proper_code[1:] > 0.0),
            )[0]
            for ig in crossing:
                mesh.volume_proper_code[ig] = (
                    (boundary_proper_code[ig + 1] ** 3) * 4.0 * np.pi / 3.0
                )
                mesh.x_proper_code[ig] = 0.75 * boundary_proper_code[ig + 1]
                mesh.area_proper_code[ig] = 0.0
    return par, mesh, fluid


def load_labeled_density_snapshots(outputfilenames, config, output_specs):
    snapshots = []
    for index, spec in enumerate(output_specs):
        label = spec.get("label", None)
        if label is None:
            continue
        out_par, out_mesh, out_fluid = load_output_state(outputfilenames[index], config)
        config["_output_par"] = out_par
        snapshots.append(
            (
                label,
                density_snapshot(out_mesh, out_fluid, config),
            ),
        )
    return snapshots


def output_files(output_directory, output_filename_prefix):
    pattern = os.path.join(output_directory, f"{output_filename_prefix}_*.hdf5")
    return sorted(glob.glob(pattern))


def interior_slice(config):
    par = config["_output_par"]
    mesh = par.mesh
    ghost_cells = int(mesh.ghost_cells)
    grid_cells = int(mesh.grid_cells)
    return slice(ghost_cells, ghost_cells + grid_cells)


def refresh_state(mesh, fluid, config, solver):
    par = config["_output_par"]
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid, verbose=par.verbose)


def apply_piecewise_isothermal_state(sim, config):
    mesh, fluid, par, solver = sim.mesh, sim.fluid, sim.par, sim.solver
    initial_condition = config["initial_condition"]
    fluid.eos.apply_piecewise_isothermal_state(
        fluid,
        par,
        initial_condition["temperature_neutral_proper"],
        initial_condition["temperature_ionized_proper"],
    )
    config["_output_par"] = par
    refresh_state(mesh, fluid, config, solver)


def time_proper_Myr(value, code_unit_system):
    myr_in_s = (1.0 * unyt.Myr).to_value(unyt.s)
    return float(code_quantity_to_cgs(value, code_unit_system, "time_proper_cgs_s") / myr_in_s)


def print_startup_diagnostics(sim, config, initial_condition):
    """Print the main physical scales before the long run starts."""
    initial_condition = config["initial_condition"]
    config["_output_par"] = sim.par
    interior = interior_slice(config)
    rho_proper_code = np.asarray(sim.fluid.rho_proper_code[interior], dtype=float)
    np.asarray(sim.fluid.vel_proper_code[interior], dtype=float)
    np.asarray(sim.fluid.temp_proper_code[interior], dtype=float)
    np.asarray(sim.fluid.xHI[interior], dtype=float)
    ngamma_code = (
        np.asarray(sim.fluid.ngamma_code[interior], dtype=float)
        if hasattr(sim.fluid, "ngamma_code")
        else None
    )
    code_units_obj = sim.par.units.CodeUnits
    code_quantity_to_cgs(
        rho_proper_code,
        code_units_obj,
        "density_cgs_g_cm3",
    )
    ngamma_cgs = None
    if ngamma_code is not None:
        ngamma_cgs = code_quantity_to_cgs(ngamma_code, code_units_obj, "number_density_cgs_cm3")

    if ngamma_code is not None and ngamma_cgs is not None:
        boundary_cgs_cm = code_quantity_to_cgs(
            sim.mesh.boundary_proper_code[interior.start : interior.start + 2],
            code_units_obj,
            "length_cgs_cm",
        )
        inner_radius_cgs_cm = 0.5 * (boundary_cgs_cm[0] + boundary_cgs_cm[1])
        initial_condition["source_photon_rate"].to_value(1 / unyt.s) / (
            4.0 * np.pi * inner_radius_cgs_cm**2 * unyt.c.to_value(unyt.cm / unyt.s)
        )
    try:
        hydro_dt = sim.solver.GetTimeStep(sim.mesh, sim.fluid, sim.par)
        hydro_dt_s = hydro_dt.to_value(unyt.s) if hasattr(hydro_dt, "to_value") else float(hydro_dt)
    except Exception:
        hydro_dt_s = None
    try:
        source_dt, thermal_rate = sim.solver.GetSourceTimestepFast(
            sim.mesh,
            sim.fluid,
            sim.par,
            sim.par.timestep.dtmax,
        )
        source_dt_s = (
            source_dt.to_value(unyt.s) if hasattr(source_dt, "to_value") else float(source_dt)
        )
        if hydro_dt_s is not None and source_dt_s > 0.0:
            pass
        if thermal_rate is not None:
            pass
    except Exception:
        pass


def make_logging_step_backend(sim, config, max_logged_steps=5):
    """Wrap the isothermal step backend with a short startup trace."""
    base_step_backend = make_piecewise_isothermal_step_backend(sim, config)
    state = {"count": 0}
    config["_output_par"] = sim.par
    interior = interior_slice(config)

    def step_backend(dt=None, mode="hydro_sources", advect_chemistry=True):
        step_index = state["count"]
        should_log = step_index < max_logged_steps
        if should_log:
            pass
        result = base_step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
        )
        if should_log:
            vel_proper_code = np.asarray(sim.fluid.vel_proper_code[interior], dtype=float)
            np.asarray(sim.fluid.rho_proper_code[interior], dtype=float)
            np.asarray(sim.fluid.xHI[interior], dtype=float)
            np.max(np.abs(vel_proper_code)) / 1.0e5
            config["_output_par"] = sim.par
            ionization_front_position(sim.mesh, sim.fluid, config)
            if step_index + 1 == max_logged_steps:
                pass
        state["count"] += 1
        return result

    return step_backend


def make_piecewise_isothermal_step_backend(sim, config):
    def step_backend(dt=None, mode="hydro_sources", advect_chemistry=True):
        result = sim.Step(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
        )
        apply_piecewise_isothermal_state(
            sim,
            config,
        )
        return result

    return step_backend


def ionization_front_position(mesh, fluid, config, ionized_fraction=0.5):
    config["_output_par"]
    interior = interior_slice(config)
    boundary_proper_pc = mesh.boundary_radarray.to(unyt.pc).value
    radius_proper_pc = 0.5 * (boundary_proper_pc[:-1] + boundary_proper_pc[1:])[interior]
    xHII = 1.0 - np.asarray(fluid.xHI[interior], dtype=float)

    ionized = xHII >= ionized_fraction
    if not np.any(ionized):
        return 0.0
    if np.all(ionized):
        return radius_proper_pc[-1]

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHII[left]
    x_right = xHII[right]
    if x_right == x_left:
        return radius_proper_pc[left]

    weight = (ionized_fraction - x_left) / (x_right - x_left)
    return radius_proper_pc[left] + weight * (radius_proper_pc[right] - radius_proper_pc[left])


def append_history(history, mesh, fluid, config):
    par = config["_output_par"]
    time_proper_Myr = (
        (np.asarray(fluid.time_proper_code).flat[0] * par.units.CodeUnits.time_unit)
        .to(unyt.Myr)
        .value
    )
    history["time_proper_Myr"].append(float(time_proper_Myr))
    history["front_radius_proper_pc"].append(ionization_front_position(mesh, fluid, config))


def load_history_from_outputs(outputfilenames, config):
    history = {
        "time_proper_Myr": [],
        "front_radius_proper_pc": [],
    }
    for outputfilename in outputfilenames:
        par, mesh, fluid = load_output_state(outputfilename, config)
        config["_output_par"] = par
        append_history(history, mesh, fluid, config)
    return history


def density_snapshot(mesh, fluid, config):
    par = config["_output_par"]
    interior = interior_slice(config)
    boundary_proper_pc = mesh.boundary_radarray.to(unyt.pc).value
    radius_proper_pc = 0.5 * (boundary_proper_pc[:-1] + boundary_proper_pc[1:])[interior]
    ngamma_radarray = fluid.ngamma_radarray[interior]
    if ngamma_radarray.ndim > 1:
        ngamma_radarray = np.sum(ngamma_radarray, axis=0)
    time_proper_Myr = (
        (np.asarray(fluid.time_proper_code).flat[0] * par.units.CodeUnits.time_unit)
        .to(unyt.Myr)
        .value
    )
    return {
        "time_proper_Myr": float(time_proper_Myr),
        "radius_proper_pc": np.asarray(radius_proper_pc, dtype=float).copy(),
        "rho_proper_cgs_g_cm3": fluid.rho_radarray[interior]
        .to(
            unyt.g / unyt.cm**3,
        )
        .value.copy(),
        "radiation_density_cgs_cm3": ngamma_radarray.to(
            1.0 / unyt.cm**3,
        ).value.copy(),
    }


def front_radius_at_time(history, time_proper_code):
    time_proper_Myr = np.asarray(history["time_proper_Myr"])
    front_radius_pc = np.asarray(history["front_radius_proper_pc"])
    target_time_myr = time_proper_code.to_value(unyt.Myr)
    if time_proper_Myr.size == 0:
        raise ValueError("history is empty")
    tol = max(1.0e-12 * max(1.0, np.max(np.abs(time_proper_Myr)), abs(target_time_myr)), 1.0e-30)
    if target_time_myr < time_proper_Myr[0] - tol or target_time_myr > time_proper_Myr[-1] + tol:
        raise ValueError("requested time is outside the recorded history")
    target_time_myr = float(np.clip(target_time_myr, time_proper_Myr[0], time_proper_Myr[-1]))
    return np.interp(target_time_myr, time_proper_Myr, front_radius_pc) * unyt.pc


def stromgren_radius(config):
    config = config["initial_condition"]
    nH = rth._cgs_hydrogen_number_density(
        config["rho_proper"].to_value(unyt.g / unyt.cm**3),
        hydrogen_mass_fraction=1.0,
    ) * (1.0 / unyt.cm**3)
    radius_stromgren_proper_unyt = (
        3.0 * config["source_photon_rate"] / (4.0 * np.pi * config["alpha_B_coefficient"] * nH**2)
    ) ** (1.0 / 3.0)
    return radius_stromgren_proper_unyt.to(unyt.pc)


def neutral_sound_speed(config):
    config = config["initial_condition"]
    return np.sqrt(
        unyt.kb * config["temperature_neutral_proper"] / unyt.mp,
    ).to(unyt.cm / unyt.s)


def stagnation_radius(config):
    initial_condition = config["initial_condition"]
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = initial_condition["ionized_sound_speed"].to(unyt.cm / unyt.s)
    return (
        (ionized_sound_speed / neutral_sound_speed(config)) ** (4.0 / 3.0) * radius_stromgren
    ).to(unyt.pc)


def spitzer_radius(time_proper_code, config):
    initial_condition = config["initial_condition"]
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = initial_condition["ionized_sound_speed"].to(unyt.cm / unyt.s)
    factor = 1.0 + 7.0 * ionized_sound_speed * time_proper_code.to(unyt.s) / (
        4.0 * radius_stromgren.to(unyt.cm)
    )
    return (radius_stromgren * factor ** (4.0 / 7.0)).to(unyt.pc)


def hosokawa_inutsuka_radius(time_proper_code, config):
    initial_condition = config["initial_condition"]
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = initial_condition["ionized_sound_speed"].to(unyt.cm / unyt.s)
    factor = 1.0 + 7.0 * np.sqrt(4.0 / 3.0) * ionized_sound_speed * time_proper_code.to(unyt.s) / (
        4.0 * radius_stromgren.to(unyt.cm)
    )
    return (radius_stromgren * factor ** (4.0 / 7.0)).to(unyt.pc)


def save_front_plot(history, config, figure_filename):
    initial_condition = config["initial_condition"]
    time_proper_unyt = np.asarray(history["time_proper_Myr"]) * unyt.Myr
    time_proper_Myr = time_proper_unyt.to_value(unyt.Myr)
    front_radius_pc = np.asarray(history["front_radius_proper_pc"])
    stromgren_radius_pc = stromgren_radius(config).to_value(unyt.pc)
    radius_spitzer_pc = spitzer_radius(time_proper_unyt, config).to_value(unyt.pc)
    radius_hosokawa_inutsuka_pc = hosokawa_inutsuka_radius(
        time_proper_unyt,
        config,
    ).to_value(unyt.pc)
    show_stagnation_radius = config.get("example", {}).get(
        "show_stagnation_radius",
        False,
    )
    if show_stagnation_radius:
        radius_stagnation_pc = stagnation_radius(config).to_value(unyt.pc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time_proper_Myr,
        front_radius_pc,
        color="tab:blue",
        lw=2.0,
        label=r"RadHydropy $x_{\rm HII}=0.5$",
    )
    ax.plot(
        time_proper_Myr,
        radius_spitzer_pc,
        color="tab:orange",
        lw=1.8,
        ls="--",
        label=(
            r"Spitzer, $c_i={:.2f}$ km s$^{{-1}}$".format(
                initial_condition["ionized_sound_speed"].to_value(unyt.km / unyt.s)
            )
        ),
    )
    ax.plot(
        time_proper_Myr,
        radius_hosokawa_inutsuka_pc,
        color="tab:green",
        lw=1.8,
        ls=":",
        label="Hosokawa-Inutsuka",
    )
    ax.axhline(
        stromgren_radius_pc,
        color="black",
        lw=1.4,
        ls="--",
        label=r"$R_{\rm S}$",
    )
    if show_stagnation_radius:
        ax.axhline(
            radius_stagnation_pc,
            color="tab:red",
            lw=1.6,
            ls="-.",
            label=r"$R_{\rm stag}$",
        )
    ax.set_xlabel("Time [Myr]")
    ax.set_ylabel("Ionization-front radius [pc]")
    ax.set_xlim(0.0, config["initial_condition"]["final_time"].to_value(unyt.Myr))
    radius_limits = (
        1.05 * np.max(front_radius_pc),
        1.05 * np.max(radius_spitzer_pc),
        1.05 * np.max(radius_hosokawa_inutsuka_pc),
        1.1 * stromgren_radius_pc,
    )
    if show_stagnation_radius:
        radius_limits += (1.1 * radius_stagnation_pc,)
    ax.set_ylim(0.0, max(radius_limits))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_density_profile_plot(snapshot, config, figure_filename):
    time_proper_unyt = snapshot["time_proper_Myr"] * unyt.Myr
    radius_proper_pc = np.asarray(snapshot["radius_proper_pc"])
    rho_proper_cgs_g_cm3 = np.asarray(snapshot["rho_proper_cgs_g_cm3"])
    radiation_density_cgs_cm3 = np.asarray(snapshot["radiation_density_cgs_cm3"])
    spitzer_radius_pc = spitzer_radius(time_proper_unyt, config).to_value(unyt.pc)
    hosokawa_inutsuka_radius_pc = hosokawa_inutsuka_radius(
        time_proper_unyt,
        config,
    ).to_value(unyt.pc)
    show_stagnation_radius = config.get("example", {}).get(
        "show_stagnation_radius",
        False,
    )
    if show_stagnation_radius:
        radius_stagnation_pc = stagnation_radius(config).to_value(unyt.pc)

    fig, (ax, radiation_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.2),
        sharex=True,
    )
    ax.plot(
        radius_proper_pc,
        rho_proper_cgs_g_cm3,
        color="tab:blue",
        lw=2.0,
        label="RadHydropy",
    )
    ax.axvline(
        spitzer_radius_pc,
        color="tab:orange",
        lw=1.8,
        ls="--",
        label="Spitzer",
    )
    ax.axvline(
        hosokawa_inutsuka_radius_pc,
        color="tab:green",
        lw=1.8,
        ls=":",
        label="Hosokawa-Inutsuka",
    )
    if show_stagnation_radius:
        ax.axvline(
            radius_stagnation_pc,
            color="tab:red",
            lw=1.6,
            ls="-.",
            label=r"$R_{\rm stag}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Radius [pc]")
    ax.set_ylabel(r"Density [g cm$^{-3}$]")
    ax.set_title("Density profile at {:.3f} Myr".format(snapshot["time_proper_Myr"]))
    ax.set_xlim(0.0, config["initial_condition"]["box_size_proper"].to_value(unyt.pc))
    positive_density = rho_proper_cgs_g_cm3[rho_proper_cgs_g_cm3 > 0.0]
    if positive_density.size:
        ymin = 10.0 ** np.floor(np.log10(0.8 * np.min(positive_density)))
        ymax = 10.0 ** np.ceil(np.log10(1.2 * np.max(positive_density)))
        ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    radiation_ax.plot(
        radius_proper_pc,
        np.where(radiation_density_cgs_cm3 > 0.0, radiation_density_cgs_cm3, np.nan),
        color="tab:purple",
        lw=2.0,
        label="RadHydropy",
    )
    radiation_ax.axvline(spitzer_radius_pc, color="tab:orange", lw=1.8, ls="--")
    radiation_ax.axvline(
        hosokawa_inutsuka_radius_pc,
        color="tab:green",
        lw=1.8,
        ls=":",
    )
    if show_stagnation_radius:
        radiation_ax.axvline(radius_stagnation_pc, color="tab:red", lw=1.6, ls="-.")
    radiation_ax.set_yscale("log")
    radiation_ax.set_xlabel("Radius [pc]")
    radiation_ax.set_ylabel(r"Photon density [cm$^{-3}$]")
    positive_radiation = radiation_density_cgs_cm3[radiation_density_cgs_cm3 > 0.0]
    if positive_radiation.size:
        ymin = 10.0 ** np.floor(np.log10(0.8 * np.min(positive_radiation)))
        ymax = 10.0 ** np.ceil(np.log10(1.2 * np.max(positive_radiation)))
        radiation_ax.set_ylim(ymin, ymax)
    radiation_ax.grid(True, alpha=0.25)
    radiation_ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_density_profile_plots(snapshots, config, figure_filenames):
    if len(snapshots) != len(figure_filenames):
        raise ValueError("density snapshots and figure filenames differ in length")
    for snapshot, figure_filename in zip(snapshots, figure_filenames, strict=False):
        save_density_profile_plot(snapshot, config, figure_filename)
