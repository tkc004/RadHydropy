"""Live dark-matter-only companion for the cosmological virial-shock test."""

import argparse
import copy
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

from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from example_utils import load_nested_example_config
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.units import _gravitational_constant_code
import tools_lambda_cdm as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_dark_matter_correlation_z100_lambda_cdm.yaml"
)


def load_correlation_table(config_filename, config):
    filename = config.get("example", {}).get(
        "linear_correlation_table_filename"
    )
    if not filename:
        return None
    filename = Path(filename)
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def run_lagrangian_top_hat(config):
    """Calibrate one finite top-hat mass before shell crossing."""
    par = config["par"]
    example = config["example"]
    initial_condition = config["initial_condition"]
    code_unit_system = config["_code_unit_system"]
    cosmology = config["_cosmology"]
    target_mass = quantity_to_value(
        initial_condition["target_halo_mass"], code_unit_system.mass_unit
    )
    delta_i = float(initial_condition["initial_overdensity"])
    initial = quantity_to_value(initial_condition["time_cosmic"], code_unit_system.time_unit)
    final = quantity_to_value(par["simulation"]["final_time"], units.time_unit)
    a_initial = float(cosmology.scale_factor(initial))
    h_initial = float(cosmology.hubble(initial))
    rho_comoving = float(cosmology.background_density(initial)) * a_initial**3
    radius_comoving_code = et.radius_perturbation_comoving_code(config)
    vel_supercomoving_code = -a_initial**2 * h_initial * delta_i * radius_comoving_code / 3.0
    angular_momentum = float(initial_condition.get("dm_specific_angular_momentum", 0.0))
    g_code = _gravitational_constant_code(code_unit_system)
    tau = float(cosmology.supercomoving_time(initial))
    final_tau = float(cosmology.supercomoving_time(final))
    timestep = float(
        example.get(
            "dm_only_calibration_timestep",
            example.get("dm_only_supercomoving_timestep", 0.0005),
        )
    )
    history_time = [initial]
    history_radius = [a_initial * radius_comoving_code]
    turnaround = None
    virial_crossing = None
    previous_velocity = float(
        h_initial * a_initial * radius_comoving_code + vel_supercomoving_code / a_initial
    )
    previous_time = initial
    previous_radius = a_initial * radius_comoving_code

    is_lcdm = getattr(cosmology, "type_name", "") == "lambda_cdm"
    if is_lcdm:
        collapse_time = turnaround_time = analytic_rta = analytic_rvir = None
    else:
        collapse_time = initial * (1.686 / delta_i) ** 1.5
        turnaround_time = 0.5 * collapse_time
        rho_ta = float(cosmology.background_density(turnaround_time))
        rho_vir = float(cosmology.background_density(collapse_time))
        analytic_rta = (target_mass / ((4.0 * np.pi / 3.0) * (9.0 * np.pi**2 / 16.0) * rho_ta)) ** (1.0 / 3.0)
        analytic_rvir = (target_mass / ((4.0 * np.pi / 3.0) * (18.0 * np.pi**2) * rho_vir)) ** (1.0 / 3.0)

    while tau < final_tau and (analytic_rvir is None or virial_crossing is None):
        dt = min(timestep, final_tau - tau)
        cosmic_start = float(cosmology.cosmic_time_from_supercomoving(tau))
        a_start = float(cosmology.scale_factor(cosmic_start))
        tau_end = tau + dt
        cosmic_end = float(cosmology.cosmic_time_from_supercomoving(tau_end))
        a_end = float(cosmology.scale_factor(cosmic_end))

        def acceleration(r, a):
            background_mass = 4.0 * np.pi / 3.0 * rho_comoving * r**3
            gravity = -g_code * a * (target_mass - background_mass) / max(r**2, 1.0e-30)
            centrifugal = angular_momentum**2 / max(r**3, 1.0e-30)
            return gravity + centrifugal

        velocity_half = vel_supercomoving_code + 0.5 * dt * acceleration(radius_comoving_code, a_start)
        radius_new = radius_comoving_code + dt * velocity_half
        vel_supercomoving_code = velocity_half + 0.5 * dt * acceleration(radius_new, a_end)
        radius_comoving_code = radius_new
        if radius_comoving_code <= 0.0:
            raise RuntimeError("top-hat boundary reached the pressureless singularity before virial crossing")
        tau = tau_end
        physical_velocity = float(
            cosmology.hubble(cosmic_end) * a_end * radius_comoving_code + vel_supercomoving_code / a_end
        )
        physical_radius = a_end * radius_comoving_code
        if previous_velocity > 0.0 >= physical_velocity:
            fraction = previous_velocity / (previous_velocity - physical_velocity)
            turnaround = (
                previous_time + fraction * (cosmic_end - previous_time),
                previous_radius + fraction * (physical_radius - previous_radius),
            )
        if analytic_rvir is not None and turnaround is not None and physical_radius <= analytic_rvir and virial_crossing is None:
            virial_crossing = (cosmic_end, physical_radius)
        previous_velocity = physical_velocity
        previous_time = cosmic_end
        previous_radius = physical_radius
        history_time.append(cosmic_end)
        history_radius.append(physical_radius)

    figure = Path(par["output"]["directory"]) / "CosmologicalTopHatDarkMatterOnly.jpg"
    Path(par["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history_time, history_radius, label="numerical top-hat radius")
    if analytic_rta is not None:
        plt.axvline(turnaround_time, color="tab:red", ls="--", label="analytic turnaround time")
        plt.axhline(analytic_rta, color="tab:green", ls=":", label="analytic turnaround radius")
    plt.xlabel("cosmic time [code units]")
    plt.ylabel("proper radius [kpc]")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure, dpi=200)
    plt.close()
    print("Lagrangian top-hat DM-only calibration passed")
    print("target halo mass = %.8g code masses (%.8g Msun)" % (target_mass, target_mass * code_unit_system.mass_in_cgs / 1.98847e33))
    if analytic_rta is not None:
        print("analytic turnaround: t=%.8g, r=%.8g kpc" % (turnaround_time, analytic_rta))
        print("analytic virial: t=%.8g, r=%.8g kpc" % (collapse_time, analytic_rvir))
    if turnaround is None:
        raise RuntimeError("Lagrangian top-hat did not reach turnaround")
    print("numerical turnaround: t=%.8g, r=%.8g kpc" % turnaround)
    if analytic_rvir is not None and virial_crossing is None:
        raise RuntimeError("Lagrangian top-hat did not reach the analytic virial radius")
    if virial_crossing is not None:
        print("numerical virial-radius crossing: t=%.8g, r=%.8g kpc, M=%.8g code" % (
            virial_crossing[0], virial_crossing[1], target_mass
        ))
    print("figure = %s" % figure)


def run_live_shell_density_profiles(config):
    """Evolve a gas-free full-matter top-hat and save density snapshots."""
    par = config["par"]
    example = config["example"]
    initial_condition = config["initial_condition"]
    code_unit_system = config["_code_unit_system"]
    cosmology = config["_cosmology"]
    dm_ic = copy.deepcopy(initial_condition)
    # With gas removed, the collisionless shells represent the full matter
    # density.  Keeping the configured baryon fraction here would weaken the
    # gravitational normalization by f_DM.
    dm_ic["baryon_fraction"] = 0.0
    dm_ic["dark_matter_shells"] = int(
        example.get("dm_only_shells", max(1024, int(initial_condition["dark_matter_shells"])))
    )
    shell_config = dict(config)
    shell_config["initial_condition"] = dm_ic
    shells = et.make_dark_matter(shell_config)
    target_times = np.asarray(
        example.get(
            "dm_only_density_times",
            [quantity_to_value(initial_condition["time_cosmic"], code_unit_system.time_unit), 4.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        ),
        dtype=float,
    )
    initial = quantity_to_value(initial_condition["time_cosmic"], units.time_unit)
    final = quantity_to_value(par["simulation"]["final_time"], units.time_unit)
    target_times = np.unique(np.clip(target_times, initial, final))
    tau = float(cosmology.supercomoving_time(initial))
    final_tau = float(cosmology.supercomoving_time(final))
    target_tau = np.asarray(cosmology.supercomoving_time(target_times), dtype=float)
    timestep = float(example.get("dm_only_supercomoving_timestep", 0.0005))
    profiles = []
    virial_radii = []
    next_snapshot = 0

    def virial_threshold(time_cosmic_code):
        """Return the LCDM spherical-collapse virial density threshold."""
        if getattr(cosmology, "type_name", "") != "lambda_cdm":
            return 200.0 * float(cosmology.background_density(time_cosmic_code))
        scale_factor = float(cosmology.scale_factor(time_cosmic_code))
        hubble = float(cosmology.hubble(time_cosmic_code))
        hubble_ref = float(cosmology.hubble(cosmology.t_ref))
        omega_m_z = cosmology.omega_m * (cosmology.a_ref / scale_factor) ** 3 / (
            hubble / hubble_ref
        ) ** 2
        x = omega_m_z - 1.0
        delta_vir = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2
        critical_density = 3.0 * hubble**2 / (
            8.0 * np.pi * cosmology.gravitational_constant
        )
        return delta_vir * critical_density

    def save_profile(time_cosmic_code):
        a = float(cosmology.scale_factor(time_cosmic_code))
        order = np.argsort(shells.radius)
        radius_comoving_code = a * np.asarray(shells.radius[order], dtype=float)
        mass_comoving_code = np.asarray(shells.mass[order], dtype=float)
        # Cold shell collapse can carry shells through the coordinate origin
        # after crossing.  They remain part of the enclosed mass; represent
        # crossed/central material at a small positive radius for logarithmic
        # density binning instead of silently dropping it from the profile.
        radius_comoving_code = np.abs(np.nan_to_num(radius_comoving_code, nan=0.0, posinf=0.0, neginf=0.0))
        radius_comoving_code = np.maximum(radius_comoving_code, 1.0e-8)
        core_mass = float(getattr(shells, "central_core_mass", 0.0))
        core_radius = a * float(getattr(shells, "central_core_radius", 0.0))
        profiles.append((float(time_cosmic_code), radius_comoving_code, mass_comoving_code, core_mass, core_radius))
        # Include the absorbed unresolved-core mass when locating r200.  The
        # profile bins already include this same mass, so the overdensity
        # marker must use the identical enclosed-mass definition.
        cumulative_mass = core_mass + np.cumsum(mass_comoving_code)
        mean_density = cumulative_mass / (
            4.0 * np.pi / 3.0 * np.maximum(radius_comoving_code, 1.0e-30) ** 3
        )
        threshold = virial_threshold(time_cosmic_code)
        candidates = np.flatnonzero(mean_density >= threshold)
        if candidates.size:
            index = int(candidates[-1])
            virial_radii.append(float(radius_comoving_code[index]))
        else:
            virial_radii.append(float("nan"))

    while next_snapshot < target_times.size and target_tau[next_snapshot] <= tau + 1.0e-12:
        save_profile(target_times[next_snapshot])
        next_snapshot += 1

    while tau < final_tau - 1.0e-12:
        dt = min(timestep, final_tau - tau)
        cosmic_start = float(cosmology.cosmic_time_from_supercomoving(tau))
        cosmic_end = float(cosmology.cosmic_time_from_supercomoving(tau + dt))
        a_start = float(cosmology.scale_factor(cosmic_start))
        a_end = float(cosmology.scale_factor(cosmic_end))
        rho_comoving = float(cosmology.background_density(cosmic_start)) * a_start**3
        background = lambda radius_comoving_code, rho_comoving_code=rho_comoving: (
            4.0 * np.pi / 3.0 * rho_comoving_code * np.asarray(radius_comoving_code, dtype=float) ** 3
        )
        shells.step(
            dt,
            crossing_safety_factor=float(
                example.get("dark_matter_crossing_safety_factor", 0.5)
            ),
            background_enclosed_mass=background,
            scale_factor=a_start,
            scale_factor_end=a_end,
            cosmological=True,
            # The softened core is an additional enclosed mass; it must not
            # replace the self-gravity of the live shells outside it.
            include_shell_mass_with_fixed=True,
        )
        tau += dt
        while next_snapshot < target_times.size and target_tau[next_snapshot] <= tau + 1.0e-12:
            save_profile(target_times[next_snapshot])
            next_snapshot += 1

    times = np.asarray([item[0] for item in profiles])
    shell_radii = [item[1] for item in profiles]
    shell_masses = [item[2] for item in profiles]
    core_masses = np.asarray([item[3] for item in profiles])
    core_radii = np.asarray([item[4] for item in profiles])
    bin_count = int(example.get("dm_density_bins", 128))
    bin_min = max(1.0e-8, min(np.min(radius_comoving_code) for radius_comoving_code in shell_radii) * 0.9)
    bin_max = max(np.max(radius_comoving_code) for radius_comoving_code in shell_radii) * 1.1
    bin_edges = np.geomspace(bin_min, bin_max, bin_count + 1)
    bin_radii = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_volumes = 4.0 * np.pi / 3.0 * np.diff(bin_edges**3)
    densities = []
    for radius_comoving_code, mass, core_mass, core_radius in zip(
        shell_radii, shell_masses, core_masses, core_radii
    ):
        mass_in_bin, _ = np.histogram(radius_comoving_code, bins=bin_edges, weights=mass)
        if core_mass > 0.0 and core_radius > 0.0:
            core_bin = int(np.searchsorted(bin_edges, core_radius, side="right") - 1)
            if 0 <= core_bin < mass_in_bin.size:
                mass_in_bin[core_bin] += core_mass
        rho_comoving_code = mass_in_bin / np.maximum(bin_volumes, 1.0e-30)
        densities.append(np.where(mass_in_bin > 0.0, rho_comoving_code, np.nan))
    densities = np.asarray(densities)
    virial_radii = np.asarray(virial_radii)
    scale_factors = np.asarray([float(cosmology.scale_factor(time_cosmic_code)) for time_cosmic_code in times])
    comoving_bin_radii = bin_radii[None, :] / scale_factors[:, None]
    target_mass = quantity_to_value(
        initial_condition["target_halo_mass"], units.mass_unit
    )
    analytic_threshold = np.asarray(
        [virial_threshold(time_cosmic_code) for time_cosmic_code in times], dtype=float
    )
    analytic_rvir = (
        target_mass / ((4.0 * np.pi / 3.0) * analytic_threshold)
    ) ** (1.0 / 3.0)
    output_dir = Path(par["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "CosmologicalDarkMatterOnlyDensityProfiles.npz"
    figure = output_dir / "CosmologicalDarkMatterOnlyDensityProfiles.jpg"
    np.savez(
        data_file,
        time_cosmic_Gyr=times,
        radius_proper_kpc=bin_radii,
        radius_comoving_kpc=comoving_bin_radii,
        bin_edges_kpc=bin_edges,
        rho_comoving_code=densities,
        central_core_mass=core_masses,
        central_core_radius_kpc=core_radii,
        rvir_kpc=virial_radii,
        analytic_rvir_kpc=analytic_rvir,
    )

    plt.figure(figsize=(7.0, 5.0))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(times)))
    for time_cosmic_code, rho_comoving_code, rvir, comoving_radius, color in zip(
        times, densities, virial_radii, comoving_bin_radii, colors
    ):
        valid = np.isfinite(rho_comoving_code) & (rho_comoving_code > 0.0)
        plt.loglog(comoving_radius[valid], rho_comoving_code[valid], color=color, lw=1.6,
                   label="t = %.1f" % time_cosmic_code)
        if np.isfinite(rvir):
            scale_factor = float(cosmology.scale_factor(time_cosmic_code))
            plt.axvline(rvir / scale_factor, color=color, ls="--", lw=1.1, alpha=0.8)
    plt.xlabel("comoving radius [kpc]")
    plt.ylabel(r"dark-matter density [code mass / kpc$^3$]")
    plt.title("Live dark-matter-only density evolution")
    plt.grid(alpha=0.25, which="both")
    plt.plot([], [], color="0.25", ls="--", label=r"$r_{\rm vir}$ (LCDM $\Delta_{\rm vir}$)")
    plt.legend(title="cosmic time [Gyr]", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure, dpi=200)
    plt.close()
    print("dark-matter density figure = %s" % figure)
    print("dark-matter density data = %s" % data_file)

    radius_figure = output_dir / "CosmologicalDarkMatterOnlyVirialRadii.jpg"
    plt.figure(figsize=(7.0, 5.0))
    finite = np.isfinite(virial_radii) & (virial_radii > 0.0)
    plt.plot(
        times[finite], virial_radii[finite], "o-", color="tab:blue",
        label=r"simulation $r_{\rm vir}$ (LCDM $\Delta_{\rm vir}$)",
    )
    plt.plot(
        times[finite], analytic_rvir[finite], "--", color="tab:orange",
        label="LCDM spherical-collapse virial radius",
    )
    plt.xlabel("cosmic time [Gyr]")
    plt.ylabel("proper virial radius [kpc]")
    plt.title("Analytic and simulated virial-radius evolution")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(radius_figure, dpi=200)
    plt.close()
    print("virial-radius comparison figure = %s" % radius_figure)


def main(config_filename=DEFAULT_CONFIG):
    config = load_nested_example_config(config_filename)

    initial_condition = config["initial_condition"]
    gravity = config["par"]["gravity"]
    units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    if gravity.get("cosmology_type") in ("lambda_cdm", "LambdaCDM", "lcdm"):
        cosmology = LambdaCDM.from_code_units(
            units, t_ref=quantity_to_value(gravity["cosmology_t_ref"], units.time_unit),
            a_ref=float(gravity["cosmology_a_ref"]),
            omega_m=float(gravity["cosmology_omega_m"]),
            omega_lambda=float(gravity["cosmology_omega_lambda"]),
            hubble_ref=float(gravity["cosmology_hubble_ref"]),
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(
            units, t_ref=quantity_to_value(gravity["cosmology_t_ref"], units.time_unit),
            a_ref=float(gravity["cosmology_a_ref"]),
        )
    example = config["example"]
    correlation_table = load_correlation_table(config_filename, config)
    config["_code_unit_system"] = units
    config["_cosmology"] = cosmology
    config["_correlation_table"] = correlation_table
    run_lagrangian_top_hat(config)
    run_live_shell_density_profiles(config)
    return
    dm_ic = copy.deepcopy(initial_condition)
    dm_ic["dark_matter_shells"] = int(
        example.get("dm_only_shells", max(1024, int(initial_condition["dark_matter_shells"])))
    )
    dead_config = dict(config)
    dead_config["initial_condition"] = dm_ic
    dead_config["_code_unit_system"] = units
    dead_config["_cosmology"] = cosmology
    shells = et.make_dark_matter(dead_config)
    dm_fraction = 1.0 - float(initial_condition["baryon_fraction"])
    initial = quantity_to_value(initial_condition["time_cosmic"], units.time_unit)
    final = quantity_to_value(config["par"]["simulation"]["final_time"], units.time_unit)
    time_cosmic_code = float(cosmology.supercomoving_time(initial))
    final_tau = float(cosmology.supercomoving_time(final))
    timestep = float(example.get("dm_only_supercomoving_timestep", 0.002))
    target_mass = float(initial_condition["target_halo_mass"])
    target_dm_mass = target_mass * (1.0 - float(initial_condition["baryon_fraction"]))
    delta_i = float(initial_condition["initial_overdensity"])
    delta_c = 1.686
    collapse_time = initial * (delta_c / delta_i) ** 1.5
    turnaround_time = 0.5 * collapse_time
    rho_comoving = float(cosmology.background_density(initial)) * float(
        cosmology.scale_factor(initial)
    ) ** 3
    rho_ta = float(cosmology.background_density(turnaround_time))
    rho_vir = float(cosmology.background_density(collapse_time))
    analytic_rta = (target_mass / ((4.0 * np.pi / 3.0) * (9.0 * np.pi**2 / 16.0) * rho_ta)) ** (1.0 / 3.0)
    analytic_rvir = (target_mass / ((4.0 * np.pi / 3.0) * (18.0 * np.pi**2) * rho_vir)) ** (1.0 / 3.0)
    history_time = [initial]
    history_inner_radius = [float(shells.radius[0])]
    history_turnaround_radius = [np.nan]
    history_turnaround_mass = [np.nan]
    target_turnaround = None
    initial_a = float(cosmology.scale_factor(initial))
    initial_h = float(cosmology.hubble(initial))
    initial_index = int(np.searchsorted(np.cumsum(shells.mass), target_dm_mass, side="left"))
    initial_index = min(initial_index, shells.number_of_shells - 1)
    previous_target_velocity = float(
        initial_h * initial_a * shells.radius[initial_index]
        + shells.velocity[initial_index] / initial_a
    )
    previous_target_radius = float(initial_a * shells.radius[initial_index])
    previous_target_time = initial
    steps = 0
    while time_cosmic_code < final_tau:
        dt = min(timestep, final_tau - time_cosmic_code)
        cosmic_start = float(cosmology.cosmic_time_from_supercomoving(time_cosmic_code))
        scale_start = float(cosmology.scale_factor(cosmic_start))
        rho_start = float(cosmology.background_density(cosmic_start)) * scale_start**3
        background = lambda radius_comoving_code: (
            4.0 * np.pi / 3.0 * rho_start * dm_fraction * np.asarray(radius_comoving_code)**3
        )
        time_end = time_cosmic_code + dt
        cosmic_end = float(cosmology.cosmic_time_from_supercomoving(time_end))
        scale_end = float(cosmology.scale_factor(cosmic_end))
        shells.step(
            dt,
            crossing_safety_factor=float(example.get("dark_matter_crossing_safety_factor", 0.5)),
            background_enclosed_mass=background,
            scale_factor=scale_start,
            scale_factor_end=scale_end,
            cosmological=True,
        )
        time_cosmic_code = time_end
        steps += 1
        history_time.append(cosmic_end)
        history_inner_radius.append(float(shells.radius[0]))
        physical_radius = scale_end * shells.radius
        physical_velocity = (
            float(cosmology.hubble(cosmic_end)) * scale_end * shells.radius
            + shells.velocity / scale_end
        )
        target_index = int(np.searchsorted(np.cumsum(shells.mass), target_dm_mass, side="left"))
        target_index = min(target_index, shells.number_of_shells - 1)
        target_velocity = float(physical_velocity[target_index])
        # A Lagrangian shell changes from Hubble expansion to infall at
        # turnaround: v_r goes from positive to negative.
        if previous_target_velocity > 0.0 >= target_velocity and target_turnaround is None:
            fraction = previous_target_velocity / (previous_target_velocity - target_velocity)
            target_radius = float(
                (1.0 - fraction) * previous_target_radius
                + fraction * physical_radius[target_index]
            )
            target_turnaround = (
                float((1.0 - fraction) * previous_target_time + fraction * cosmic_end),
                target_radius,
                float(shells.enclosed_mass(target_radius / scale_end)),
            )
        previous_target_velocity = target_velocity
        previous_target_radius = float(physical_radius[target_index])
        previous_target_time = cosmic_end
        crossing = np.flatnonzero(
            (physical_velocity[:-1] <= 0.0) & (physical_velocity[1:] >= 0.0)
        )
        if crossing.size:
            index = int(crossing[0])
            fraction = physical_velocity[index] / (
                physical_velocity[index] - physical_velocity[index + 1]
            )
            radius_ta = physical_radius[index] + fraction * (
                physical_radius[index + 1] - physical_radius[index]
            )
            history_turnaround_radius.append(float(radius_ta))
            history_turnaround_mass.append(float(shells.enclosed_mass(radius_ta / scale_end)))
        else:
            history_turnaround_radius.append(np.nan)
            history_turnaround_mass.append(np.nan)

    if not np.all(np.isfinite(shells.radius)) or np.any(np.diff(shells.radius) < 0.0):
        raise RuntimeError("dark-matter-only shells became invalid or unsorted")
    directory = Path(config["par"]["output"]["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    figure = directory / "CosmologicalDarkMatterOnly.jpg"
    plt.figure(figsize=(6, 4))
    plt.plot(history_time, history_inner_radius)
    plt.xlabel("cosmic time [code units]")
    plt.ylabel("innermost shell comoving radius")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure, dpi=200)
    plt.close()
    print("dark-matter-only run passed")
    print("steps = %d, shells = %d" % (steps, shells.number_of_shells))
    print("final cosmic time = %.8g" % history_time[-1])
    print("target halo mass = %.8g code masses (%.8g Msun)" % (target_mass, target_mass * units.mass_in_cgs / 1.98847e33))
    print("analytic turnaround: t=%.8g, r=%.8g code lengths" % (turnaround_time, analytic_rta))
    print("analytic virial: t=%.8g, r=%.8g code lengths" % (collapse_time, analytic_rvir))
    final_a = float(cosmology.scale_factor(final))
    final_h = float(cosmology.hubble(final))
    final_index = int(np.searchsorted(np.cumsum(shells.mass), target_dm_mass, side="left"))
    final_index = min(final_index, shells.number_of_shells - 1)
    final_velocity = final_h * final_a * shells.radius[final_index] + shells.velocity[final_index] / final_a
    print("target-shell final radius=%.8g, radial velocity=%.8g" % (final_a * shells.radius[final_index], final_velocity))
    finite_ta = np.flatnonzero(np.isfinite(history_turnaround_radius))
    if target_turnaround is not None:
        print("numerical target-shell turnaround: t=%.8g, r=%.8g, M=%.8g code" % target_turnaround)
    elif finite_ta.size:
        index = int(finite_ta[-1])
        print("numerical outer turnaround: t=%.8g, r=%.8g, M=%.8g code" % (
            history_time[index], history_turnaround_radius[index], history_turnaround_mass[index]
        ))
    else:
        print("numerical turnaround: not detected")
    print("figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
