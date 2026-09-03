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

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits
from radhydropy.units import _gravitational_constant_code
from example_utils import load_nested_example_config
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name("cosmological_dark_matter_correlation_z100.yaml")


def load_correlation_table(config_filename, runparams):
    filename = runparams.get("linear_correlation_table_filename")
    if not filename:
        return None
    filename = Path(filename)
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def run_lagrangian_top_hat(runparams, icparams, units, cosmology):
    """Calibrate one finite top-hat mass before shell crossing."""
    target_mass = float(icparams["target_halo_mass"])
    delta_i = float(icparams["initial_overdensity"])
    initial = float(icparams["initial_cosmic_time"])
    final = float(runparams["simulation"]["final_time"])
    a_initial = float(cosmology.scale_factor(initial))
    h_initial = float(cosmology.hubble(initial))
    rho_comoving = float(cosmology.background_density(initial)) * a_initial**3
    radius = et.perturbation_radius(icparams, cosmology)
    velocity = -a_initial**2 * h_initial * delta_i * radius / 3.0
    angular_momentum = float(icparams.get("dm_specific_angular_momentum", 0.0))
    g_code = _gravitational_constant_code(units)
    tau = float(cosmology.supercomoving_time(initial))
    final_tau = float(cosmology.supercomoving_time(final))
    timestep = float(
        runparams.get(
            "dm_only_calibration_timestep",
            runparams.get("dm_only_supercomoving_timestep", 0.0005),
        )
    )
    history_time = [initial]
    history_radius = [a_initial * radius]
    turnaround = None
    virial_crossing = None
    previous_velocity = float(
        h_initial * a_initial * radius + velocity / a_initial
    )
    previous_time = initial
    previous_radius = a_initial * radius

    collapse_time = initial * (1.686 / delta_i) ** 1.5
    turnaround_time = 0.5 * collapse_time
    rho_ta = float(cosmology.background_density(turnaround_time))
    rho_vir = float(cosmology.background_density(collapse_time))
    analytic_rta = (target_mass / ((4.0 * np.pi / 3.0) * (9.0 * np.pi**2 / 16.0) * rho_ta)) ** (1.0 / 3.0)
    analytic_rvir = (target_mass / ((4.0 * np.pi / 3.0) * (18.0 * np.pi**2) * rho_vir)) ** (1.0 / 3.0)

    while tau < final_tau and virial_crossing is None:
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

        velocity_half = velocity + 0.5 * dt * acceleration(radius, a_start)
        radius_new = radius + dt * velocity_half
        velocity = velocity_half + 0.5 * dt * acceleration(radius_new, a_end)
        radius = radius_new
        if radius <= 0.0:
            raise RuntimeError("top-hat boundary reached the pressureless singularity before virial crossing")
        tau = tau_end
        physical_velocity = float(
            cosmology.hubble(cosmic_end) * a_end * radius + velocity / a_end
        )
        physical_radius = a_end * radius
        if previous_velocity > 0.0 >= physical_velocity:
            fraction = previous_velocity / (previous_velocity - physical_velocity)
            turnaround = (
                previous_time + fraction * (cosmic_end - previous_time),
                previous_radius + fraction * (physical_radius - previous_radius),
            )
        if turnaround is not None and physical_radius <= analytic_rvir and virial_crossing is None:
            virial_crossing = (cosmic_end, physical_radius)
        previous_velocity = physical_velocity
        previous_time = cosmic_end
        previous_radius = physical_radius
        history_time.append(cosmic_end)
        history_radius.append(physical_radius)

    savedir = Path(runparams["output"]["savedir"])
    figure = savedir / "CosmologicalTopHatDarkMatterOnly.jpg"
    savedir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history_time, history_radius, label="numerical top-hat radius")
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
    print("target halo mass = %.8g code masses (%.8g Msun)" % (target_mass, target_mass * units.mass_in_cgs / 1.98847e33))
    print("analytic turnaround: t=%.8g, r=%.8g kpc" % (turnaround_time, analytic_rta))
    print("analytic virial: t=%.8g, r=%.8g kpc" % (collapse_time, analytic_rvir))
    if turnaround is None:
        raise RuntimeError("Lagrangian top-hat did not reach turnaround")
    print("numerical turnaround: t=%.8g, r=%.8g kpc" % turnaround)
    if virial_crossing is None:
        raise RuntimeError("Lagrangian top-hat did not reach the analytic virial radius")
    print("numerical virial-radius crossing: t=%.8g, r=%.8g kpc, M=%.8g code" % (
        virial_crossing[0], virial_crossing[1], target_mass
    ))
    print("figure = %s" % figure)


def run_live_shell_density_profiles(
    runparams, icparams, units, cosmology, correlation_table=None
):
    """Evolve a gas-free full-matter top-hat and save density snapshots."""
    dm_ic = copy.deepcopy(icparams)
    # With gas removed, the collisionless shells represent the full matter
    # density.  Keeping the configured baryon fraction here would weaken the
    # gravitational normalization by f_DM.
    dm_ic["baryon_fraction"] = 0.0
    dm_ic["dark_matter_shells"] = int(
        runparams.get("dm_only_shells", max(1024, int(icparams["dark_matter_shells"])))
    )
    shells = et.make_dark_matter(
        dm_ic, units, cosmology, correlation_table=correlation_table,
        softening=runparams["dark_matter"]["softening"],
    )
    target_times = np.asarray(
        runparams.get(
            "dm_only_density_times",
            [float(icparams["initial_cosmic_time"]), 4.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        ),
        dtype=float,
    )
    initial = float(icparams["initial_cosmic_time"])
    final = float(runparams["simulation"]["final_time"])
    target_times = np.unique(np.clip(target_times, initial, final))
    tau = float(cosmology.supercomoving_time(initial))
    final_tau = float(cosmology.supercomoving_time(final))
    target_tau = np.asarray(cosmology.supercomoving_time(target_times), dtype=float)
    timestep = float(runparams.get("dm_only_supercomoving_timestep", 0.0005))
    profiles = []
    virial_radii = []
    next_snapshot = 0

    def save_profile(cosmic_time):
        a = float(cosmology.scale_factor(cosmic_time))
        order = np.argsort(shells.radius)
        radius = a * np.asarray(shells.radius[order], dtype=float)
        mass = np.asarray(shells.mass[order], dtype=float)
        # Cold shell collapse can carry shells through the coordinate origin
        # after crossing.  They remain part of the enclosed mass; represent
        # crossed/central material at a small positive radius for logarithmic
        # density binning instead of silently dropping it from the profile.
        radius = np.abs(np.nan_to_num(radius, nan=0.0, posinf=0.0, neginf=0.0))
        radius = np.maximum(radius, 1.0e-8)
        core_mass = float(getattr(shells, "central_core_mass", 0.0))
        core_radius = a * float(getattr(shells, "central_core_radius", 0.0))
        profiles.append((float(cosmic_time), radius, mass, core_mass, core_radius))
        # Include the absorbed unresolved-core mass when locating r200.  The
        # profile bins already include this same mass, so the overdensity
        # marker must use the identical enclosed-mass definition.
        cumulative_mass = core_mass + np.cumsum(mass)
        mean_density = cumulative_mass / (
            4.0 * np.pi / 3.0 * np.maximum(radius, 1.0e-30) ** 3
        )
        threshold = 200.0 * float(cosmology.background_density(cosmic_time))
        candidates = np.flatnonzero(mean_density >= threshold)
        if candidates.size:
            index = int(candidates[-1])
            virial_radii.append(float(radius[index]))
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
        background = lambda radius, rho=rho_comoving: (
            4.0 * np.pi / 3.0 * rho * np.asarray(radius, dtype=float) ** 3
        )
        shells.step(
            dt,
            crossing_safety_factor=float(
                runparams.get("dark_matter_crossing_safety_factor", 0.5)
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
    bin_count = int(runparams.get("dm_density_bins", 128))
    bin_min = max(1.0e-8, min(np.min(radius) for radius in shell_radii) * 0.9)
    bin_max = max(np.max(radius) for radius in shell_radii) * 1.1
    bin_edges = np.geomspace(bin_min, bin_max, bin_count + 1)
    bin_radii = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_volumes = 4.0 * np.pi / 3.0 * np.diff(bin_edges**3)
    densities = []
    for radius, mass, core_mass, core_radius in zip(
        shell_radii, shell_masses, core_masses, core_radii
    ):
        mass_in_bin, _ = np.histogram(radius, bins=bin_edges, weights=mass)
        if core_mass > 0.0 and core_radius > 0.0:
            core_bin = int(np.searchsorted(bin_edges, core_radius, side="right") - 1)
            if 0 <= core_bin < mass_in_bin.size:
                mass_in_bin[core_bin] += core_mass
        density = mass_in_bin / np.maximum(bin_volumes, 1.0e-30)
        densities.append(np.where(mass_in_bin > 0.0, density, np.nan))
    densities = np.asarray(densities)
    virial_radii = np.asarray(virial_radii)
    scale_factors = np.asarray([float(cosmology.scale_factor(time)) for time in times])
    comoving_bin_radii = bin_radii[None, :] / scale_factors[:, None]
    target_mass = float(icparams["target_halo_mass"])
    virial_overdensity = 18.0 * np.pi**2
    analytic_rvir = (
        target_mass
        / ((4.0 * np.pi / 3.0) * virial_overdensity
           * np.asarray([float(cosmology.background_density(time)) for time in times]))
    ) ** (1.0 / 3.0)
    output_dir = Path(runparams["output"]["savedir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "CosmologicalDarkMatterOnlyDensityProfiles.npz"
    figure = output_dir / "CosmologicalDarkMatterOnlyDensityProfiles.jpg"
    np.savez(
        data_file,
        time_Gyr=times,
        radius_kpc=bin_radii,
        radius_comoving_kpc=comoving_bin_radii,
        bin_edges_kpc=bin_edges,
        density_code=densities,
        central_core_mass=core_masses,
        central_core_radius_kpc=core_radii,
        rvir_kpc=virial_radii,
        analytic_rvir_kpc=analytic_rvir,
    )

    plt.figure(figsize=(7.0, 5.0))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(times)))
    for time, density, rvir, comoving_radius, color in zip(
        times, densities, virial_radii, comoving_bin_radii, colors
    ):
        valid = np.isfinite(density) & (density > 0.0)
        plt.loglog(comoving_radius[valid], density[valid], color=color, lw=1.6,
                   label="t = %.1f" % time)
        if np.isfinite(rvir):
            scale_factor = float(cosmology.scale_factor(time))
            plt.axvline(rvir / scale_factor, color=color, ls="--", lw=1.1, alpha=0.8)
    # Bertschinger/Fillmore--Goldreich similarity reference for the local
    # initial perturbation slope delta M/M propto M^-s.  Normalize the
    # nonlinear rho propto r^(-9s/(1+3s)) reference to the latest resolved
    # r200 profile so it indicates the predicted slope rather than an
    # arbitrary density normalization.
    similarity_s = 0.2
    similarity_slope = 9.0 * similarity_s / (1.0 + 3.0 * similarity_s)
    resolved = np.flatnonzero(np.isfinite(virial_radii) & (virial_radii > 0.0))
    if resolved.size:
        reference_index = int(resolved[-1])
        reference_radius = float(virial_radii[reference_index])
        reference_profile = np.asarray(densities[reference_index], dtype=float)
        reference_radius_bins = np.asarray(
            bin_radii, dtype=float
        )
        valid_reference = np.isfinite(reference_profile) & (reference_profile > 0.0)
        if np.count_nonzero(valid_reference) >= 2:
            reference_density = float(np.exp(np.interp(
                np.log(reference_radius),
                np.log(reference_radius_bins[valid_reference]),
                np.log(reference_profile[valid_reference]),
            )))
            line_radius = np.geomspace(
                max(reference_radius * 0.5, reference_radius_bins[valid_reference].min()),
                min(reference_radius * 3.0, reference_radius_bins[valid_reference].max()),
                64,
            )
            line_density = reference_density * (
                line_radius / reference_radius
            ) ** (-similarity_slope)
            plt.plot(
                line_radius / scale_factors[reference_index], line_density,
                color="black", ls="-.", lw=1.8,
                label=r"Bertschinger reference ($s=0.2$, $\rho\propto r^{-1.125}$)",
            )
    plt.xlabel("comoving radius [kpc]")
    plt.ylabel(r"dark-matter density [code mass / kpc$^3$]")
    plt.title("Live dark-matter-only density evolution")
    plt.grid(alpha=0.25, which="both")
    plt.plot([], [], color="0.25", ls="--", label=r"$r_{200}$")
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
        label=r"simulation $r_{200}$",
    )
    plt.plot(
        times, analytic_rvir, "--", color="tab:orange",
        label="analytic spherical-collapse virial radius",
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


def main(config_filename=DEFAULT_CONFIG, final_time_override=None):
    config = load_nested_example_config(config_filename)
    runparams = config["par"]
    icparams = config["initial_condition"]
    if final_time_override is not None:
        runparams["simulation"] = dict(runparams["simulation"])
        runparams["simulation"]["final_time"] = float(final_time_override)
    units = CodeUnits.from_mapping(runparams["units"]["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams["gravity"]["cosmology_t_ref"]),
        a_ref=float(runparams["gravity"]["cosmology_a_ref"]),
    )
    correlation_table = load_correlation_table(config_filename, runparams)
    run_lagrangian_top_hat(runparams, icparams, units, cosmology)
    run_live_shell_density_profiles(
        runparams, icparams, units, cosmology,
        correlation_table=correlation_table,
    )
    return
    dm_ic = copy.deepcopy(icparams)
    dm_ic["dark_matter_shells"] = int(
        runparams.get("dm_only_shells", max(1024, int(icparams["dark_matter_shells"])))
    )
    shells = et.make_dark_matter(dm_ic, units, cosmology)
    dm_fraction = 1.0 - float(icparams["baryon_fraction"])
    initial = float(icparams["initial_cosmic_time"])
    final = float(runparams["final_cosmic_time"])
    time = float(cosmology.supercomoving_time(initial))
    final_tau = float(cosmology.supercomoving_time(final))
    timestep = float(runparams.get("dm_only_supercomoving_timestep", 0.002))
    target_mass = float(icparams["target_halo_mass"])
    target_dm_mass = target_mass * (1.0 - float(icparams["baryon_fraction"]))
    delta_i = float(icparams["initial_overdensity"])
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
    while time < final_tau:
        dt = min(timestep, final_tau - time)
        cosmic_start = float(cosmology.cosmic_time_from_supercomoving(time))
        scale_start = float(cosmology.scale_factor(cosmic_start))
        rho_start = float(cosmology.background_density(cosmic_start)) * scale_start**3
        background = lambda radius: (
            4.0 * np.pi / 3.0 * rho_start * dm_fraction * np.asarray(radius)**3
        )
        time_end = time + dt
        cosmic_end = float(cosmology.cosmic_time_from_supercomoving(time_end))
        scale_end = float(cosmology.scale_factor(cosmic_end))
        shells.step(
            dt,
            crossing_safety_factor=float(runparams.get("dark_matter_crossing_safety_factor", 0.5)),
            background_enclosed_mass=background,
            scale_factor=scale_start,
            scale_factor_end=scale_end,
            cosmological=True,
        )
        time = time_end
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
    savedir = Path(runparams["output"]["savedir"])
    savedir.mkdir(parents=True, exist_ok=True)
    figure = savedir / "CosmologicalDarkMatterOnly.jpg"
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
    parser.add_argument(
        "--final-time", type=float, default=None,
        help="override the final cosmic time without changing the YAML",
    )
    args = parser.parse_args()
    main(args.config, final_time_override=args.final_time)
