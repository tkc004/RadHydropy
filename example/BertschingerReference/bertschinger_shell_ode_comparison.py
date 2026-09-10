"""Compare live ``DarkMatterShells`` with the Bertschinger Eq. (4.1) curve."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(
    tempfile.gettempdir(), 'radhydropy-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import quantity_to_value
import tools as example_tools
from bertschinger_ode import (
    first_outer_caustic,
    first_post_centre_apocentre,
    solve_eq41_self_similar,
)
from shell_orbit_tracker import ShellOrbitTracker


DEFAULT_CONFIG = Path(__file__).with_name('bertschinger_reference.yaml')


def _radius_turnaround_proper_code(shells, time_cosmic_code, config):
    """Return the first velocity sign-change radius, or ``None``."""
    cosmology = config['_cosmology']
    radius_proper_code = float(cosmology.scale_factor(time_cosmic_code)) * shells.radius
    velocity_proper_code = example_tools.peculiar_velocity_proper_code(
        shells, time_cosmic_code, config)
    crossing = np.flatnonzero(velocity_proper_code[:-1] * velocity_proper_code[1:] <= 0.0)
    if crossing.size == 0:
        return None
    # After shell crossing there can be several sign changes. The
    # Bertschinger turnaround scale is the outermost infall/expansion
    # interface, not the innermost central bounce.
    index = int(crossing[-1])
    denominator = velocity_proper_code[index] - velocity_proper_code[index + 1]
    if denominator == 0.0:
        return float(radius_proper_code[index])
    fraction = velocity_proper_code[index] / denominator
    return float(radius_proper_code[index] + fraction * (radius_proper_code[index + 1] - radius_proper_code[index]))


def _outer_lagrangian_caustic_radius(
        shells, initial_q, time_cosmic_code, config, turnaround,
        smoothing_bins=2.0):
    """Find the outer fold of the Lagrangian map ``r(q)``.

    Shell mass is invariant, so it provides an identity through radius
    crossings.  Reordering current shells by their initial mass reconstructs
    the Lagrangian coordinate ``q``.  A caustic is a fold where ``dr/dq=0``;
    the outermost interior fold is returned.
    """
    initial_q = np.asarray(initial_q, dtype=float)
    if shells.shell_id is None:
        raise RuntimeError('shell identities are required for Lagrangian maps')
    q = initial_q[np.asarray(shells.shell_id, dtype=int)]
    cosmology = config['_cosmology']
    radius_proper_code = (float(cosmology.scale_factor(time_cosmic_code)) *
                          np.asarray(shells.radius, dtype=float))
    vel_peculiar_proper_code = example_tools.peculiar_velocity_proper_code(
        shells, time_cosmic_code, config)
    order = np.argsort(q)
    q = q[order]
    radius_proper_code = radius_proper_code[order]
    vel_peculiar_proper_code = vel_peculiar_proper_code[order]
    lower = max(float(radius_proper_code.min()), turnaround * 1.0e-4)
    upper = turnaround * (1.0 - 1.0e-6)
    if not lower < upper:
        return None
    smoothed_radius = gaussian_filter1d(
        radius_proper_code, float(smoothing_bins), mode='nearest')
    derivative = np.gradient(smoothed_radius, q)
    # The outer caustic is the fold separating an outgoing inner stream from
    # an infalling outer stream. Require both the Lagrangian fold orientation
    # and this phase-space orientation; radius alone selects inner folds.
    folds = np.flatnonzero((derivative[:-1] >= 0.0) &
                           (derivative[1:] < 0.0))
    fold_radius = []
    for index in folds:
        fold_radius_proper_code = 0.5 * (radius_proper_code[index] + radius_proper_code[index + 1])
        velocity_left_proper_code = vel_peculiar_proper_code[index]
        velocity_right_proper_code = vel_peculiar_proper_code[index + 1]
        if (lower * 1.01 < fold_radius_proper_code < upper and velocity_left_proper_code >= 0.0 and
                velocity_right_proper_code <= 0.0):
            fold_radius.append(fold_radius_proper_code)
    if not fold_radius:
        return None
    return float(max(fold_radius))


def _density_slope_profile(shells, time_cosmic_code, config, turnaround, bins=192,
                           smoothing_bins=3.0):
    """Return rho(r), its logarithmic slope, and profile splashback radii."""
    cosmology = config['_cosmology']
    a = float(cosmology.scale_factor(time_cosmic_code))
    radius_proper_code = a * np.asarray(shells.radius, dtype=float)
    mass_comoving_code = np.asarray(shells.mass, dtype=float)
    rho_background = float(cosmology.background_density(time_cosmic_code))
    edges = np.geomspace(radius_proper_code.min(), radius_proper_code.max(), int(bins) + 1)
    index = np.clip(np.searchsorted(edges, radius_proper_code) - 1, 0, len(edges) - 2)
    deposited_mass_comoving_code = np.bincount(index, weights=mass_comoving_code, minlength=len(edges) - 1)
    shell_volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(edges**3)
    density_proper_code = deposited_mass_comoving_code / np.maximum(shell_volume_proper_code, 1.0e-300)
    # First smooth the density profile itself, then differentiate its
    # logarithm. Smoothing deposited mass instead changes the radial measure
    # before the density is formed.
    density_proper_code = gaussian_filter1d(density_proper_code, float(smoothing_bins),
                                mode='nearest')
    valid = density_proper_code > 0.0
    log_radius = np.log(np.sqrt(edges[:-1] * edges[1:]))
    log_density = np.full_like(log_radius, np.nan)
    log_density[valid] = np.log(density_proper_code[valid])
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 8:
        return None
    slope = np.gradient(log_density, log_radius)

    cumulative_mass_comoving_code = np.cumsum(mass_comoving_code)
    mean_density_proper_code = cumulative_mass_comoving_code / (4.0 * np.pi / 3.0 * radius_proper_code**3)
    crossing = np.flatnonzero(
        (mean_density_proper_code[:-1] >= 200.0 * rho_background) &
        (mean_density_proper_code[1:] < 200.0 * rho_background))
    if not crossing.size:
        return None
    virial_index = int(crossing[-1])
    rvir = float(np.exp(np.interp(
        np.log(200.0 * rho_background),
        np.log(mean_density_proper_code[virial_index:virial_index + 2][::-1]),
        np.log(radius_proper_code[virial_index:virial_index + 2][::-1])))
    )
    candidates = (valid & (np.exp(log_radius) > 1.05 * rvir) &
                  (np.exp(log_radius) < 0.95 * turnaround))
    candidates[:3] = False
    candidates[-3:] = False
    if not np.any(candidates):
        return None
    candidate_indices = np.flatnonzero(candidates)
    splashback_index = int(candidate_indices[np.argmin(slope[candidates])])
    return {
        'radius_proper_code': np.exp(log_radius),
        'rho_proper_code': np.exp(log_density),
        'slope': slope,
        'radius_virial_proper_code': rvir,
        'radius_splashback_proper_code': float(np.exp(log_radius[splashback_index])),
        'rho_background_proper_code': rho_background,
    }


def run_comparison(config_filename=DEFAULT_CONFIG):
    config = example_tools.load_reference_config(config_filename)

    initial_condition = config['initial_condition']
    example = config['example']
    units = example_tools.code_units_from_config(config)
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(example['cosmology_t_ref']),
        a_ref=float(example['cosmology_a_ref']),
    )
    Path(config["par"]['output']['directory']).mkdir(parents=True, exist_ok=True)
    config["_code_units"] = units
    config["_cosmology"] = cosmology
    shells, _ = example_tools.make_scale_free_shells(config)
    initial_q = shells.radius.copy()
    tracker = ShellOrbitTracker(
        initial_q, cosmology,
        recent_window_fraction=float(example.get(
            'recent_accretion_window_fraction', 0.5)))
    initial_time = quantity_to_value(initial_condition['time_cosmic'], units.time_unit)
    time_comparison_final_cosmic = float(example.get(
        'time_comparison_final_cosmic',
        initial_time * np.exp(float(example['ode_xi_end']))))
    timestep = float(example.get('comparison_timestep', 0.002))
    snapshot_stride = int(example.get('comparison_snapshot_stride', 5))
    shell_stride = int(example.get('comparison_shell_stride', 16))
    if time_comparison_final_cosmic <= initial_time or timestep <= 0.0:
        raise ValueError('invalid shell comparison time configuration')

    tau = float(cosmology.supercomoving_time(initial_time))
    tau_comparison_final_supercomoving = float(
        cosmology.supercomoving_time(time_comparison_final_cosmic)
    )
    snapshot = 0
    xi_values = []
    lambda_values = []
    turnaround_values = []
    caustic_values = []
    apocentre_values = []
    apocentre_event_xi = np.empty(0)
    apocentre_event_lambda = np.empty(0)
    slope_profiles = []
    profile_targets = np.asarray([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    next_profile = 0
    caustic_smoothing = float(example.get('caustic_smoothing_bins', 2.0))
    tracker.observe(initial_time, float(cosmology.scale_factor(initial_time)),
                    shells.radius, shells.velocity, shells.mass,
                    shells.shell_id)
    while tau < tau_comparison_final_supercomoving - 1.0e-12:
        dt = min(timestep, tau_comparison_final_supercomoving - tau)
        time_cosmic_code = float(cosmology.cosmic_time_from_supercomoving(tau))
        next_time = float(cosmology.cosmic_time_from_supercomoving(tau + dt))
        a_start = float(cosmology.scale_factor(time_cosmic_code))
        a_end = float(cosmology.scale_factor(next_time))
        rho_comoving = float(cosmology.background_density(time_cosmic_code)) * a_start**3
        background = 4.0 * np.pi / 3.0 * rho_comoving * shells.radius**3
        tau_start = tau
        actual_dt = shells.step(
            dt,
            crossing_safety_factor=float(example['crossing_safety_factor']),
            background_enclosed_mass=background,
            scale_factor=a_start,
            scale_factor_end=a_end,
            cosmological=True,
            include_shell_mass_with_fixed=True,
            state_callback=lambda elapsed, a, radius, velocity, mass, shell_id:
                tracker.observe(
                    float(cosmology.cosmic_time_from_supercomoving(
                        tau_start + elapsed)), a, radius, velocity, mass,
                    shell_id),
        )
        # Radial shells have no centrifugal barrier. Match the ODE's
        # controlled centre treatment after a finite leapfrog step.
        central_shell = shells.radius <= float(initial_condition['softening'])
        if np.any(central_shell):
            shells.radius[central_shell] = float(initial_condition['softening'])
            shells.velocity[central_shell] = np.abs(
                shells.velocity[central_shell])
            shells.sort_by_radius()
        tau += actual_dt
        snapshot += 1
        if snapshot % snapshot_stride:
            continue
        time_cosmic_code = float(cosmology.cosmic_time_from_supercomoving(tau))
        turnaround = _radius_turnaround_proper_code(shells, time_cosmic_code, config)
        if turnaround is None or not np.isfinite(turnaround):
            continue
        radius_proper_code = (
            float(cosmology.scale_factor(time_cosmic_code)) * shells.radius
        )
        selected = slice(None, None, shell_stride)
        xi_values.extend(np.full(radius_proper_code[selected].shape,
                                 np.log(time_cosmic_code / initial_time)))
        lambda_values.extend((radius_proper_code[selected] / turnaround).tolist())
        turnaround_values.append((time_cosmic_code, turnaround))
        caustic = _outer_lagrangian_caustic_radius(
            shells, initial_q, time_cosmic_code, config,
            turnaround, smoothing_bins=caustic_smoothing)
        if caustic is not None and np.isfinite(caustic):
            caustic_values.append((np.log(time_cosmic_code / initial_time),
                                   caustic / turnaround))
        xi = np.log(time_cosmic_code / initial_time)
        if (next_profile < profile_targets.size and
                xi >= profile_targets[next_profile]):
            profile = _density_slope_profile(
                shells, time_cosmic_code, config, turnaround,
                bins=int(example.get('slope_profile_bins', 192)),
                smoothing_bins=float(example.get(
                    'slope_smoothing_bins', 3.0)))
            if profile is not None:
                profile['xi'] = xi
                profile['time_cosmic_code'] = time_cosmic_code
                profile['radius_turnaround_proper_code'] = turnaround
                slope_profiles.append(profile)
            next_profile += 1

    if not xi_values:
        raise RuntimeError('the shell simulation produced no turnaround samples')
    ode = solve_eq41_self_similar(
        xi_end=float(example['ode_xi_end']),
        points=int(example['ode_points']),
        similarity_exponent=float(example['ode_similarity_exponent']),
        centre_match_lambda=float(example['ode_centre_match_lambda']),
        centre_matching_velocity=float(example['ode_centre_matching_velocity']),
    )
    splashback_xi, splashback_lambda = first_post_centre_apocentre(ode)
    caustic_xi, caustic_lambda = first_outer_caustic(ode)
    xi_values = np.asarray(xi_values)
    lambda_values = np.asarray(lambda_values)
    finite = np.isfinite(xi_values) & np.isfinite(lambda_values) & (lambda_values > 0.0)
    lambda_max = float(example.get('comparison_lambda_max', 2.0))
    finite &= lambda_values <= lambda_max
    figure = Path(config["par"]['output']['directory']) / 'BertschingerDarkMatterShellsVsODE.jpg'
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(xi_values[finite], lambda_values[finite], s=1.0, alpha=0.12,
                 label='RadHydropy DarkMatterShells')
    axis.plot(ode.xi, ode.lam, color='black', linewidth=2.0,
              label='Bertschinger Eq. (4.1) ODE')
    if caustic_values:
        caustic_values = np.asarray(caustic_values)
        axis.scatter(caustic_values[:, 0], caustic_values[:, 1],
                     color='tab:red', s=10, alpha=0.35,
                     label='shell-ensemble caustic estimates')
        caustic_trace = gaussian_filter1d(caustic_values[:, 1], 2.0)
        axis.plot(caustic_values[:, 0], caustic_trace,
                  color='tab:red', linewidth=1.5,
                  label='caustic trend')
        axis.axhline(caustic_lambda, color='tab:green', linestyle='--',
                     linewidth=1.5,
                     label='ODE fixed-time outer caustic')
        np.savez(
            Path(config["par"]['output']['directory']) / 'BertschingerDarkMatterCaustic.npz',
            xi=caustic_values[:, 0], lambda_caustic=caustic_values[:, 1],
            ode_splashback_xi=splashback_xi,
            ode_splashback_lambda=splashback_lambda,
            ode_outer_caustic_xi=caustic_xi,
            ode_outer_caustic_lambda=caustic_lambda,
        )
    else:
        # Overwrite stale products when the current run has no resolved,
        # phase-space-consistent Lagrangian fold.
        np.savez(
            Path(config["par"]['output']['directory']) / 'BertschingerDarkMatterCaustic.npz',
            xi=np.empty(0), lambda_caustic=np.empty(0),
            ode_splashback_xi=splashback_xi,
            ode_splashback_lambda=splashback_lambda,
            ode_outer_caustic_xi=caustic_xi,
            ode_outer_caustic_lambda=caustic_lambda,
        )
    # Bin first-apocentre events by the time at which they occur. Normalize
    # each event with r_ta at that same time, not with the later output
    # turnaround radius.
    apocentre_events = tracker.first_apocenter_events()
    if apocentre_events.size and len(turnaround_values) >= 2:
        ta_history = np.asarray(turnaround_values, dtype=float)
        event_time = apocentre_events[:, 0]
        event_ta = np.interp(event_time, ta_history[:, 0], ta_history[:, 1])
        event_lambda = apocentre_events[:, 1] / event_ta
        apocentre_event_xi = np.log(event_time / initial_time)
        apocentre_event_lambda = event_lambda
        bins = []
        for lower, upper in zip(ta_history[:-1, 0], ta_history[1:, 0]):
            selected = (event_time >= lower) & (event_time < upper)
            if not np.any(selected):
                continue
            values = event_lambda[selected]
            bins.append((np.log(0.5 * (lower + upper) / initial_time),
                         np.median(values), np.percentile(values, 16.0),
                         np.percentile(values, 84.0), values.size,
                         np.interp(0.5 * (lower + upper), ta_history[:, 0],
                                   ta_history[:, 1])))
        apocentre_values = np.asarray(bins)
    if len(apocentre_values):
        np.savez(
            Path(config["par"]['output']['directory']) / 'BertschingerRecentApocenters.npz',
            xi=apocentre_values[:, 0],
            radius_median=apocentre_values[:, 1],
            radius_p16=apocentre_values[:, 2],
            radius_p84=apocentre_values[:, 3],
            number_of_shells=apocentre_values[:, 4],
            radius_turnaround_proper_code=apocentre_values[:, 5],
            lambda_median=apocentre_values[:, 1],
            lambda_p16=apocentre_values[:, 2],
            lambda_p84=apocentre_values[:, 3],
        )
        orbit_figure, orbit_axis = plt.subplots(figsize=(8, 5))
        orbit_axis.plot(apocentre_values[:, 0],
                        apocentre_values[:, 1],
                        color='tab:purple', label='recent-shell median')
        orbit_axis.fill_between(
            apocentre_values[:, 0],
            apocentre_values[:, 2], apocentre_values[:, 3],
            color='tab:purple', alpha=0.2, label='16--84 percentile')
        orbit_axis.axhline(caustic_lambda, color='tab:green', linestyle='--',
                           label='ODE fixed-time caustic')
        orbit_axis.set_xlabel(r'$\xi=\ln(t/t_{\rm ref})$')
        orbit_axis.set_ylabel(r'$R_{\rm apo}/r_{\rm ta}(t)$')
        orbit_axis.set_title('Recently accreted first apocenters')
        orbit_axis.grid(alpha=0.25)
        orbit_axis.legend(fontsize=8)
        orbit_figure.tight_layout()
        orbit_figure.savefig(
            Path(config["par"]['output']['directory']) /
            'BertschingerRecentApocenters.jpg', dpi=200)
        plt.close(orbit_figure)
    if slope_profiles:
        density_figure, density_axis = plt.subplots(figsize=(8, 5))
        for profile in slope_profiles:
            density_axis.plot(
                profile['radius_proper_code'] / profile['radius_virial_proper_code'],
                profile['rho_proper_code'] / profile['rho_background_proper_code'],
                linewidth=1.6,
                label=r'$\xi=%.2f$' % profile['xi'])
        density_axis.axvline(1.0, color='black', linestyle=':',
                             label=r'$R_{200m}$')
        density_axis.set_xscale('log')
        density_axis.set_yscale('log')
        density_axis.set_xlabel(r'$r/R_{200m}$')
        density_axis.set_ylabel(r'$\rho_{\rm smooth}(r)/\rho_{\rm bg}$')
        density_axis.set_title('Bertschinger smoothed dark-matter profiles')
        density_axis.grid(alpha=0.25, which='both')
        density_axis.legend(fontsize=8)
        density_figure.tight_layout()
        density_figure.savefig(
            Path(config["par"]['output']['directory']) /
            'BertschingerDarkMatterDensityProfile.jpg', dpi=200)
        plt.close(density_figure)

        slope_figure, slope_axis = plt.subplots(figsize=(8, 5))
        for profile in slope_profiles:
            x = np.log10(profile['radius_proper_code'] / profile['radius_virial_proper_code'])
            line, = slope_axis.plot(
                x, profile['slope'],
                linewidth=1.6,
                label=r'$\xi=%.2f$' % profile['xi'])
            slope_axis.plot(
                np.log10(profile['radius_splashback_proper_code'] /
                          profile['radius_virial_proper_code']),
                np.interp(profile['radius_splashback_proper_code'], profile['radius_proper_code'],
                          profile['slope']),
                            marker='o', color=line.get_color(),
                            markeredgecolor='black')
        slope_axis.axvline(0.0, color='black', linestyle=':',
                           label=r'$R_{200m}$')
        slope_axis.set_xlabel(r'$\log_{10}(r/R_{200m})$')
        slope_axis.set_ylabel(r'$d\ln\rho/d\ln r$')
        slope_axis.set_title('Bertschinger dark-matter density slopes')
        slope_axis.grid(alpha=0.25)
        slope_axis.legend(fontsize=8)
        slope_figure.tight_layout()
        slope_figure.savefig(
            Path(config["par"]['output']['directory']) /
            'BertschingerDarkMatterDensitySlope.jpg', dpi=200)
        plt.close(slope_figure)
        np.savez(
            Path(config["par"]['output']['directory']) /
            'BertschingerDarkMatterDensitySlope.npz',
            xi=np.asarray([p['xi'] for p in slope_profiles]),
            radius_proper_code=np.asarray([p['radius_proper_code'] for p in slope_profiles], dtype=object),
            rho_proper_code=np.asarray([p['rho_proper_code'] for p in slope_profiles], dtype=object),
            slope=np.asarray([p['slope'] for p in slope_profiles], dtype=object),
            radius_virial_proper_code=np.asarray([p['radius_virial_proper_code'] for p in slope_profiles]),
            radius_splashback_proper_code=np.asarray(
                [p['radius_splashback_proper_code'] for p in slope_profiles]),
            radius_turnaround_proper_code=np.asarray(
                [p['radius_turnaround_proper_code'] for p in slope_profiles]),
        )
    if len(caustic_values) or slope_profiles or len(apocentre_values):
        comparison_data = {}
        comparison_figure, comparison_axis = plt.subplots(figsize=(8, 5))
        if slope_profiles:
            slope_xi = np.asarray([p['xi'] for p in slope_profiles])
            slope_lambda = np.asarray([
                p['radius_splashback_proper_code'] / p['radius_turnaround_proper_code']
                for p in slope_profiles])
            comparison_axis.plot(
                slope_xi, slope_lambda, 'o-', color='tab:blue',
                label=r'density slope $R_{\rm sp}/R_{\rm ta}$')
            comparison_data.update(slope_xi=slope_xi,
                                    slope_lambda=slope_lambda)
        if len(caustic_values):
            caustic_array = np.asarray(caustic_values)
            comparison_axis.plot(
                caustic_array[:, 0], caustic_array[:, 1], '.',
                color='tab:green', markersize=5,
                label=r'Lagrangian caustic $R_{\rm c}/R_{\rm ta}$')
            comparison_data.update(caustic_xi=caustic_array[:, 0],
                                    caustic_lambda=caustic_array[:, 1])
        if len(apocentre_values):
            apo_array = np.asarray(apocentre_values)
            # Event values were already normalized by r_ta at each
            # apocentre time above; do not normalize them a second time.
            apo_lambda = apo_array[:, 1]
            comparison_axis.plot(
                apo_array[:, 0], apo_lambda, 's-', color='tab:purple',
                label=r'recent-shell apocentre median $R_{\rm apo}/R_{\rm ta}$')
            comparison_axis.fill_between(
                apo_array[:, 0], apo_array[:, 2], apo_array[:, 3],
                color='tab:purple',
                alpha=0.18, label='apocentre 16--84 percentile')
            comparison_data.update(
                apocentre_xi=apo_array[:, 0],
                apocentre_lambda=apo_lambda,
                apocentre_lambda_p16=apo_array[:, 2],
                apocentre_lambda_p84=apo_array[:, 3],
                apocentre_count=apo_array[:, 4])
        if len(apocentre_event_xi):
            comparison_axis.scatter(
                apocentre_event_xi, apocentre_event_lambda,
                color='tab:purple', edgecolor='white', linewidth=0.5,
                s=28, alpha=0.7, zorder=3,
                label=r'individual events at $\xi_{\rm apo}$')
            comparison_data.update(
                apocentre_event_xi=apocentre_event_xi,
                apocentre_event_lambda=apocentre_event_lambda)
        comparison_axis.axhline(
            caustic_lambda, color='black', linestyle='--', linewidth=1.2,
            label=r'ODE envelope $R_{\rm c}/R_{\rm ta}=%.3f$' % caustic_lambda)
        comparison_axis.axhline(
            splashback_lambda, color='tab:red', linestyle=':', linewidth=1.5,
            label=r'ODE first apocentre $R_{\rm sp}/R_{\rm ta}=%.3f$' %
                  splashback_lambda)
        comparison_data.update(ode_splashback_lambda=splashback_lambda,
                                ode_outer_caustic_lambda=caustic_lambda)
        comparison_axis.set_xlabel(r'$\xi=\ln(t/t_{\rm ref})$')
        comparison_axis.set_ylabel(r'$R/r_{\rm ta}(t)$')
        time_axis = comparison_axis.secondary_xaxis(
            'top', functions=(np.exp,
                              lambda value: np.log(np.maximum(
                                  value, np.finfo(float).tiny))))
        time_axis.set_xlabel(r'$t_{\rm apo}/t_{\rm ref}$')
        comparison_axis.set_title('Splashback-radius comparison')
        comparison_axis.grid(alpha=0.25)
        comparison_axis.legend(fontsize=8)
        comparison_figure.tight_layout()
        comparison_figure.savefig(
            Path(config["par"]['output']['directory']) /
            'BertschingerSplashbackComparison.jpg', dpi=200)
        plt.close(comparison_figure)
        np.savez(Path(config["par"]['output']['directory']) /
                 'BertschingerSplashbackComparison.npz', **comparison_data)
    axis.set_xlim(0.0, float(example['ode_xi_end']))
    axis.set_ylim(0.0, lambda_max)
    axis.set_xlabel(r'$\xi=\ln(t/t_{\rm ref})$')
    axis.set_ylabel(r'$\lambda=r/r_{\rm ta}(t)$')
    axis.grid(alpha=0.25)
    axis.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    if not np.all(np.isfinite(shells.radius)):
        raise RuntimeError('dark-matter shell radii became non-finite')
    if not np.all(np.diff(shells.radius) >= 0.0):
        raise RuntimeError('dark-matter shells are not sorted after evolution')
    print('DarkMatterShells/ODE comparison generated')
    print('simulation snapshots = %d' % len(turnaround_values))
    print('outer-caustic samples = %d' % len(caustic_values))
    print('density-slope profiles = %d' % len(slope_profiles))
    print('recent-apocentre samples = %d' % len(apocentre_values))
    for profile in slope_profiles:
        print('xi = %.4g: R200m = %.8g, Rsp(slope) = %.8g' %
              (profile['xi'], profile['radius_virial_proper_code'],
               profile['radius_splashback_proper_code']))
    print('ODE splashback: xi = %.8g, lambda = %.8g' %
          (splashback_xi, splashback_lambda))
    print('ODE outer caustic: xi = %.8g, lambda = %.8g' %
          (caustic_xi, caustic_lambda))
    print('simulation time range = %.8g .. %.8g' % (
        initial_time, time_comparison_final_cosmic
    ))
    print('figure = %s' % figure)
    return figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    run_comparison(parser.parse_args().config)
