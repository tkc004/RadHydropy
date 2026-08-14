"""Pre-crossing Lagrangian-shell comparison with Bertschinger Eq. (4.1)."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.example_config import load_example_parameters
import tools as example_tools
from bertschinger_ode import solve_eq41_self_similar


DEFAULT_CONFIG = Path(__file__).with_name('bertschinger_reference.yaml')


def make_turnaround_shells(icparams, units, cosmology):
    """Create background interior shells and one tracked shell at ``r_a``."""
    # Isolate the Lagrangian turnaround shell for the pre-crossing benchmark.
    turnaround_radius = float(icparams.get('pre_crossing_turnaround_radius', 1.0))
    time = float(icparams['initial_cosmic_time'])
    scale_factor = float(cosmology.scale_factor(time))
    hubble = float(cosmology.hubble(time))
    background_coefficient = 2.0 / (9.0 * cosmology.gravitational_constant)
    turnaround_mass = (9.0 * np.pi * np.pi / 16.0) * background_coefficient
    fixed_total_mass = lambda r: turnaround_mass + background_coefficient * np.asarray(r) * np.asarray(r) * np.asarray(r)
    shells = DarkMatterShells(radius=np.asarray([turnaround_radius]),
                              velocity=np.asarray([-scale_factor * scale_factor * hubble * turnaround_radius]),
                              mass=np.asarray([1.0e-12]),
                              fixed_enclosed_mass=fixed_total_mass,
                              softening=float(icparams.get('pre_crossing_softening', 1.0e-3)),
                              code_units=units)
    return shells, 0
def run_pre_crossing(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_example_parameters(config_filename, Path.cwd().resolve())
    units = example_tools.load_units(runparams)
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams['cosmology_t_ref']),
        a_ref=float(runparams['cosmology_a_ref']),
    )
    shells, tracked = make_turnaround_shells(icparams, units, cosmology)
    initial_time = float(icparams['initial_cosmic_time'])
    final_xi = float(runparams.get('pre_crossing_final_xi', 0.9))
    match_lambda = float(runparams.get('pre_crossing_match_lambda', 0.002))
    final_time = initial_time * np.exp(final_xi)
    timestep = float(runparams.get('pre_crossing_timestep', 2.0e-4))
    tau = float(cosmology.supercomoving_time(initial_time))
    final_tau = float(cosmology.supercomoving_time(final_time))
    xi_history = [0.0]
    lambda_history = [1.0]
    turnaround_radius = float(icparams.get('pre_crossing_turnaround_radius', 1.0))
    while tau < final_tau - 1.0e-12:
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        rho_comoving = float(cosmology.background_density(cosmic_time)) * scale_factor**3
        background_coefficient = 4.0 * np.pi / 3.0 * rho_comoving
        background = lambda radius: background_coefficient * np.asarray(radius)**3
        dt = min(timestep, final_tau - tau)
        approaching = shells.velocity < 0.0
        if np.any(approaching):
            centre_dt = 0.05 * np.min(
                (shells.radius[approaching] + shells.softening)
                / np.maximum(-shells.velocity[approaching], 1.0e-30))
            dt = min(dt, centre_dt)
        next_time = float(cosmology.cosmic_time_from_supercomoving(tau + dt))
        actual_dt = shells.step(
            dt,
            crossing_safety_factor=float(runparams['crossing_safety_factor']),
            background_enclosed_mass=background,
            scale_factor=scale_factor,
            scale_factor_end=float(cosmology.scale_factor(next_time)),
            cosmological=True,
            include_shell_mass_with_fixed=True,
        )
        tau += actual_dt
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
        xi = np.log(cosmic_time / initial_time)
        physical_radius = float(cosmology.scale_factor(cosmic_time)) * shells.radius[tracked]
        similarity_turnaround = turnaround_radius * np.exp(8.0 * xi / 9.0)
        lam = physical_radius / similarity_turnaround
        if shells.radius[tracked] <= shells.softening:
            lam = match_lambda
        xi_history.append(float(xi))
        lambda_history.append(float(max(lam, 0.0)))
        if lam <= match_lambda:
            break
    ode = solve_eq41_self_similar(
        xi_end=max(final_xi, 0.9),
        points=int(runparams['ode_points']),
        similarity_exponent=float(runparams['ode_similarity_exponent']),
        centre_match_lambda=match_lambda,
        centre_matching_velocity=float(runparams['ode_centre_matching_velocity']),
    )
    figure = Path(runparams['savedir']) / 'BertschingerDarkMatterShellPreCrossingVsODE.jpg'
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(xi_history, lambda_history, linestyle='None', marker='s',
              markersize=2.5, color='tab:blue', markevery=8,
              label='RadHydropy tracked shell at $r_a$')
    axis.plot(ode.xi, ode.lam, color='black', linewidth=2.0,
              label='Bertschinger Eq. (4.1)')
    axis.set_xlim(0.0, max(final_xi, float(ode.xi[-1])))
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(r'$\xi=\ln(t/t_a)$')
    axis.set_ylabel(r'$\lambda=r/r_{ta}(t)$')
    axis.grid(alpha=0.25)
    axis.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    if not np.all(np.isfinite(lambda_history)):
        raise RuntimeError('tracked shell became non-finite')
    print('pre-crossing DarkMatterShells comparison generated')
    print('tracked shell index = %d' % tracked)
    print('first-centre xi = %.8g' % xi_history[-1])
    print('figure = %s' % figure)
    return figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    run_pre_crossing(parser.parse_args().config)
