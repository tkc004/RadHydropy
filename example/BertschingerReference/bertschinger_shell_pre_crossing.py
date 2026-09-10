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
from radhydropy.units import quantity_to_value
import tools as example_tools
from bertschinger_ode import solve_eq41_self_similar


DEFAULT_CONFIG = Path(__file__).with_name('bertschinger_reference.yaml')


def make_turnaround_shells(config):
    """Create background interior shells and one tracked shell at ``r_a``."""
    # Isolate the Lagrangian turnaround shell for the pre-crossing benchmark.
    initial_condition = config['initial_condition']
    code_unit_system = config['_code_units']
    cosmology = config['_cosmology']
    radius_turnaround_comoving_code = float(
        initial_condition.get('pre_crossing_turnaround_radius', 1.0)
    )
    time_cosmic_code = quantity_to_value(
        initial_condition['time_cosmic'], code_unit_system.time_unit
    )
    scale_factor_dimensionless = float(cosmology.scale_factor(time_cosmic_code))
    hubble_code = float(cosmology.hubble(time_cosmic_code))
    background_coefficient = 2.0 / (9.0 * cosmology.gravitational_constant)
    turnaround_mass = (9.0 * np.pi * np.pi / 16.0) * background_coefficient
    fixed_total_mass = lambda radius_comoving_code: (
        turnaround_mass + background_coefficient * np.asarray(radius_comoving_code) ** 3
    )
    shells = DarkMatterShells(
                              radius=np.asarray([radius_turnaround_comoving_code]),
                              velocity=np.asarray([-scale_factor_dimensionless**2 * hubble_code * radius_turnaround_comoving_code]),
                              mass=np.asarray([1.0e-12]),
                              fixed_enclosed_mass=fixed_total_mass,
                              softening=float(initial_condition.get('pre_crossing_softening', 1.0e-3)),
                              code_units=code_unit_system)
    return shells, 0
def run_pre_crossing(config_filename=DEFAULT_CONFIG):
    config = example_tools.load_reference_config(config_filename)

    initial_condition = config['initial_condition']
    example = config['example']
    units = example_tools.code_units_from_config(config)
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=quantity_to_value(example['cosmology_t_ref'], units.time_unit),
        a_ref=float(example['cosmology_a_ref']),
    )
    config['_code_units'] = units
    config['_cosmology'] = cosmology
    shells, tracked = make_turnaround_shells(config)
    initial_time = quantity_to_value(initial_condition['time_cosmic'], units.time_unit)
    final_xi = float(example.get('pre_crossing_final_xi', 0.9))
    match_lambda = float(example.get('pre_crossing_match_lambda', 0.002))
    final_time = initial_time * np.exp(final_xi)
    timestep = float(example.get('pre_crossing_timestep', 2.0e-4))
    tau = float(cosmology.supercomoving_time(initial_time))
    final_tau = float(cosmology.supercomoving_time(final_time))
    xi_history = [0.0]
    lambda_history = [1.0]
    radius_turnaround_comoving_code = float(
        initial_condition.get('pre_crossing_turnaround_radius', 1.0)
    )
    while tau < final_tau - 1.0e-12:
        time_cosmic_code = float(cosmology.cosmic_time_from_supercomoving(tau))
        scale_factor_dimensionless = float(cosmology.scale_factor(time_cosmic_code))
        rho_comoving_code = float(cosmology.background_density(time_cosmic_code)) * scale_factor_dimensionless**3
        background_coefficient = 4.0 * np.pi / 3.0 * rho_comoving_code
        background = lambda radius_comoving_code: background_coefficient * np.asarray(radius_comoving_code)**3
        dt = min(timestep, final_tau - tau)
        approaching = shells.velocity < 0.0
        if np.any(approaching):
            centre_dt = 0.05 * np.min(
                (shells.radius[approaching] + shells.softening)
                / np.maximum(-shells.velocity[approaching], 1.0e-30))
            dt = min(dt, centre_dt)
        next_time_cosmic_code = float(cosmology.cosmic_time_from_supercomoving(tau + dt))
        actual_dt = shells.step(
            dt,
            crossing_safety_factor=float(example['crossing_safety_factor']),
            background_enclosed_mass=background,
            scale_factor=scale_factor_dimensionless,
            scale_factor_end=float(cosmology.scale_factor(next_time_cosmic_code)),
            cosmological=True,
            include_shell_mass_with_fixed=True,
        )
        tau += actual_dt
        time_cosmic_code = float(cosmology.cosmic_time_from_supercomoving(tau))
        xi = np.log(time_cosmic_code / initial_time)
        radius_proper_code = float(cosmology.scale_factor(time_cosmic_code)) * shells.radius[tracked]
        radius_turnaround_proper_code = radius_turnaround_comoving_code * np.exp(8.0 * xi / 9.0)
        lambda_dimensionless = radius_proper_code / radius_turnaround_proper_code
        if shells.radius[tracked] <= shells.softening:
            lambda_dimensionless = match_lambda
        xi_history.append(float(xi))
        lambda_history.append(float(max(lambda_dimensionless, 0.0)))
        if lambda_dimensionless <= match_lambda:
            break
    ode = solve_eq41_self_similar(
        xi_end=max(final_xi, 0.9),
        points=int(example['ode_points']),
        similarity_exponent=float(example['ode_similarity_exponent']),
        centre_match_lambda=match_lambda,
        centre_matching_velocity=float(example['ode_centre_matching_velocity']),
    )
    figure = Path(config["par"]['output']['directory']) / 'BertschingerDarkMatterShellPreCrossingVsODE.jpg'
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
