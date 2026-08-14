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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.example_config import load_example_parameters
import tools as example_tools
from bertschinger_ode import solve_eq41_self_similar


DEFAULT_CONFIG = Path(__file__).with_name('bertschinger_reference.yaml')


def _turnaround_radius(shells, cosmic_time, cosmology):
    """Return the first velocity sign-change radius, or ``None``."""
    radius = float(cosmology.scale_factor(cosmic_time)) * shells.radius
    velocity = example_tools.physical_velocity(shells, cosmic_time, cosmology)
    crossing = np.flatnonzero(velocity[:-1] * velocity[1:] <= 0.0)
    if crossing.size == 0:
        return None
    # After shell crossing there can be several sign changes. The
    # Bertschinger turnaround scale is the outermost infall/expansion
    # interface, not the innermost central bounce.
    index = int(crossing[-1])
    denominator = velocity[index] - velocity[index + 1]
    if denominator == 0.0:
        return float(radius[index])
    fraction = velocity[index] / denominator
    return float(radius[index] + fraction * (radius[index + 1] - radius[index]))


def run_comparison(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_example_parameters(config_filename, Path.cwd().resolve())
    units = example_tools.load_units(runparams)
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams['cosmology_t_ref']),
        a_ref=float(runparams['cosmology_a_ref']),
    )
    shells, _ = example_tools.make_scale_free_shells(icparams, units, cosmology)
    initial_time = float(icparams['initial_cosmic_time'])
    final_time = float(runparams.get(
        'comparison_final_cosmic_time',
        initial_time * np.exp(float(runparams['ode_xi_end']))))
    timestep = float(runparams.get('comparison_timestep', 0.002))
    snapshot_stride = int(runparams.get('comparison_snapshot_stride', 5))
    shell_stride = int(runparams.get('comparison_shell_stride', 16))
    if final_time <= initial_time or timestep <= 0.0:
        raise ValueError('invalid shell comparison time configuration')

    tau = float(cosmology.supercomoving_time(initial_time))
    final_tau = float(cosmology.supercomoving_time(final_time))
    snapshot = 0
    xi_values = []
    lambda_values = []
    time_values = []
    turnaround_values = []
    while tau < final_tau - 1.0e-12:
        dt = min(timestep, final_tau - tau)
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
        next_time = float(cosmology.cosmic_time_from_supercomoving(tau + dt))
        a_start = float(cosmology.scale_factor(cosmic_time))
        a_end = float(cosmology.scale_factor(next_time))
        rho_comoving = float(cosmology.background_density(cosmic_time)) * a_start**3
        background = 4.0 * np.pi / 3.0 * rho_comoving * shells.radius**3
        actual_dt = shells.step(
            dt,
            crossing_safety_factor=float(runparams['crossing_safety_factor']),
            background_enclosed_mass=background,
            scale_factor=a_start,
            scale_factor_end=a_end,
            cosmological=True,
            include_shell_mass_with_fixed=True,
        )
        # Radial shells have no centrifugal barrier. Match the ODE's
        # controlled centre treatment after a finite leapfrog step.
        central_shell = shells.radius <= float(icparams['softening'])
        if np.any(central_shell):
            shells.radius[central_shell] = float(icparams['softening'])
            shells.velocity[central_shell] = np.abs(
                shells.velocity[central_shell])
            shells.sort_by_radius()
        tau += actual_dt
        snapshot += 1
        if snapshot % snapshot_stride:
            continue
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
        turnaround = _turnaround_radius(shells, cosmic_time, cosmology)
        if turnaround is None or not np.isfinite(turnaround):
            continue
        radius = float(cosmology.scale_factor(cosmic_time)) * shells.radius
        selected = slice(None, None, shell_stride)
        xi_values.extend(np.full(radius[selected].shape,
                                 np.log(cosmic_time / initial_time)))
        lambda_values.extend((radius[selected] / turnaround).tolist())
        time_values.extend(np.full(radius[selected].shape, cosmic_time))
        turnaround_values.append((cosmic_time, turnaround))

    if not xi_values:
        raise RuntimeError('the shell simulation produced no turnaround samples')
    ode = solve_eq41_self_similar(
        xi_end=float(runparams['ode_xi_end']),
        points=int(runparams['ode_points']),
        similarity_exponent=float(runparams['ode_similarity_exponent']),
        centre_match_lambda=float(runparams['ode_centre_match_lambda']),
        centre_matching_velocity=float(runparams['ode_centre_matching_velocity']),
    )
    xi_values = np.asarray(xi_values)
    lambda_values = np.asarray(lambda_values)
    finite = np.isfinite(xi_values) & np.isfinite(lambda_values) & (lambda_values > 0.0)
    lambda_max = float(runparams.get('comparison_lambda_max', 2.0))
    finite &= lambda_values <= lambda_max
    figure = Path(runparams['savedir']) / 'BertschingerDarkMatterShellsVsODE.jpg'
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(xi_values[finite], lambda_values[finite], s=1.0, alpha=0.12,
                 label='RadHydropy DarkMatterShells')
    axis.plot(ode.xi, ode.lam, color='black', linewidth=2.0,
              label='Bertschinger Eq. (4.1) ODE')
    axis.set_xlim(0.0, float(runparams['ode_xi_end']))
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
    print('simulation time range = %.8g .. %.8g' % (initial_time, final_time))
    print('figure = %s' % figure)
    return figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    run_comparison(parser.parse_args().config)
