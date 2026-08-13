"""Einstein--de Sitter linear growth of collisionless dark-matter shells."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radhydropy.cosmology import EinsteinDeSitter
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name('einstein_de_sitter_dark_matter_shell_growth1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    from radhydropy.example_config import load_example_parameters

    runparams, icparams = load_example_parameters(config_filename, Path.cwd().resolve())
    units = et.load_units(runparams)
    cosmology = EinsteinDeSitter.from_code_units(
        units, t_ref=float(runparams['cosmology_t_ref']),
        a_ref=float(runparams['cosmology_a_ref']),
    )

    # First verify that the discretized homogeneous background has no peculiar force.
    homogeneous, _ = et.make_shells(icparams, units, cosmology, overdensity=0.0)
    tau = cosmology.supercomoving_time(float(icparams['cosmic_time']))
    a_initial = float(cosmology.scale_factor_from_supercomoving(tau))
    rho_comoving = float(cosmology.background_density(float(icparams['cosmic_time']))) * a_initial**3
    background_mass = 4.0 * np.pi / 3.0 * rho_comoving * homogeneous.radius**3
    homogeneous_acceleration = homogeneous.acceleration(
        background_enclosed_mass=background_mass,
        scale_factor=a_initial,
        cosmological=True,
    )
    homogeneous_error = float(np.max(np.abs(homogeneous_acceleration)))
    if homogeneous_error > float(runparams['homogeneous_acceleration_tolerance']):
        raise RuntimeError('homogeneous shell acceleration %.6g is nonzero' % homogeneous_error)

    shells, boundaries = et.make_shells(icparams, units, cosmology)
    top_hat_radius = float(icparams['top_hat_radius'])
    inside = shells.radius < top_hat_radius
    target_mass = float(np.sum(shells.mass[inside]))
    # The top-hat is an exact equal-volume boundary, so this is the actual
    # discretized initial perturbation used by the shell masses.
    lagrangian_radius = top_hat_radius
    lagrangian_velocity = -a_initial**2 * float(cosmology.hubble(float(icparams['cosmic_time']))) * float(icparams['overdensity']) * lagrangian_radius / 3.0
    initial_delta = et.overdensity_inside(lagrangian_radius, target_mass, rho_comoving)
    history_a = [a_initial]
    history_delta = [initial_delta]
    final_cosmic_time = float(runparams['final_cosmic_time'])
    final_tau = float(cosmology.supercomoving_time(final_cosmic_time))
    time = float(tau)
    dt = float(runparams['supercomoving_timestep'])
    while time < final_tau:
        step = min(dt, final_tau - time)
        time_end = time + step
        a_start = float(cosmology.scale_factor_from_supercomoving(time))
        a_end = float(cosmology.scale_factor_from_supercomoving(time_end))
        cosmic_start = float(cosmology.cosmic_time_from_supercomoving(time))
        rho_start = float(cosmology.background_density(cosmic_start)) * a_start**3
        background = 4.0 * np.pi / 3.0 * rho_start * shells.radius**3
        cosmic_end = float(cosmology.cosmic_time_from_supercomoving(time_end))
        rho_end = float(cosmology.background_density(cosmic_end)) * a_end**3
        shells.step(
            step,
            crossing_safety_factor=float(runparams['crossing_safety_factor']),
            background_enclosed_mass=background,
            scale_factor=a_start,
            scale_factor_end=a_end,
            cosmological=True,
        )
        lagrangian_radius, lagrangian_velocity = et.step_lagrangian_boundary(
            lagrangian_radius, lagrangian_velocity, step, target_mass,
            rho_start, rho_end, a_start, a_end, units,
        )
        time = time_end
        history_a.append(a_end)
        history_delta.append(et.overdensity_inside(lagrangian_radius, target_mass, rho_end))

    expected = initial_delta * history_a[-1] / a_initial
    relative_error = abs(history_delta[-1] - expected) / abs(expected)
    if not np.isfinite(relative_error) or relative_error > float(runparams['growth_tolerance']):
        raise RuntimeError('dark-matter linear growth error %.6g exceeds tolerance' % relative_error)
    if not np.all(np.isfinite(shells.radius)) or not np.all(np.diff(shells.radius) >= 0.0):
        raise RuntimeError('dark-matter shells became invalid or unsorted')

    figure = Path(runparams['savedir']) / 'EinsteinDeSitterDarkMatterShellGrowth1D.jpg'
    a_plot = np.linspace(a_initial, history_a[-1], 200)
    plt.figure(figsize=(6, 4))
    plt.plot(history_a, history_delta, label='shells')
    plt.plot(a_plot, initial_delta * a_plot / a_initial, '--', label='linear theory')
    plt.xlabel('scale factor $a$')
    plt.ylabel('dark-matter overdensity $\\delta_{DM}$')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=200)
    plt.close()
    print('Einstein-De Sitter dark-matter shell growth passed')
    print('homogeneous acceleration max = %.6g' % homogeneous_error)
    print('delta: %.8g measured, %.8g linear, relative error %.6g' %
          (history_delta[-1], expected, relative_error))
    print('figure = %s' % figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
