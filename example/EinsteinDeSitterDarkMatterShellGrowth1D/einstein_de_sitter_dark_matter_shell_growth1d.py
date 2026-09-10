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
from radhydropy.units import quantity_to_value
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name('einstein_de_sitter_dark_matter_shell_growth1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)

    initial_condition = config['initial_condition']
    units = et.code_units_from_config(config)
    gravity = config["par"]['gravity']
    timestep = config["par"]['timestep']
    example = config.get('example', {})
    cosmology = EinsteinDeSitter.from_code_units(
        units, t_ref=quantity_to_value(gravity['cosmology_t_ref'], units.time_unit),
        a_ref=float(gravity['cosmology_a_ref']),
    )

    # First verify that the discretized homogeneous background has no peculiar force.
    homogeneous, _ = et.make_shells(config, overdensity=0.0)
    initial_time = quantity_to_value(initial_condition['time_cosmic'], units.time_unit)
    tau = cosmology.supercomoving_time(initial_time)
    a_initial = float(cosmology.scale_factor_from_supercomoving(tau))
    rho_comoving = float(cosmology.background_density(initial_time)) * a_initial**3
    background_mass = 4.0 * np.pi / 3.0 * rho_comoving * homogeneous.radius**3
    homogeneous_acceleration = homogeneous.acceleration(
        background_enclosed_mass=background_mass,
        scale_factor=a_initial,
        cosmological=True,
    )
    homogeneous_error = float(np.max(np.abs(homogeneous_acceleration)))
    if homogeneous_error > float(example['homogeneous_acceleration_tolerance']):
        raise RuntimeError('homogeneous shell acceleration %.6g is nonzero' % homogeneous_error)

    shells, boundaries = et.make_shells(config)
    radius_top_hat_comoving_code = quantity_to_value(
        initial_condition['radius_perturbation_comoving'], units.length_unit
    )
    inside = shells.radius < radius_top_hat_comoving_code
    target_mass = float(np.sum(shells.mass[inside]))
    # The top-hat is an exact equal-volume boundary, so this is the actual
    # discretized initial perturbation used by the shell masses.
    lagrangian_radius_comoving_code = radius_top_hat_comoving_code
    lagrangian_velocity = -a_initial**2 * float(cosmology.hubble(initial_time)) * float(initial_condition['overdensity']) * lagrangian_radius_comoving_code / 3.0
    initial_delta = et.overdensity_inside(lagrangian_radius_comoving_code, target_mass, rho_comoving)
    history_a = [a_initial]
    history_delta = [initial_delta]
    final_cosmic_time_code = quantity_to_value(
        config["par"]['simulation']['final_time'], units.time_unit
    )
    final_tau = float(cosmology.supercomoving_time(final_cosmic_time_code))
    time_supercomoving_code = float(tau)
    dt = float(timestep['supercomoving_timestep'])
    while time_supercomoving_code < final_tau:
        step = min(dt, final_tau - time_supercomoving_code)
        time_supercomoving_end_code = time_supercomoving_code + step
        a_start = float(cosmology.scale_factor_from_supercomoving(time_supercomoving_code))
        a_end = float(cosmology.scale_factor_from_supercomoving(time_supercomoving_end_code))
        cosmic_start = float(cosmology.cosmic_time_from_supercomoving(time_supercomoving_code))
        rho_start = float(cosmology.background_density(cosmic_start)) * a_start**3
        background = 4.0 * np.pi / 3.0 * rho_start * shells.radius**3
        cosmic_end = float(cosmology.cosmic_time_from_supercomoving(time_supercomoving_end_code))
        rho_end = float(cosmology.background_density(cosmic_end)) * a_end**3
        shells.step(
            step,
            crossing_safety_factor=float(timestep['crossing_safety_factor']),
            background_enclosed_mass=background,
            scale_factor=a_start,
            scale_factor_end=a_end,
            cosmological=True,
        )
        lagrangian_radius_comoving_code, lagrangian_velocity = et.step_lagrangian_boundary(
            lagrangian_radius_comoving_code, lagrangian_velocity, step, target_mass,
            rho_start, rho_end, a_start, a_end, units,
        )
        time_supercomoving_code = time_supercomoving_end_code
        history_a.append(a_end)
        history_delta.append(et.overdensity_inside(lagrangian_radius_comoving_code, target_mass, rho_end))

    expected = initial_delta * history_a[-1] / a_initial
    relative_error = abs(history_delta[-1] - expected) / abs(expected)
    if not np.isfinite(relative_error) or relative_error > float(example['growth_tolerance']):
        raise RuntimeError('dark-matter linear growth error %.6g exceeds tolerance' % relative_error)
    if not np.all(np.isfinite(shells.radius)) or not np.all(np.diff(shells.radius) >= 0.0):
        raise RuntimeError('dark-matter shells became invalid or unsorted')

    figure = Path(config["par"]['output']['directory']) / 'EinsteinDeSitterDarkMatterShellGrowth1D.jpg'
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
