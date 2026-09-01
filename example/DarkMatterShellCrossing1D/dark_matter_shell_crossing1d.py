"""Pure dark-matter spherical shells with self-gravity and crossings."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radhydropy.units import quantity_to_value
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'dark_matter_shell_crossing1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runtime = config['par']
    icparams = config['initial_condition']
    runparams = eu.legacy_example_parameters(config)
    runparams['timesim'] = runtime['simulation']['final_time']
    runparams['output_interval'] = runtime['timestep']['output_interval']
    runparams['crossing_safety_factor'] = runtime['timestep']['crossing_safety_factor']
    runparams['savedir'] = runtime['output']['savedir']
    runparams['CodeUnits'] = runtime['units']['CodeUnits']
    code_units = et.load_units(runparams)
    shells = et.make_shells(icparams, code_units)
    time = 0.0
    history_time = [time]
    history_radius = [shells.radius.copy()]
    history_energy = [np.sum(shells.mass * shells.specific_energy())]
    crossings = 0

    while time < runparams['timesim']:
        dt = min(
            float(runparams['output_interval']) / 4.0,
            float(runparams['timesim']) - time,
        )
        predicted = shells.crossing_timestep(
            safety_factor=float(runparams['crossing_safety_factor'])
        )
        if predicted < dt:
            crossings += 1
        actual_dt = shells.step(
            dt,
            crossing_safety_factor=float(runparams['crossing_safety_factor']),
        )
        time += actual_dt
        history_time.append(time)
        history_radius.append(shells.radius.copy())
        history_energy.append(np.sum(shells.mass * shells.specific_energy()))

    history_radius = np.asarray(history_radius)
    history_energy = np.asarray(history_energy)
    if not np.all(np.isfinite(history_radius)):
        raise RuntimeError('dark-matter shell radii became non-finite')
    if not np.all(np.diff(history_radius, axis=1) >= 0.0):
        raise RuntimeError('dark-matter shells are not sorted after evolution')
    if not np.isclose(np.sum(shells.mass), float(icparams['total_mass'])):
        raise RuntimeError('dark-matter shell mass was not conserved')

    radius_unit = code_units.length_unit
    radius_pc = quantity_to_value(history_radius * radius_unit, 'pc')
    time_myr = np.asarray(history_time) * code_units.time_unit.to_value('Myr')
    energy_fractional_change = np.abs(
        (history_energy - history_energy[0]) / max(abs(history_energy[0]), np.finfo(float).tiny)
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(time_myr, radius_pc)
    axes[0].set_xlabel('time [Myr]')
    axes[0].set_ylabel('sorted shell radius [pc]')
    axes[1].plot(time_myr, energy_fractional_change)
    axes[1].set_xlabel('time [Myr]')
    axes[1].set_ylabel('fractional diagnostic energy change')
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(runparams['savedir']) / 'DarkMatterShellCrossing1D.jpg'
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print('crossing-limited steps = %d' % crossings)
    print('maximum fractional diagnostic energy change = %.6g' % np.max(energy_fractional_change))
    print('figure = %s' % figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the pure dark-matter shell-crossing example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
