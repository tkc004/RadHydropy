"""Compare C²-Ray and instantaneous Strömgren-sphere front propagation."""

import argparse
import csv
import importlib.util
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))
static_example = Path(__file__).resolve().parents[1] / 'StaticStromgrenSphere1D'
if str(static_example) not in sys.path:
    sys.path.insert(0, str(static_example))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

from radhydropy.rsim import Rsim
import radhydropy.io as rio
import stromgren_analytic as sa
import example_utils as eu

STATIC_EXAMPLE = static_example


def _load_static_tools():
    tools_path = STATIC_EXAMPLE / 'tools.py'
    spec = importlib.util.spec_from_file_location('static_stromgren_tools', tools_path)
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    return tools


def _run_case(base_runtime, initial_condition, tools, label, scheme, steps, root):
    par_case = deepcopy(base_runtime)
    initial_condition_case = deepcopy(initial_condition)
    case_dir = root / label
    case_dir.mkdir(parents=True, exist_ok=True)
    par_case['simulation']['name'] = f'StaticStromgren_{label}'
    par_case['output']['directory'] = str(case_dir)
    par_case['output']['savedir'] = str(case_dir)
    par_case['output']['filename_prefix'] = 'Output'
    par_case['simulation']['initial_condition_filename'] = str(case_dir / 'InitialCondition.hdf5')
    par_case['radiation']['radiative_transfer_temporal_scheme'] = scheme
    par_case['timestep']['chemistry_timestep'] = par_case['simulation']['final_time'] / steps
    case_config = {
        'par': par_case,
        'initial_condition': initial_condition_case,
        'example': {},
    }
    tools.write_initial_condition(case_config)

    sim = Rsim(par_case)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    history = sim.EvolveStaticThermochemistry(
        par_case['simulation']['final_time'],
        par_case['timestep']['chemistry_timestep'],
    )
    output_filename = case_dir / 'Output_000.hdf5'
    rio.writehdf5(sim, output_filename)
    return history


def _plot(histories, config, filename):
    fig, (ax, ax_difference) = plt.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.08},
    )
    styles = {
        'c2ray_100': ('tab:red', '-', 'C²-Ray (100 steps)'),
        'instantaneous_100': ('tab:blue', '--', 'Instantaneous (100 steps)'),
        'instantaneous_1000': ('tab:orange', '--', 'Instantaneous (1,000 steps)'),
        'instantaneous_10000': ('tab:green', '--', 'Instantaneous (10,000 steps)'),
        'instantaneous_100000': ('tab:purple', '--', 'Instantaneous (100,000 steps)'),
    }
    for label, history in histories.items():
        color, linestyle, legend = styles[label]
        ax.plot(history['time_Myr'], history['front_radius_kpc'], color=color,
                lw=1.8, ls=linestyle, label=legend)
    reference = histories['instantaneous_100000']
    reference_time = np.asarray(reference['time_Myr'])
    reference_radius = np.asarray(reference['front_radius_kpc'])
    for label, history in histories.items():
        color, linestyle, _ = styles[label]
        time_samples = np.asarray(history['time_Myr'])
        radius_samples = np.asarray(history['front_radius_kpc'])
        reference_at_time = np.interp(time_samples, reference_time, reference_radius)
        relative_difference = np.zeros_like(radius_samples)
        nonzero = reference_at_time > 0.0
        relative_difference[nonzero] = (
            radius_samples[nonzero] - reference_at_time[nonzero]
        ) / reference_at_time[nonzero]
        ax_difference.plot(
            time_samples, relative_difference, color=color,
            lw=1.5, ls=linestyle, label=styles[label][2],
        )
    initial = config['initial_condition']
    example = config['example']
    time = np.linspace(0.0, config['par']['simulation']['final_time'].to_value(unyt.Myr), 1200) * unyt.Myr
    analytic = sa.ionization_front_radius(
        time, example['source_photon_rate'], initial['hydrogen_number_density'],
        example['alpha_B_coefficient'],
    ).to_value(unyt.kpc)
    radius_stromgren = sa.stromgren_radius(
        example['source_photon_rate'], initial['hydrogen_number_density'],
        example['alpha_B_coefficient'],
    ).to_value(unyt.kpc)
    ax.plot(time.to_value(unyt.Myr), analytic, 'k-', lw=2.0, label='Analytic $R_I(t)$')
    ax.axhline(radius_stromgren, color='0.3', lw=1.0, ls=':', label='$R_S$')
    ax_difference.axhline(0.0, color='0.3', lw=1.0, ls=':', label='Reference')
    ax.set_ylabel('Ionization-front radius [kpc]')
    ax_difference.set(xlabel='Time [Myr]', ylabel=r'$(R-R_{100000})/R_{100000}$')
    ax.set_xlim(0.0, time[-1].to_value(unyt.Myr))
    ax.set_ylim(0.0, example['plot_radius_max'].to_value(unyt.kpc))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax_difference.grid(True, alpha=0.25)
    ax_difference.legend(
        frameon=False, fontsize=8, loc='center left',
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.subplots_adjust(left=0.11, right=0.80, bottom=0.09, top=0.98, hspace=0.08)
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _write_summary(histories, config, filename):
    initial = config['initial_condition']
    example = config['example']
    analytic_final = sa.ionization_front_radius(
        config['par']['simulation']['final_time'], example['source_photon_rate'],
        initial['hydrogen_number_density'], example['alpha_B_coefficient'],
    ).to_value(unyt.kpc)
    with open(filename, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'time_Myr', 'front_radius_kpc', 'analytic_radius_kpc', 'absolute_error_kpc'])
        for label, history in histories.items():
            for time, radius in zip(history['time_Myr'], history['front_radius_kpc']):
                analytic = sa.ionization_front_radius(
                    time * unyt.Myr, example['source_photon_rate'],
                    initial['hydrogen_number_density'], example['alpha_B_coefficient'],
                ).to_value(unyt.kpc)
                writer.writerow([label, time, radius, analytic, abs(radius - analytic)])
            final_error = abs(history['front_radius_kpc'][-1] - analytic_final)
            print(f'{label}: final front = {history["front_radius_kpc"][-1]:.6f} kpc, '
                  f'absolute error = {final_error:.6e} kpc')


def main(config_filename=Path(__file__).with_name('static_stromgren_c2ray_comparison.yaml')):
    nested = eu.load_nested_example_config(config_filename)
    runtime = nested['par']
    initial_condition = nested['initial_condition']
    example = nested['example']
    root = Path(runtime['output']['savedir']) / 'comparison_runs'
    root.mkdir(parents=True, exist_ok=True)
    tools = _load_static_tools()
    config = nested
    histories = {}
    c2ray_steps = int(example['comparison_c2ray_steps'])
    histories[f'c2ray_{c2ray_steps}'] = _run_case(
        runtime, initial_condition, tools, f'c2ray_{c2ray_steps}', 'c2ray', c2ray_steps, root,
    )
    for steps in example['comparison_instantaneous_steps']:
        steps = int(steps)
        histories[f'instantaneous_{steps}'] = _run_case(
            runtime, initial_condition, tools, f'instantaneous_{steps}',
            'instantaneous', steps, root,
        )
    figure = Path(runtime['output']['savedir']) / 'StaticStromgrenC2RayComparison_IFront.jpg'
    summary = Path(runtime['output']['savedir']) / 'StaticStromgrenC2RayComparison_IFront.csv'
    _plot(histories, config, figure)
    _write_summary(histories, config, summary)
    print(f'comparison figure = {figure}')
    print(f'comparison data = {summary}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=Path(__file__).with_name('static_stromgren_c2ray_comparison.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
