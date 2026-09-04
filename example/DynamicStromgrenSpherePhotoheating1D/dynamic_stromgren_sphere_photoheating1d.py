"""Photoheated Stromgren sphere with hydrodynamic expansion.

The example is configured from YAML, writes HDF5 snapshots during the run,
reloads those saved outputs, and plots the final profiles and ionization-front
history from the on-disk snapshots rather than from live simulation state.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
example_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

from radhydropy.rsim import Rsim
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'dynamic_stromgren_sphere_photoheating1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, icparams = et.load_parameters(config_filename, rundir)
    config = {**runparams, **icparams}
    nested_config = eu.load_nested_example_config(config_filename)
    runtime_params = eu.runtime_parameters(nested_config)

    Path(runparams['outdir']).mkdir(parents=True, exist_ok=True)
    Path(runparams['savedir']).mkdir(parents=True, exist_ok=True)

    et.write_initial_condition(config, runparams)

    sim = Rsim(runtime_params)
    sim.RunAll(outputtime=0)

    outputfilenames = et.output_files(runparams['outdir'], runparams['outfileprefix'])
    history = et.load_history_from_outputs(outputfilenames, config)
    out_par, out_mesh, out_fluid = et.load_output_state(outputfilenames[-1], config)

    figure_stem = 'DynamicStromgrenSpherePhotoheating1D'
    if runparams.get('radiative_transfer_temporal_scheme') == 'c2ray':
        figure_stem += '_C2Ray'
    figure_filename = Path(runparams['savedir']) / f'{figure_stem}.jpg'
    front_figure_filename = (
        Path(runparams['savedir']) / f'{figure_stem}_IFront.jpg'
    )
    et.save_plot(out_mesh, out_fluid, out_par, config, figure_filename)
    et.save_front_plot(history, config, front_figure_filename)

    print('time = %s' % out_fluid.time)
    print('output files = %d' % len(outputfilenames))
    print('final front radius = %.3e kpc' % history['front_radius_kpc'][-1])
    print('mean ionized temperature = %.3e K' % history['mean_ionized_temperature_cgs_K'][-1])
    print('IC file = %s' % runparams['ICfilename'])
    for outputfilename in outputfilenames:
        print('output file = %s' % outputfilename)
    print('figure = %s' % figure_filename)
    print('front figure = %s' % front_figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the dynamic photoheated Stromgren sphere example.',
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file containing runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
