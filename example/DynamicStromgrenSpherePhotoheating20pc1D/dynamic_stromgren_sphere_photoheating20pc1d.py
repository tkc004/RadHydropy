"""Photoheated Stromgren sphere in a 20 pc, 100 cm^-3 cloud.

This is a compact variant of the DynamicStromgrenSpherePhotoheating1D
example.  It uses the same tested workflow and helper implementation while
keeping its configuration and generated outputs separate.
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
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'dynamic_stromgren_sphere_photoheating20pc1d.yaml'
)


def main(config_filename=None):
    if config_filename is None:
        config_filename = DEFAULT_CONFIG
    config = eu.load_nested_example_config(config_filename)
    runtime_params = config['par']
    output = runtime_params['output']
    config['_code_units'] = CodeUnits.from_mapping(
        runtime_params['units']['CodeUnits']
    )

    Path(output['directory']).mkdir(parents=True, exist_ok=True)
    Path(output['savedir']).mkdir(parents=True, exist_ok=True)
    et.write_initial_condition(config)

    sim = Rsim(runtime_params)
    sim.RunAll(outputtime=0)

    outputfilenames = et.output_files(
        output['directory'], output['filename_prefix']
    )
    history = et.load_history_from_outputs(outputfilenames, config)
    out_par, out_mesh, out_fluid = et.load_output_state(
        outputfilenames[-1], config
    )
    figure_stem = 'DynamicStromgrenSpherePhotoheating20pc1D'
    if runtime_params['radiation'].get(
        'radiative_transfer_temporal_scheme'
    ) == 'c2ray':
        figure_stem += '_C2Ray'
    figure_filename = Path(output['savedir']) / f'{figure_stem}.jpg'
    front_figure_filename = Path(output['savedir']) / f'{figure_stem}_IFront.jpg'
    et.save_plot(out_mesh, out_fluid, out_par, config, figure_filename)
    et.save_front_plot(history, config, front_figure_filename)

    rhd_csv_filename = Path(output['directory']) / 'radial_profile_rhd.csv'
    eu.write_radial_profile_csv(outputfilenames[-1], rhd_csv_filename)
    print('output files = %d' % len(outputfilenames))
    print('final front radius = %.3e kpc' % history['front_radius_kpc'][-1])
    print('RHD profile CSV = %s' % rhd_csv_filename)
    print('figure = %s' % figure_filename)
    print('front figure = %s' % front_figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the 20 pc dynamic photoheated Stromgren sphere example.',
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
