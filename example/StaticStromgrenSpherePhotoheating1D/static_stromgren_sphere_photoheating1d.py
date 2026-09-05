"""Static Stromgren sphere with photoheating.

This repeats the static Stromgren sphere benchmark, but lets the hydrogen
source update heat and cool the gas. Hydrodynamic motion is disabled: density
is fixed and only radiative transfer, chemistry, and thermal source terms are
advanced. The example is configured from YAML, writes HDF5 snapshots, reloads
the final snapshot, and plots from the saved output rather than live state.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
static_stromgren_dir = Path(__file__).resolve().parents[1] / 'StaticStromgrenSphere1D'
if str(static_stromgren_dir) not in sys.path:
    sys.path.append(str(static_stromgren_dir))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import unyt

from radhydropy.rsim import Rsim
import radhydropy.io as rio
import stromgren_analytic as sa
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'static_stromgren_sphere_photoheating1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    nested = eu.load_nested_example_config(config_filename)
    runtime = nested['par']
    config = nested
    initial = config['initial_condition']
    example = config.get('example', {})
    eu.clean_previous_outputs(runtime['output'])
    config_dir = Path(config_filename).resolve().parent
    for key in (
        'temperature_reference_filename',
        'neutral_fraction_reference_filename',
    ):
        if key in example:
            value = Path(example[key])
            if not value.is_absolute():
                example[key] = str(config_dir / value)

    Path(runtime['output']['directory']).mkdir(parents=True, exist_ok=True)
    Path(runtime['output']['savedir']).mkdir(parents=True, exist_ok=True)

    et.write_initial_condition(config)

    mainrun = Rsim(runtime)
    mainrun.Callreadhdf5()
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()

    history = mainrun.EvolveStaticThermochemistry(
        runtime['simulation']['final_time'],
        runtime['timestep']['evolution_timestep'],
        include_thermal_history=True,
        reference_time=example['reference_time'],
    )

    output_filename = Path(runtime['output']['directory']) / f"{runtime['output']['filename_prefix']}_000.hdf5"
    rio.writehdf5(mainrun, output_filename)

    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    figure_name = 'StaticStromgrenSpherePhotoheating1D.jpg'
    if runtime['radiation'].get('radiative_transfer_temporal_scheme') == 'c2ray':
        figure_name = 'StaticStromgrenSpherePhotoheating1D_C2Ray.jpg'
    figure_filename = Path(runtime['output']['savedir']) / figure_name
    et.save_plot(out_mesh, out_fluid, out_par, history, config, figure_filename)

    print('time = %s' % out_fluid.time_code)
    if runtime['thermochemistry'].get('hydrogen_alpha_B') is None:
        print('stromgren radius = temperature-dependent alpha_H(T)')
        print('analytic front radius = unavailable for temperature-dependent rates')
    else:
        print(
            'stromgren radius = %s'
            % sa.stromgren_radius(
                runtime['radiation']['radiative_transfer_source_photon_rate'],
                initial['hydrogen_number_density'],
                runtime['thermochemistry']['hydrogen_alpha_B'],
            ).to(unyt.kpc)
        )
        print(
            'analytic front radius = %s'
            % sa.ionization_front_radius(
                runtime['simulation']['final_time'],
                runtime['radiation']['radiative_transfer_source_photon_rate'],
                initial['hydrogen_number_density'],
                runtime['thermochemistry']['hydrogen_alpha_B'],
            ).to(unyt.kpc)
        )
    print('mean ionized temperature = %.3e K' % history['mean_ionized_temp_cgs_K'][-1])
    print('front radius = %.3e kpc' % history['front_radius_kpc'][-1])
    print('evolution steps = %d' % history['evolution_steps'])
    print('IC file = %s' % runtime['simulation']['initial_condition_filename'])
    print('output file = %s' % output_filename)
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the static Stromgren sphere photoheating example.',
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
