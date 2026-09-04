"""Fixed-radiation hydrogen photoionization box.

The gas starts neutral at ``T = 2e4 K`` and ``nH = 1 cm^-3``. A fixed,
spatially uniform photon number density photoionizes the gas while the
radiation-field evolution and thermal source update are disabled. The run
stops once the gas is 99 percent ionized, writes HDF5 snapshots, reloads them,
and plots the neutral-fraction evolution against the analytic fixed-field
solution.
"""

import os
import sys
from pathlib import Path
import tempfile
import argparse

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import unyt

from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import radhydropy.io as rio
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('hydrogen_photoionization1d.yaml')

def RunHydrogenPhotoionization(sim, target_neutral_fraction, outputtime=0):
    """Run the fixed-field photoionization example until neutral fraction falls."""
    return sim.RunAll(
        outputtime=outputtime,
        mode="hydro_sources",
        stop_condition=lambda runner: (
            et.mean_neutral_fraction(runner) <= target_neutral_fraction
        ),
    )


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    config = eu.load_nested_example_config(config_filename)
    runparams = config['par']
    ICparams = config['initial_condition']
    exampleparams = config['example']
    output = runparams['output']
    eu.clean_previous_outputs(output)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])

    config['_code_units'] = code_units_obj
    ric = et.build_initial_condition(config)
    rio.writehdf5(ric, runparams['simulation']['initial_condition_filename'])

    sim = Rsim(runparams)
    RunHydrogenPhotoionization(
        sim,
        exampleparams['target_neutral_fraction'],
        outputtime=0,
    )

    outputfiles = et.output_files(
        output['directory'],
        output['filename_prefix'],
    )
    history = et.load_history_from_outputs(
        outputfiles,
        config,
    )

    figure_filename = Path(output['savedir']) / exampleparams['plot_filename']
    et.save_history_plot(
        history,
        str(figure_filename),
        config,
        exampleparams['target_neutral_fraction'],
    )

    print('Hydrogen photoionization example finished')
    print('time = %.3e yr' % et.time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % et.mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % et.mean_neutral_fraction(sim))
    print(
        'mean photon number density = %.3e cm^-3'
        % et.mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
    )
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the fixed-radiation hydrogen photoionization example.',
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



