"""Fixed-temperature case-B hydrogen recombination box.

The gas starts fully ionized at ``T = 2e4 K``. Hydrogen cooling/heating terms
and collisional ionization are disabled, leaving pure case-B recombination.
The run stops once the gas is 99 percent neutral and writes a JPG comparing
the ionized fraction against the analytic case-B expectation.
"""

import argparse
import os
import sys
from pathlib import Path
import tempfile

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault(
    'MPLCONFIGDIR',
    mplconfig_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unyt
import yaml

from radhydropy.rsim import Rsim
from radhydropy.example_config import load_example_parameters
import radhydropy.io as rio
import tools as et

DEFAULT_CONFIG = Path(__file__).resolve().with_name('hydrogen_recombination1d.yaml')


def load_parameters(config_filename=DEFAULT_CONFIG, rundir=None):
    config_filename = Path(config_filename)
    runparams, ICparams = load_example_parameters(config_filename, rundir)
    with config_filename.open() as config_file:
        config = yaml.safe_load(config_file)
    return runparams, ICparams, config.get('target_neutral_fraction', 0.99)


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams, target_neutral_fraction = load_parameters(
        config_filename,
        rundir,
    )

    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric, runparams['ICfilename'])

    sim = Rsim(runparams)
    et.run_hydrogen_recombination(sim, target_neutral_fraction, outputtime=0)

    outputfiles = et.output_files(
        runparams['outdir'],
        runparams['outfileprefix'],
    )
    history = et.load_history_from_outputs(
        outputfiles,
        ICparams,
        runparams['noghost'],
    )

    figure_filename = rundir / 'HydrogenRecombination1D.jpg'
    et.save_history_plot(
        history,
        str(figure_filename),
        ICparams,
        target_neutral_fraction,
    )

    print('Hydrogen recombination example finished')
    print('time = %.3e yr' % et.time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % et.mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % et.mean_neutral_fraction(sim))
    print('mean ionized fraction = %.3e' % et.mean_ionized_fraction(sim))
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the 1D hydrogen recombination example.',
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file containing runparams, ICparams, and target_neutral_fraction.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
