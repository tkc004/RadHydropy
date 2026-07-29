import argparse
import os
import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.example_config import load_example_parameters
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import radhydropy.io as rio
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('hydrostatic_equilibrium1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams = load_example_parameters(config_filename, rundir)
    eu.clean_previous_outputs(runparams)
    code_units = CodeUnits.from_mapping(runparams.get('CodeUnits'))

    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric, runparams['ICfilename'])

    mainrun = Rsim(runparams)
    mainrun.par.gravity = Gravity(
        externalgravity=True,
        acceleration=et.constant_gravity_acceleration(
            ICparams['gravity_strength'],
            code_units=code_units,
        ),
        code_units=code_units,
    )
    mainrun.RunAll(outputtime=0, mode='hydro')

    final_outfile = os.path.join(
        runparams['outdir'],
        runparams['outfileprefix'] + '_001.hdf5',
    )
    et.ReadandPlot(
        final_outfile,
        ICparams,
        runparams,
        ls='none',
        marker='o',
        mfc='none',
        markevery=1,
        color='C0',
    )
    figure_filename = os.path.join(runparams['savedir'], 'HydrostaticEquilibrium1D.jpg')
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the hydrostatic-equilibrium gravity check example.'
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file with runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
