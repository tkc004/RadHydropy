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
    config = eu.load_nested_example_config(config_filename)
    runparams = config['par']
    ICparams = config['initial_condition']
    eu.clean_previous_outputs(runparams)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])

    config['_code_units'] = code_units_obj
    ric = et.build_initial_condition(config)
    rio.writehdf5(ric, runparams['simulation']['initial_condition_filename'])

    mainrun = Rsim(runparams)
    mainrun.par.gravity = Gravity(
        externalgravity=True,
        acceleration=et.constant_gravity_acceleration(
        ICparams['gravity_strength'],
            code_units=code_units_obj,
        ),
        code_units=code_units_obj,
    )
    mainrun.RunAll(outputtime=0, mode='hydro')

    output_files = sorted(
        Path(runparams['output']['directory']).glob(
            runparams['output']['filename_prefix'] + '_*.hdf5'
        )
    )
    if not output_files:
        raise FileNotFoundError('hydrostatic run produced no output snapshot')
    final_outfile = str(output_files[-1])
    et.ReadandPlot(
        final_outfile,
        config,
        ls='none',
        marker='o',
        mfc='none',
        markevery=1,
        color='C0',
    )
    figure_filename = os.path.join(runparams['output']['savedir'], 'HydrostaticEquilibrium1D.jpg')
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
