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

from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import unyt

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


DEFAULT_CONFIG = Path(__file__).resolve().with_name('sodshock1d.yaml')


def main(config_filename=DEFAULT_CONFIG, riemann_solver=None):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    config = eu.load_nested_example_config(config_filename)
    runparams, ICparams = config['par'], config['initial_condition']
    exampleparams = config['example']
    if riemann_solver is not None:
        runparams['hydrodynamics']['riemann_solver'] = riemann_solver
    output = runparams['output']
    eu.clean_previous_outputs(output)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])

    config['_code_units'] = code_units_obj
    ric = et.build_initial_condition(config)
    rio.writehdf5(ric, runparams['simulation']['initial_condition_filename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll()
    outindex = exampleparams['output_index']
    outfilename = os.path.join(
        output['directory'],
        output['filename_prefix'] + '_%03d' % outindex + '.hdf5',
    )
    et.ReadandPlot(
        outfilename,
            config,
        ls='none',
        marker='o',
        mfc='none',
        markevery=5,
    )
    figure_filename = os.path.join(output['savedir'], exampleparams['plot_filename'])
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the Sod shock example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='YAML file with runparams and ICparams.')
    parser.add_argument('--riemann-solver', choices=('Rusanov', 'HLLC'))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, riemann_solver=args.riemann_solver)



