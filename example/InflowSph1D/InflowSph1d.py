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


DEFAULT_CONFIG = Path(__file__).resolve().with_name('InflowSph1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    config = eu.load_nested_example_config(config_filename)

    output = config["par"]['output']
    eu.clean_previous_outputs(config)
    code_units_obj = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])

    config['_code_units'] = code_units_obj
    ric = et.build_initial_condition(config)
    rio.writehdf5(ric, config["par"]['simulation']['initial_condition_filename'])
    mainrun = Rsim(config["par"])
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(0, 9, 2):
        outfilename = os.path.join(
            output['directory'],
            output['filename_prefix'] + '_%03d' % outindex + '.hdf5',
        )
        et.plot_snapshot(
            outfilename,
            config,
            ls='none',
            marker='o',
            mfc='none',
            markevery=1,
            color=next(ax._get_lines.prop_cycler)['color'],
        )
    figure_filename = os.path.join(output['savedir'], config['example']['plot_filename'])
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the spherical inflow example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='YAML file with nested runtime and initial-condition settings.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
