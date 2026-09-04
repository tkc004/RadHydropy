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


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'ballistic_infall_spherical_point_mass1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    nested = eu.load_nested_example_config(config_filename)
    runtime = nested['par']
    ICparams = nested['initial_condition']
    eu.clean_previous_outputs(runtime['output'])
    code_units_obj = CodeUnits.from_mapping(runtime['units']['CodeUnits'])

    ric = et.Simwrap(ICparams, code_units=code_units_obj)
    rio.writehdf5(ric, runtime['simulation']['initial_condition_filename'])

    mainrun = Rsim(runtime)
    mainrun.Callreadhdf5()
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()
    mainrun.par.gravity = Gravity(
        externalgravity=True,
        acceleration=et.point_mass_acceleration(
            ICparams['point_mass'],
            code_units=code_units_obj,
        ),
        code_units=code_units_obj,
    )
    mainrun.Run(mode='hydro')

    final_outfile = os.path.join(
        runtime['output']['directory'],
        runtime['output']['filename_prefix'] + '_001.hdf5',
    )
    et.ReadandPlot(
        final_outfile,
        ICparams,
        runtime,
        ls='none',
        marker='o',
        mfc='none',
        markevery=1,
        color='C0',
    )
    figure_filename = os.path.join(
        runtime['output']['savedir'],
        'BallisticInfallSphericalPointMass1D.jpg',
    )
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a spherical ballistic-infall gravity check example.'
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
