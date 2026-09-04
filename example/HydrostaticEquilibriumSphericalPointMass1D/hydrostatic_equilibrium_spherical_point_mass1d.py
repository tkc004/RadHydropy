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

from radhydropy.gravity import Gravity, point_mass_potential
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
    'hydrostatic_equilibrium_spherical_point_mass1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    initial_condition = config['initial_condition']
    eu.clean_previous_outputs(par['output'])
    code_units_obj = CodeUnits.from_mapping(par['units']['CodeUnits'])

    config['_code_units'] = code_units_obj
    ric = et.build_initial_condition(config)
    initial_filename = Path(par['simulation']['initial_condition_filename'])
    rio.writehdf5(ric, initial_filename)

    runtime = {**par, 'simulation': {**par['simulation'],
        'initial_condition_filename': str(initial_filename)}}
    mainrun = Rsim(runtime)
    mainrun.Callreadhdf5()
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()
    mainrun.par.gravity = Gravity(
        externalgravity=True,
        potential=point_mass_potential(
            mainrun.mesh.coordinate,
            initial_condition['point_mass'],
            code_units=code_units_obj,
        ),
        coordinate=mainrun.mesh.coordinate.copy(),
        code_units=code_units_obj,
    )
    mainrun.Run(mode='hydro')

    final_outfile = os.path.join(
        par['output']['directory'],
        par['output']['filename_prefix'] + '_001.hdf5',
    )
    if not os.path.exists(final_outfile):
        raise FileNotFoundError(
            'Expected an evolved snapshot at %s, but it was not written.'
            % final_outfile
        )
    et.ReadandPlot(
        final_outfile,
        config,
        ls='none',
        marker='o',
        mfc='none',
        markevery=1,
        color='C0',
    )
    figure_filename = os.path.join(
        par['output']['savedir'],
        'HydrostaticEquilibriumSphericalPointMass1D.jpg',
    )
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the spherical hydrostatic-equilibrium point-mass example.'
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file with nested solver and initial-condition mappings.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)

