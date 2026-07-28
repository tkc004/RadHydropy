import argparse
import os
import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radhydropy.example_config import load_example_parameters
from radhydropy.gravity import point_mass_potential
from radhydropy.rsim import Rsim

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import radhydropy.io as rio
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'hydrostatic_equilibrium_spherical_point_mass1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams = load_example_parameters(config_filename, rundir)

    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric, runparams['ICfilename'])

    mainrun = Rsim(runparams)
    mainrun.Callreadhdf5()
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()
    mainrun.par.externalgravity = True
    mainrun.par.gravity_coordinate = mainrun.mesh.coordinate.copy()
    mainrun.par.gravity_potential = point_mass_potential(
        mainrun.par.gravity_coordinate,
        ICparams['point_mass'],
    )
    mainrun.Run(mode='hydro')

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
    figure_filename = os.path.join(
        runparams['savedir'],
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
        help='YAML file with runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
