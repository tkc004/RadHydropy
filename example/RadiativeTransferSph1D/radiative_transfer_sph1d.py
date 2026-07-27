"""Spherical long-characteristic radiative-transfer example.

A source at the coordinate origin emits ionizing photons at a constant rate.
Hydrodynamics and hydrogen thermo-chemistry are not advanced; the script only
applies the optional long-characteristic radiative-transfer update and compares
the resulting photon number density with the analytic optically thin spherical
dilution solution.

The example builds the static spherical problem from YAML parameters, applies
the long-characteristic radiative-transfer update once through ``Rsim``, writes
an HDF5 snapshot, reloads that snapshot, and compares the result with the
analytic optically thin spherical dilution solution.
"""

import argparse
import os
import sys
from pathlib import Path
import tempfile

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('radiative_transfer_sph1d.yaml')


def load_parameters(config_filename=DEFAULT_CONFIG, rundir=None):
    config_filename = Path(config_filename)
    runparams, ICparams = load_example_parameters(config_filename, rundir)
    return runparams, ICparams


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams = load_parameters(config_filename, rundir)
    config = {**runparams, **ICparams}

    par, mesh, fluid, solver = et.build_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    result = sim.RunRadiativeTransferOnly()

    output_filename = Path(runparams['outdir']) / f"{runparams['outfileprefix']}_000.hdf5"
    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    relative_error = et.save_plot(
        out_mesh,
        out_fluid,
        out_par,
        config['source_photon_rate'],
        str(Path(runparams['savedir']) / 'RadiativeTransferSph1D.jpg'),
    )

    print('outer face photon rate = %s' % result.face_photon_rate[-1])
    print('max relative error = %.3e' % relative_error)
    print('figure = %s' % (Path(runparams['savedir']) / 'RadiativeTransferSph1D.jpg'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the spherical radiative-transfer example.',
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
