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
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import radhydropy.io as rio
from radhydropy.rsim import Rsim
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('radiative_transfer_sph1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    nested = eu.load_nested_example_config(config_filename)
    runtime = nested['par']
    config = nested
    eu.clean_previous_outputs(runtime['output'])

    Path(runtime['output']['directory']).mkdir(parents=True, exist_ok=True)
    Path(runtime['output']['savedir']).mkdir(parents=True, exist_ok=True)

    et.write_initial_condition(config)

    mainrun = Rsim(runtime)
    mainrun.Callreadhdf5()
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()
    if runtime.get('radiation', {}).get('radiative_transfer_temporal_scheme', 'instantaneous') == 'c2ray':
        mainrun.EvolveStaticThermochemistry(
            runtime['simulation']['final_time'],
            runtime['timestep']['evolution_timestep'],
        )
    rio.write_numbered_hdf5(mainrun, 0)

    output_filename = Path(runtime['output']['directory']) / f"{runtime['output']['filename_prefix']}_000.hdf5"
    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    relative_error = et.save_plot(
        out_mesh,
        out_fluid,
        out_par,
        config,
        str(
            Path(runtime['output']['savedir'])
            / (
                'RadiativeTransferSph1D_C2Ray.jpg'
                if runtime.get('radiation', {}).get('radiative_transfer_temporal_scheme', 'instantaneous') == 'c2ray'
                else 'RadiativeTransferSph1D.jpg'
            )
        ),
    )

    print('max relative error = %.3e' % relative_error)
    figure_name = (
        'RadiativeTransferSph1D_C2Ray.jpg'
        if runtime.get('radiation', {}).get('radiative_transfer_temporal_scheme', 'instantaneous') == 'c2ray'
        else 'RadiativeTransferSph1D.jpg'
    )
    print('figure = %s' % (Path(runtime['output']['savedir']) / figure_name))


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
