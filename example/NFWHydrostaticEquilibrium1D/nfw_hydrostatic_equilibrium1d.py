"""Hydrostatic gas in a 1e8 Msun NFW dark-matter halo."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import unyt

import radhydropy.io as rio
from radhydropy.gravity import Gravity, nfw_potential
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'nfw_hydrostatic_equilibrium1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    icparams = config['initial_condition']
    eu.clean_previous_outputs(par['output'])
    code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    halo = et.nfw_halo_parameters(
        icparams['halo_mass'],
        icparams['concentration'],
        icparams['redshift'],
        icparams['overdensity'],
        icparams['h0'],
    )
    temperature = et.virial_temperature(halo, icparams['mu'])

    config['_code_units'] = code_units
    initial_condition = et.build_initial_condition(config)
    rio.writehdf5(initial_condition, par['simulation']['initial_condition_filename'])

    sim = Rsim(par)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.coordinate,
            halo['scale_density'],
            halo['scale_radius'],
            code_units=code_units,
        ),
        coordinate=sim.mesh.coordinate.copy(),
        code_units=code_units,
    )
    sim.Run(mode='hydro')

    final_outfile = os.path.join(
        par['output']['directory'], par['output']['filename_prefix'] + '_001.hdf5',
    )
    if not os.path.exists(final_outfile):
        raise FileNotFoundError(f'Expected evolved snapshot at {final_outfile}')
    figure_filename = os.path.join(
        par['output']['savedir'],
        'NFWHydrostaticEquilibrium1D.jpg',
    )
    max_relative_error = et.read_and_plot(
        final_outfile,
        config,
        halo,
        temperature,
        figure_filename,
    )
    print('halo mass = %.6g Msun' % halo['mass'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % halo['virial_radius'].to_value(unyt.kpc))
    print('r_s = %.6g kpc' % halo['scale_radius'].to_value(unyt.kpc))
    print('V200 = %.6g km/s' % halo['virial_velocity'].to_value(unyt.km / unyt.s))
    print('Tvir = %.6g K' % temperature.to_value(unyt.K))
    print('maximum density relative error = %.6g' % max_relative_error)
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the NFW hydrostatic-equilibrium example.',
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)



