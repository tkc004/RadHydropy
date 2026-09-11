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
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'nfw_hydrostatic_equilibrium1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    initial_condition = config['initial_condition']
    eu.clean_previous_outputs(config)
    code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    halo = et.nfw_halo_parameters(
        initial_condition['halo_mass'],
        initial_condition['concentration'],
        initial_condition['redshift'],
        initial_condition['overdensity'],
        initial_condition['h0'],
    )
    temperature_proper_unyt = et.virial_temperature(halo, initial_condition['mu'])

    config['_code_units'] = code_units
    initial_state = et.build_initial_condition(config)
    initial_state.write(par['simulation']['initial_condition_filename'], validate=True)

    sim = rio.loadhdf5(config, par['simulation']['initial_condition_filename'])
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.geometry_state.x_proper_code,
            halo['rho_scale_cgs_g_cm3_unyt'],
            halo['radius_scale_proper_kpc_unyt'],
            code_units=code_units,
        ),
        coordinate=sim.mesh.geometry_state.x_proper_code.copy(),
        code_units=code_units,
    )
    sim.Run(mode='hydro')

    final_outfile = os.path.join(
        par['output']['directory'], par['output']['filename_prefix'] + '_001.hdf5',
    )
    if not os.path.exists(final_outfile):
        raise FileNotFoundError(f'Expected evolved snapshot at {final_outfile}')
    figure_filename = os.path.join(
        par['output']['directory'],
        'NFWHydrostaticEquilibrium1D.jpg',
    )
    max_relative_error = et.read_and_plot(
        final_outfile,
        config,
        halo,
        temperature_proper_unyt,
        figure_filename,
    )
    print('halo mass = %.6g Msun' % halo['mass_halo_proper_g_unyt'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % halo['radius_virial_proper_kpc_unyt'].to_value(unyt.kpc))
    print('r_s = %.6g kpc' % halo['radius_scale_proper_kpc_unyt'].to_value(unyt.kpc))
    print('V200 = %.6g km/s' % halo['vel_virial_proper_km_s_unyt'].to_value(unyt.km / unyt.s))
    print('Tvir = %.6g K' % temperature_proper_unyt.to_value(unyt.K))
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
