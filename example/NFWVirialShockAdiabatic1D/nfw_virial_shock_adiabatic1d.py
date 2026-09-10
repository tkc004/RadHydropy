"""Adiabatic accretion shock benchmark for a 1e12 Msun NFW halo."""

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
    'nfw_virial_shock_adiabatic1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']; initial_condition = config['initial_condition']
    eu.clean_previous_outputs(config)
    code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    halo = et.nfw_halo_parameters(
        initial_condition['halo_mass'],
        initial_condition['concentration'],
        initial_condition['redshift'],
        initial_condition['overdensity'],
        initial_condition['h0'],
    )

    config['_code_units'] = code_units
    initial_state = et.build_initial_condition(config)
    rio.writehdf5(initial_state, par['simulation']['initial_condition_filename'])

    sim = Rsim(config["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
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

    output_files = [
        os.path.join(par['output']['directory'], name)
        for name in sorted(os.listdir(par['output']['directory']))
        if name.startswith(par['output']['filename_prefix'] + '_')
        and name.endswith('.hdf5')
    ]
    figure_filename = os.path.join(
        par['output']['directory'],
        'NFWVirialShockAdiabatic1D.jpg',
    )
    rows = et.rankine_hugoniot_diagnostics(
        output_files,
        config,
        halo,
    )
    report_filename = os.path.join(
        par['output']['directory'],
        'NFWVirialShockAdiabatic1D_RankineHugoniot.txt',
    )
    et.plot_snapshots(output_files, config, halo, figure_filename)
    et.write_rankine_hugoniot_report(rows, report_filename)

    virial_radius = halo['radius_virial_proper_kpc_unyt'].to_value(unyt.kpc)
    virial_velocity = halo['vel_virial_proper_km_s_unyt'].to_value(unyt.km / unyt.s)
    virial_temperature = et.virial_temperature(
        halo,
        initial_condition['mu'],
    ).to_value(unyt.K)
    print('halo mass = %.6g Msun' % halo['mass_halo_proper_g_unyt'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % virial_radius)
    print('4 R200 = %.6g kpc' % (4.0 * virial_radius))
    print('V200 = %.6g km/s' % virial_velocity)
    print('Tvir = %.6g K' % virial_temperature)
    print('snapshots = %d' % len(output_files))
    print('Rankine-Hugoniot checks = %d' % len(rows))
    for row in rows:
        print(
            'RH t=%(time_proper_Myr).0f Myr, r_shock=%(shock_radius_proper_kpc).3g kpc, '
            'Mach=%(mach_number_dimensionless).3g, rho=%(density_ratio_measured_dimensionless).3g/'
            '%(density_ratio_predicted_dimensionless).3g, T=%(temperature_ratio_measured_dimensionless).3g/'
            '%(temperature_ratio_predicted_dimensionless).3g' % row
        )
    print('Rankine-Hugoniot report = %s' % report_filename)
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the adiabatic NFW virial-shock benchmark.'
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
