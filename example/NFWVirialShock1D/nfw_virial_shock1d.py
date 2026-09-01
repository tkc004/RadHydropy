"""Cold gas shell infall and virial shock in a fixed NFW halo."""

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
from radhydropy.example_config import load_example_parameters
from radhydropy.gravity import Gravity, nfw_potential
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('nfw_virial_shock1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_example_parameters(config_filename)
    runparams['nogrid'] = icparams['nogrid']
    eu.clean_previous_outputs(runparams)
    code_units = CodeUnits.from_mapping(runparams['CodeUnits'])
    halo = et.NFW.nfw_halo_parameters(
        icparams['halo_mass'],
        icparams['concentration'],
        icparams['redshift'],
        icparams['overdensity'],
        icparams['h0'],
    )
    initial_condition = et.Simwrap(icparams, code_units=code_units)
    rio.writehdf5(initial_condition, runparams['ICfilename'])

    sim = Rsim(runparams)
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

    output_files = [
        os.path.join(runparams['outdir'], name)
        for name in sorted(os.listdir(runparams['outdir']))
        if name.startswith(runparams['outfileprefix'] + '_')
        and name.endswith('.hdf5')
    ]
    figure_filename = os.path.join(
        runparams['savedir'],
        'NFWVirialShock1D.jpg',
    )
    et.plot_snapshots(output_files, icparams, runparams, figure_filename)
    rh_rows = et.rankine_hugoniot_diagnostics(
        output_files,
        icparams,
        runparams,
    )
    rh_filename = os.path.join(
        runparams['savedir'],
        'NFWVirialShock1D_RankineHugoniot.txt',
    )
    et.write_rankine_hugoniot_report(rh_rows, rh_filename)
    print('halo mass = %.6g Msun' % halo['mass'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % halo['virial_radius'].to_value(unyt.kpc))
    print('Tvir = %.6g K' % et.NFW.virial_temperature(halo, icparams['mu']).to_value(unyt.K))
    print('snapshots = %d' % len(output_files))
    print('Rankine-Hugoniot checks = %d' % len(rh_rows))
    for row in rh_rows:
        print(
            'RH t=%(time_Myr).0f Myr, r_shock=%(shock_radius_kpc).3g kpc, '
            'Mach=%(mach_number).3g, rho=%(measured_density_ratio).3g/'
            '%(predicted_density_ratio).3g, T=%(measured_temperature_ratio).3g/'
            '%(predicted_temperature_ratio).3g' % row
        )
    print('Rankine-Hugoniot report = %s' % rh_filename)
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the NFW virial-shock example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
