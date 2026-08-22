"""Virial-shock cooling experiment using the z=0 HM12 PIE background."""

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
ADIABATIC_TOOLS = EXAMPLE_ROOT / 'NFWVirialShockAdiabatic1D' / 'tools.py'
for path in (PROJECT_ROOT, EXAMPLE_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
import example_utils as eu

spec = importlib.util.spec_from_file_location('adiabatic_nfw_tools', ADIABATIC_TOOLS)
et = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(et)
pie_spec = importlib.util.spec_from_file_location(
    'nfw_virial_shock_pie_tools', Path(__file__).resolve().with_name('tools.py')
)
diag = importlib.util.module_from_spec(pie_spec)
assert pie_spec.loader is not None
pie_spec.loader.exec_module(diag)

DEFAULT_CONFIG = Path(__file__).resolve().with_name('nfw_virial_shock_pie1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    runparams, icparams = load_example_parameters(config_filename)
    runparams['metal_pie_table_filename'] = str(
        (config_filename.parent / runparams['metal_pie_table_filename']).resolve()
    )
    eu.clean_previous_outputs(runparams)
    os.makedirs(runparams['outdir'], exist_ok=True)
    code_units = et.CodeUnits.from_mapping(runparams['CodeUnits'])
    pie_table = diag.MetalPIETable(runparams['metal_pie_table_filename'])
    halo = et.nfw_halo_parameters(
        icparams['halo_mass'], icparams['concentration'], icparams['redshift'],
        icparams['overdensity'], icparams['h0'],
    )
    initial_condition = diag.Simwrap(
        icparams,
        code_units=code_units,
        pie_table=pie_table,
        pie_redshift=runparams['metal_pie_redshift'],
        metallicity=runparams['metallicity'],
        hydrogen_mass_fraction=runparams['hydrogen_mass_fraction'],
    )
    # OutflowSph copies these values into the inner ghost cells.  Match them
    # to the innermost physical PIE-equilibrium cell for every halo variant;
    # the generic defaults (1 g cm^-3, 0 K) create a spurious central spike.
    inner = initial_condition.fluid
    runparams['vel_outflow'] = inner.vel[0].to('km/s')
    runparams['rho_outflow'] = inner.rho[0].to('g/cm**3')
    runparams['temp_outflow'] = inner.temp[0].to('K')
    runparams['mu_outflow'] = float(icparams['mu'])
    rio.writehdf5(initial_condition, runparams['ICfilename'])
    print(
        'initial HM12 PIE temperature = %.6g--%.6g K' % (
            initial_condition.fluid.temp.min().to_value(unyt.K),
            initial_condition.fluid.temp.max().to_value(unyt.K),
        )
    )

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.coordinate, halo['scale_density'], halo['scale_radius'],
            code_units=sim.par.CodeUnits,
        ),
        coordinate=sim.mesh.coordinate.copy(),
        code_units=sim.par.CodeUnits,
    )
    # PIE heating/cooling is a source update coupled to the hydro step.
    sim.Run(mode='hydro_sources')

    output_files = [
        os.path.join(runparams['outdir'], name)
        for name in sorted(os.listdir(runparams['outdir']))
        if name.startswith(runparams['outfileprefix'] + '_')
        and name.endswith('.hdf5')
    ]
    output_tag = runparams.get('output_tag', 'NFWVirialShockPIE1D')
    figure = os.path.join(runparams['savedir'], output_tag + '.jpg')
    report = os.path.join(runparams['savedir'], output_tag + '_Stability.txt')
    diag.plot_snapshots(output_files, icparams, runparams, halo, figure)
    rows = diag.pie_stability_diagnostics(output_files, icparams, runparams, halo)
    diag.write_stability_report(rows, report)
    print('halo mass = %.6g Msun' % halo['mass'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % halo['virial_radius'].to_value(unyt.kpc))
    print('Tvir = %.6g K' % et.virial_temperature(halo, icparams['mu']).to_value(unyt.K))
    print('snapshots = %d' % len(output_files))
    print('stability diagnostics = %d' % len(rows))
    print('stability report = %s' % report)
    print('figure = %s' % figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the HM12 PIE virial-shock experiment.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
