"""HM12 PIE relaxation of a hydrostatic atmosphere in a fixed NFW halo."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import radhydropy.io as rio
from radhydropy.gravity import Gravity, nfw_potential
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu

SPEC = importlib.util.spec_from_file_location('pie_nfw_hse_tools', EXAMPLE_DIR / 'tools.py')
et = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(et)

DEFAULT_CONFIG = EXAMPLE_DIR / 'pie_cooling_nfw_hydrostatic_relaxation1d.yaml'


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    initial_mapping = config['initial_condition']
    thermochemistry = par['thermochemistry']
    table_filename = str((config_filename.parent / thermochemistry['metal_pie_table_filename']).resolve())
    thermochemistry['metal_pie_table_filename'] = table_filename
    eu.clean_previous_outputs(par['output'])
    Path(par['output']['directory']).mkdir(parents=True, exist_ok=True)
    code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    halo = et.nfw_halo_parameters(
        initial_mapping['halo_mass'], initial_mapping['concentration'], initial_mapping['redshift'],
        initial_mapping['overdensity'], initial_mapping['h0'],
    )
    temperature = et.virial_temperature(halo, initial_mapping['mu'])
    initial = et.Simwrap(config, code_units=code_units)
    rio.writehdf5(initial, par['simulation']['initial_condition_filename'])
    runtime_only = {
        'box_size', 'coordinate_system', 'current_time', 'grid_cells',
        'number_of_cells', 'inner_radius', 'outer_radius', 'halo_mass',
        'concentration', 'redshift', 'overdensity', 'h0', 'gas_fraction',
        'mean_molecular_weight', 'mu', 'reference_density',
        'initial_temperature', 'final_time', 'evolution_timestep',
        'chemistry_timestep', 'runaway_density_factor',
    }
    sim = Rsim(par)
    sim.Callreadhdf5(); sim.SetMesh(); sim.SetFluid(); sim.SetInitFluid()
    nghost = int(par['mesh']['ghost_cells'])
    interior = slice(nghost, -nghost if nghost else None)
    initial_density_max = float(np.max(np.asarray(sim.fluid.rho_code[interior])))
    floor = thermochemistry['cooling_temperature_floor'].to_value(unyt.K)
    runaway_factor = float(thermochemistry.get('runaway_density_factor', 100.0))

    def stop_on_runaway(runner):
        density = np.asarray(runner.fluid.rho_code[interior])
        temperature_state = np.asarray(runner.fluid.temp_code[interior])
        runaway = np.max(density) >= runaway_factor * initial_density_max
        # Do not terminate because a tenuous outer cell reaches the imposed
        # floor.  The relevant runaway is central loss of pressure support.
        ncentral = max(8, int(0.1 * temperature_state.size))
        floor_reached = np.min(temperature_state[:ncentral]) <= 1.01 * floor
        if runaway or floor_reached:
            reason = 'density runaway' if runaway else 'temperature floor'
            print('stopping relaxation: %s' % reason)
            return True
        return False

    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.coordinate, halo['scale_density'], halo['scale_radius'],
            code_units=sim.par.units.CodeUnits,
        ),
        coordinate=sim.mesh.coordinate.copy(),
        code_units=sim.par.units.CodeUnits,
    )
    sim.Run(mode='hydro_sources', stop_condition=stop_on_runaway)
    all_outputs = sorted(
        Path(par['output']['directory']).glob(f"{par['output']['filename_prefix']}_*.hdf5")
    )
    scheduled_times = [
        float(value) for value in Path(par['output']['time_list_filename']).read_text().splitlines()[1:]
    ]
    outputs = all_outputs[:len(scheduled_times)]
    if len(outputs) < 2:
        raise RuntimeError('expected at least two saved snapshots')
    results = [et.analyze_snapshot(name, config, halo, temperature)
               for name in outputs]
    for result, scheduled_time in zip(results, scheduled_times):
        result['time_Myr'] = scheduled_time
    result_stem = par['simulation']['name']
    report = EXAMPLE_DIR / f'{result_stem}_Report.txt'
    figure = EXAMPLE_DIR / f'{result_stem}.jpg'
    et.write_report(results, report, floor)
    et.plot_results(results, halo, figure)
    print('halo mass = %.6g Msun' % halo['mass'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % halo['virial_radius'].to_value(unyt.kpc))
    print('Tvir = %.6g K' % temperature.to_value(unyt.K))
    print('central T final = %.6g K' % results[-1]['central_temperature_cgs_K'])
    print('central density final = %.6g g/cm^3' % results[-1]['central_density_cgs_g_cm3'])
    print('temperature floor reached = %s' % (results[-1]['minimum_temperature_cgs_K'] <= 1.01 * floor))
    print('figure = %s' % figure)
    print('report = %s' % report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
