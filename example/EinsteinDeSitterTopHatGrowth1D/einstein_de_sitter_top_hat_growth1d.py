"""Einstein--de Sitter linear-growth test for a spherical top-hat."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name('einstein_de_sitter_top_hat_growth1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)
    runtime = config['par']
    icparams = config['initial_condition']
    example = config.get('example', {})
    eu.clean_previous_outputs(runtime['output'])
    units = CodeUnits.from_mapping(runtime['units']['CodeUnits'])
    cosmology = et.EinsteinDeSitter.from_code_units(
        units, t_ref=float(runtime['gravity']['cosmology_t_ref']),
        a_ref=float(runtime['gravity']['cosmology_a_ref']),
    )
    config['_code_units'] = units
    config['_cosmology'] = cosmology
    initial = et.build_initial_condition(config)
    rio.writehdf5(initial, runtime['simulation']['initial_condition_filename'])

    runtime = {key: (dict(value) if isinstance(value, dict) else value)
               for key, value in runtime.items()}
    runtime['simulation'] = {**runtime['simulation'], 'initial_condition_filename': runtime['simulation']['initial_condition_filename']}
    sim = Rsim(runtime)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    # SetUpFluid initializes its runtime clock to zero; cosmological runs must
    # retain the supercomoving time stored in the IC header.
    sim.fluid.SetFluidTime(sim.par.time)
    sim.SetInitFluid()
    sim.par.set_cosmology_model(cosmology)
    physical = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    initial_mass = float(np.sum(sim.fluid.rho_code[physical] * sim.mesh.vol[physical]))
    top_hat_radius = float(icparams['top_hat_radius'])
    initial_inside = sim.mesh.coordinate[physical] < top_hat_radius
    target_mass = float(np.sum(sim.fluid.rho_code[physical][initial_inside] * sim.mesh.vol[physical][initial_inside]))
    initial_tau = float(np.asarray(sim.fluid.time_code).flat[0])
    initial_a = sim.par.cosmology.scale_factor_from_supercomoving(initial_tau)
    initial_delta = float(icparams['overdensity'])
    history = {'a': [], 'delta': [], 'time': []}

    def record(state):
        tau = float(np.asarray(state.fluid.time_code).flat[0])
        a = state.par.cosmology.scale_factor_from_supercomoving(tau)
        radius = et.enclosed_mass_radius(
            state.mesh.boundary[physical.start:physical.stop + 1],
            state.fluid.rho_code[physical], state.mesh.vol[physical], target_mass,
        )
        cosmic_time = state.par.cosmology.cosmic_time_from_supercomoving(tau)
        rho_background = state.par.cosmology.background_density(cosmic_time) * a**3
        mean_density = 3.0 * target_mass / (4.0 * np.pi * radius**3)
        history['a'].append(float(a))
        history['delta'].append(float(mean_density / rho_background - 1.0))
        history['time'].append(float(cosmic_time))

    record(sim)
    sim.Evolve(
        final_time=sim.par.simulation.final_time,
        mode='hydro',
        history_callback=record,
    )
    final = sim
    final_physical = slice(final.par.mesh.ghost_cells, final.par.mesh.ghost_cells + final.par.mesh.grid_cells)
    final_tau = float(np.asarray(final.fluid.time_code).flat[0])
    final_a = final.par.cosmology.scale_factor_from_supercomoving(final_tau)
    final_cosmic_time = final.par.cosmology.cosmic_time_from_supercomoving(final_tau)
    final_background = final.par.cosmology.background_density(final_cosmic_time) * final_a**3
    final_radius = et.enclosed_mass_radius(
        final.mesh.boundary[final_physical.start:final_physical.stop + 1],
        final.fluid.rho_code[final_physical], final.mesh.vol[final_physical], target_mass,
    )
    measured_delta = 3.0 * target_mass / (4.0 * np.pi * final_radius**3) / final_background - 1.0
    expected_delta = et.linear_overdensity(initial_delta, final_a, initial_a)
    relative_error = abs(measured_delta - expected_delta) / expected_delta
    if not np.isfinite(relative_error) or relative_error > float(example['growth_tolerance']):
        raise RuntimeError('linear growth error %.6g exceeds tolerance' % relative_error)

    figure_filename = Path(runtime['output']['savedir']) / 'EinsteinDeSitterTopHatGrowth1D.jpg'
    a_plot = np.linspace(initial_a, final_a, 100)
    plt.figure(figsize=(6, 4))
    plt.plot(history['a'], history['delta'], 'o', label='simulation')
    plt.plot(a_plot, initial_delta * a_plot / initial_a, '--', label='linear theory')
    plt.xlabel('scale factor $a$')
    plt.ylabel('mean overdensity $\\delta$')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('Einstein-De Sitter top-hat linear growth passed')
    print('a: %.8g -> %.8g' % (initial_a, final_a))
    print('delta: %.8g (measured), %.8g (linear), relative error %.6g' %
          (measured_delta, expected_delta, relative_error))
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    main(parser.parse_args().config)

