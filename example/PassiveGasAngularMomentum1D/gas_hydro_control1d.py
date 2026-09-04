"""Hydro control case without gas angular-momentum storage or advection."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', 'radhydropy-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('gas_hydro_control1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runparams = config['par']
    Path(runparams['output']['directory']).mkdir(parents=True, exist_ok=True)
    Path(runparams['output']['savedir']).mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(runparams)
    config['_code_units'] = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    initial = et.build_initial_condition(config)
    rio.writehdf5(initial, runparams['simulation']['initial_condition_filename'])

    sim = Rsim(runparams)
    sim.RunAll(outputtime=0, mode='hydro')
    if hasattr(sim.fluid, 'AngularMomentum_code'):
        raise RuntimeError('control case unexpectedly created AngularMomentum')

    interior = slice(
        runparams['mesh']['ghost_cells'],
        runparams['mesh']['ghost_cells'] + runparams['mesh']['grid_cells'],
    )
    radius = np.asarray(sim.mesh.coordinate[interior], dtype=float)
    figure = Path(runparams['output']['savedir']) / 'GasHydroControl1D.jpg'
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for axis, initial_values, final_values, ylabel in (
        (axes[0], initial.fluid.rho_code, sim.fluid.rho_code[interior], 'density [code units]'),
        (axes[1], initial.fluid.vel_code, sim.fluid.vel_code[interior], 'radial velocity [code units]'),
        (axes[2], initial.fluid.temp_code, sim.fluid.temp_code[interior], 'temperature [code units]'),
    ):
        axis.plot(radius, initial_values, '--', label='initial')
        axis.plot(radius, final_values, 'o', ms=3, label='final')
        axis.set_xlabel('cell coordinate [code length]')
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle('Hydro control without angular-momentum advection')
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print('hydro control without angular-momentum storage passed')
    print('figure = %s' % figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
