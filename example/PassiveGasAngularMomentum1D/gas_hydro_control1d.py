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

from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import radhydropy.io as rio
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('gas_hydro_control1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)

    Path(config["par"]['output']['directory']).mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(config)
    config['_code_units'] = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    writer = et.build_initial_condition(config)
    initial_filename = config["par"]['simulation']['initial_condition_filename']
    writer.write(initial_filename, validate=True)
    initial = rio.loadhdf5(config, initial_filename)
    sim = Rsim(config["par"])
    sim.RunAll(outputtime=0, mode='hydro')
    if hasattr(sim.fluid, 'AngularMomentum_code'):
        raise RuntimeError('control case unexpectedly created AngularMomentum')

    interior = slice(
        config["par"]['mesh']['ghost_cells'],
        config["par"]['mesh']['ghost_cells'] + config["par"]['mesh']['grid_cells'],
    )
    output_files = sorted(
        Path(config["par"]["output"]["directory"]).glob("Output_*.hdf5")
    )
    if not output_files:
        raise FileNotFoundError("no control output snapshot was written")
    final_snapshot = rio.loadhdf5(config, str(output_files[-1]))
    boundary_proper_code = np.asarray(
        final_snapshot.mesh.boundary_radarray.to_value(
            config["_code_units"].length_unit
        ),
        dtype=float,
    )
    active_boundary_proper_code = boundary_proper_code[
        config["par"]["mesh"]["ghost_cells"] :
        config["par"]["mesh"]["ghost_cells"]
        + config["par"]["mesh"]["grid_cells"]
        + 1
    ]
    radius_proper_code = 0.5 * (
        active_boundary_proper_code[1:] + active_boundary_proper_code[:-1]
    )
    figure = Path(config["par"]['output']['directory']) / 'GasHydroControl1D.jpg'
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for axis, initial_values, final_values, ylabel in (
        (axes[0], initial.fluid.rho_radarray.value, final_snapshot.fluid.rho_radarray.value[interior], 'density [code units]'),
        (axes[1], initial.fluid.vel_radarray.value, final_snapshot.fluid.vel_radarray.value[interior], 'radial velocity [code units]'),
        (axes[2], initial.fluid.temp_radarray.value, final_snapshot.fluid.temp_radarray.value[interior], 'temperature [code units]'),
    ):
        axis.plot(radius_proper_code, initial_values, '--', label='initial')
        axis.plot(radius_proper_code, final_values, 'o', ms=3, label='final')
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
