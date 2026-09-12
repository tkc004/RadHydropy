"""Inject a spherical outflow into an initially exact vacuum."""

import argparse
import os
from pathlib import Path
import sys

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
from example.OutflowSphVacuum1D import tools


DEFAULT_CONFIG = Path(__file__).with_name('outflow_sph_vacuum1d.yaml')


def run(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    initial_config = config['initial_condition']
    exampleparams = config['example']
    eu.clean_previous_outputs(config)
    units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    config['_code_units'] = units
    initial = tools.build_initial_condition(config)
    initial.write(
        rundir / config["par"]['simulation']['initial_condition_filename'],
        validate=False,
    )

    sim = Rsim(config["par"])
    sim.RunAll(outputtime=0)

    profiles = []
    first = int(config["par"]['mesh']['ghost_cells'])
    active_count = int(initial_config['grid_cells'])
    output = config["par"]['output']
    output_files = sorted(
        Path(output['directory']).glob(f"{output['filename_prefix']}_*.hdf5")
    )
    for filename in output_files:
        snapshot = rio.loadhdf5(config, filename)
        rho_proper_code = snapshot.fluid.rho_radarray.to(
            units.density_unit
        ).value
        temp_proper_code = snapshot.fluid.temp_radarray.to(
            units.temperature_unit
        ).value
        energy_code = np.asarray(snapshot.fluid.Energy_code, dtype=float)
        if not (np.all(np.isfinite(rho_proper_code)) and np.all(rho_proper_code >= 0.0)):
            raise RuntimeError('vacuum outflow produced invalid density')
        if not (np.all(np.isfinite(energy_code)) and np.all(energy_code >= 0.0)):
            raise RuntimeError('vacuum outflow produced invalid energy')
        boundary_proper_code = snapshot.mesh.boundary_radarray.to(
            units.length_unit
        ).value
        profiles.append((float(snapshot.fluid.time_proper_code), rho_proper_code, temp_proper_code, boundary_proper_code))

    if not profiles:
        raise RuntimeError('vacuum outflow produced no output snapshots')
    filled = [
        np.count_nonzero(
            rho[first:first + active_count] > config["par"]['diagnostics']['cfl_density_floor']
        )
        for _, rho, _, _ in profiles
    ]
    if filled[-1] == 0:
        raise RuntimeError('outflow did not fill any physical vacuum cells')
    figure = Path(output['directory']) / exampleparams['plot_filename']
    analytic_label_used = False
    for time_proper_code, rho_proper_code, _, boundary_proper_code in profiles:
        radius_proper_code = 0.5 * (
            boundary_proper_code[1:] + boundary_proper_code[:-1]
        )
        radius_proper_code = radius_proper_code[first:first + active_count]
        rho_proper_code = rho_proper_code[first:first + active_count]
        positive = rho_proper_code > 0.0
        if np.any(positive):
            line, = plt.loglog(
                radius_proper_code[positive], rho_proper_code[positive],
                label=f't={time_proper_code:.2f} s'
            )
            analytic, front = tools.analytic_density_profile(
                radius_proper_code, time_proper_code, config,
                cell_faces=boundary_proper_code[first:first + active_count + 1],
            )
            label = 'cold analytic profile' if not analytic_label_used else None
            plt.loglog(
                radius_proper_code, analytic, '--', color=line.get_color(), alpha=0.65,
                label=label,
            )
            if not analytic_label_used:
                analytic_label_used = True
            plt.axvline(front, color=line.get_color(), ls=':', alpha=0.35)
    plt.xlabel('radius [cm]')
    plt.ylabel('density [code units]')
    plt.title(
        'Spherical outflow into vacuum\n'
        'solid: numerical Rusanov; dashed: exact cell-average reference\n'
        '(low-density tail is numerical front diffusion)'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()
    print(f'figure = {figure}')
    print(f'outputs = {len(profiles)}')
    print(f'filled physical cells = {filled}')
    return figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)
