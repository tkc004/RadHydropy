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
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools


DEFAULT_CONFIG = Path(__file__).with_name('outflow_sph_vacuum1d.yaml')


def run(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    runparams, icparams = load_example_parameters(config_filename, rundir)
    eu.clean_previous_outputs(runparams)
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    initial = tools.Simwrap(icparams, units)
    rio.writehdf5(initial, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.RunAll(outputtime=0)

    profiles = []
    first = int(runparams['noghost'])
    active_count = int(icparams['nogrid'])
    output_files = sorted(Path(runparams['outdir']).glob('Output_*.hdf5'))
    for filename in output_files:
        par, mesh, fluid = tools.Par(), tools.Mesh(), tools.Fluid()
        par.CodeUnits = units
        rio.readhdf5(par, mesh, fluid, filename)
        rho = np.asarray(fluid.rho, dtype=float)
        temp = np.asarray(getattr(fluid, 'temp', np.zeros_like(rho)), dtype=float)
        energy = np.asarray(
            getattr(fluid, 'Energy', np.zeros_like(rho)), dtype=float
        )
        if not (np.all(np.isfinite(rho)) and np.all(rho >= 0.0)):
            raise RuntimeError('vacuum outflow produced invalid density')
        if not (np.all(np.isfinite(energy)) and np.all(energy >= 0.0)):
            raise RuntimeError('vacuum outflow produced invalid energy')
        profiles.append((float(fluid.time), rho, temp, np.asarray(mesh.boundary)))

    if not profiles:
        raise RuntimeError('vacuum outflow produced no output snapshots')
    filled = [
        np.count_nonzero(rho[first:first + active_count] > runparams['cfl_density_floor'])
        for _, rho, _, _ in profiles
    ]
    if filled[-1] == 0:
        raise RuntimeError('outflow did not fill any physical vacuum cells')
    figure = Path(runparams['savedir']) / 'OutflowSphVacuum1D.jpg'
    analytic_label_used = False
    for time, rho, _, boundary in profiles:
        radius = 0.5 * (boundary[1:] + boundary[:-1])
        radius = radius[first:first + active_count]
        rho = rho[first:first + active_count]
        positive = rho > 0.0
        if np.any(positive):
            line, = plt.loglog(
                radius[positive], rho[positive], label=f't={time:.2f} s'
            )
            analytic, front = tools.analytic_density_profile(
                radius, time, icparams, runparams,
                cell_faces=boundary[first:first + active_count + 1],
            )
            label = 'cold analytic profile' if not analytic_label_used else None
            plt.loglog(
                radius, analytic, '--', color=line.get_color(), alpha=0.65,
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
