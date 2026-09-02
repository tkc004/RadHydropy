"""Timestep convergence study for the coupled centrifugal source update."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gas_centrifugal_hydro_expansion1d import (
    CONFIG, run_simulation, spherical_centers,
)
import example_utils as eu
from shell_remap import centrifugal_shell_reference


def total_energy_error(runparams, icparams, runtime):
    (sim, saved_mesh, saved, _initial_mass, _initial_energy,
     _initial_radius, _gravity_work, _potential_change,
     _potential_flux) = run_simulation(runparams, icparams, runtime)
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    saved_boundary = np.asarray(saved_mesh.boundary, dtype=float)
    source_boundary = saved_boundary[
        sim.par.noghost:sim.par.noghost + sim.par.nogrid + 1
    ]
    saved_radius = spherical_centers(saved_boundary)[active]
    central_mass = float(icparams['central_mass'])
    rotation_factor = float(icparams['rotation_factor'])
    final_time = float(sim.fluid.time)
    reference = centrifugal_shell_reference(
        source_boundary,
        source_boundary,
        final_time,
        float(icparams['density']),
        central_mass,
        rotation_factor,
        samples_per_cell=int(icparams.get('reference_samples_per_cell', 32)),
    )
    saved_mass = np.asarray(saved.Mass_code[active], dtype=float)
    saved_total = np.sum(
        np.asarray(saved.Energy_code[active], dtype=float)
        + np.asarray(saved.GravitationalPotentialEnergy_code[active], dtype=float)
    )
    ode_total = np.sum(reference['energy'])
    return abs(saved_total - ode_total) / max(abs(ode_total), 1.0e-12)


def main():
    config = eu.load_nested_example_config(CONFIG)
    runparams = eu.legacy_example_parameters(config)
    icparams = {**config['initial_condition'], 'nogrid': 128}
    # Keep the mesh fixed so this isolates source time integration rather than
    # mixing temporal and spatial convergence errors.
    runparams['nogrid'] = 128
    dtmax_values = np.asarray((1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4), dtype=float)
    errors = []
    for dtmax in dtmax_values:
        case = dict(runparams)
        case['dtmax'] = float(dtmax)
        runtime = {**config['par'], 'mesh': {**config['par']['mesh'], 'grid_cells': 128, 'ghost_cells': 2}, 'timestep': {**config['par']['timestep'], 'dtmax': float(dtmax)}}
        error = total_energy_error(case, icparams, runtime)
        errors.append(error)
        print('dtmax %.6g: total-energy error %.8g' % (dtmax, error))

    errors = np.asarray(errors)
    output = ROOT / 'outputs' / 'GasCentrifugalHydroExpansion1D_time_convergence.jpg'
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.loglog(dtmax_values, errors, 'o-')
    axis.set_xlabel('maximum timestep')
    axis.set_ylabel('relative total-energy error')
    axis.grid(alpha=0.25, which='both')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print('time-convergence figure = %s' % output)


if __name__ == '__main__':
    main()
