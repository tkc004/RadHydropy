"""Resolution convergence study for the hydro expansion benchmark."""

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
from radhydropy.example_config import load_example_parameters
from shell_remap import centrifugal_shell_reference


def measure(runparams, icparams):
    (sim, saved_mesh, saved, initial_mass, initial_energy, initial_radius,
     cumulative_gravity_work, cumulative_potential_change,
     cumulative_potential_flux) = (
        run_simulation(runparams, icparams)
    )
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    source_boundary = np.asarray(
        saved_mesh.boundary[sim.par.noghost:sim.par.noghost + sim.par.nogrid + 1],
        dtype=float,
    )
    saved_radius = spherical_centers(np.asarray(saved_mesh.boundary, dtype=float))[active]
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
    ode_velocity = reference['velocity']
    ode_j = reference['specific_angular_momentum']
    saved_velocity = np.asarray(saved.vel[active], dtype=float)
    saved_j = np.asarray(saved.specific_angular_momentum[active], dtype=float)
    saved_mass = np.asarray(saved.Mass[active], dtype=float)
    saved_energy = np.asarray(saved.Energy[active], dtype=float)
    velocity_error = np.max(np.abs(saved_velocity - ode_velocity))
    j_error = np.max(np.abs(saved_j - ode_j))
    simulated_energy = np.sum(saved_energy - central_mass * saved_mass / saved_radius)
    shell_energy = np.sum(reference['energy'])
    energy_error = abs(simulated_energy - shell_energy) / max(abs(shell_energy), 1.0e-12)
    mass_error = abs(np.sum(saved_mass) - np.sum(initial_mass)) / max(
        abs(np.sum(initial_mass)), 1.0e-300
    )
    potential_initial = -central_mass * np.sum(initial_mass / initial_radius)
    potential_final = -central_mass * np.sum(saved_mass / saved_radius)
    potential_residual = abs(cumulative_gravity_work + potential_final - potential_initial)
    return (
        float(velocity_error), float(j_error), float(energy_error),
        float(mass_error), float(potential_residual),
    )


def main():
    runparams, icparams = load_example_parameters(CONFIG)
    resolutions = (32, 64, 128)
    results = []
    for resolution in resolutions:
        case = dict(runparams)
        case['nogrid'] = resolution
        results.append(measure(case, icparams))
        print('resolution %d: velocity=%g J/M=%g energy=%g mass=%g potential=%g' % (
            resolution, *results[-1]
        ))
    results = np.asarray(results)
    output = ROOT / 'outputs' / 'GasCentrifugalHydroExpansion1D_convergence.jpg'
    fig, axis = plt.subplots(figsize=(6, 4))
    for index, label in enumerate(('velocity', 'J/M', 'energy', 'mass', 'potential')):
        axis.loglog(resolutions, results[:, index], 'o-', label=label)
    axis.set_xlabel('number of cells')
    axis.set_ylabel('absolute / relative error')
    axis.grid(alpha=0.25, which='both')
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print('convergence figure = %s' % output)


if __name__ == '__main__':
    main()
