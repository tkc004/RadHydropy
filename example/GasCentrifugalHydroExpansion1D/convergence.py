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
from scipy.integrate import solve_ivp

from gas_centrifugal_hydro_expansion1d import (
    CONFIG, run_simulation, spherical_centers,
)
from radhydropy.example_config import load_example_parameters


def measure(runparams, icparams):
    (sim, saved_mesh, saved, initial_mass, initial_energy, initial_radius,
     cumulative_gravity_work, cumulative_potential_change,
     cumulative_potential_flux) = (
        run_simulation(runparams, icparams)
    )
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    radius = np.asarray(sim.mesh.coordinate[active], dtype=float)
    central_mass = float(icparams['central_mass'])
    rotation_factor = float(icparams['rotation_factor'])
    shell_j = rotation_factor * np.sqrt(central_mass * radius)
    final_time = float(sim.fluid.time)

    def rhs(_time, state, specific_j):
        position, velocity = state
        position = max(position, np.finfo(float).tiny)
        return velocity, -central_mass / position**2 + specific_j**2 / position**3

    shell_final = np.empty((2, len(radius)))
    for index, (position, specific_j) in enumerate(zip(radius, shell_j)):
        solution = solve_ivp(
            lambda time, state: rhs(time, state, specific_j),
            (0.0, final_time), (position, 0.0), rtol=1.0e-10, atol=1.0e-12,
        )
        shell_final[:, index] = solution.y[:, -1]
    saved_radius = spherical_centers(np.asarray(saved_mesh.boundary, dtype=float))[active]
    order = np.argsort(shell_final[0])
    ode_velocity = np.interp(saved_radius, shell_final[0, order], shell_final[1, order])
    ode_j = np.interp(saved_radius, shell_final[0, order], shell_j[order])
    saved_velocity = np.asarray(saved.vel[active], dtype=float)
    saved_j = np.asarray(saved.specific_angular_momentum[active], dtype=float)
    saved_mass = np.asarray(saved.Mass[active], dtype=float)
    saved_energy = np.asarray(saved.Energy[active], dtype=float)
    velocity_error = np.max(np.abs(saved_velocity - ode_velocity))
    j_error = np.max(np.abs(saved_j - ode_j))
    simulated_energy = np.sum(saved_energy - central_mass * saved_mass / saved_radius)
    shell_specific_energy = (
        0.5 * shell_final[1]**2
        + 0.5 * shell_j**2 / shell_final[0]**2
        - central_mass / shell_final[0]
    )
    mapped_specific_energy = np.interp(
        saved_radius, shell_final[0, order], shell_specific_energy[order]
    )
    shell_energy = np.sum(saved_mass * mapped_specific_energy)
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
