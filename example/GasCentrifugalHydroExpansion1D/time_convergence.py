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
from scipy.integrate import solve_ivp

from gas_centrifugal_hydro_expansion1d import (
    CONFIG, run_simulation, spherical_centers,
)
from radhydropy.example_config import load_example_parameters


def total_energy_error(runparams, icparams):
    (sim, saved_mesh, saved, _initial_mass, _initial_energy,
     _initial_radius, _gravity_work, _potential_change,
     _potential_flux) = run_simulation(runparams, icparams)
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
            (0.0, final_time), (position, 0.0),
            rtol=1.0e-10, atol=1.0e-12,
        )
        shell_final[:, index] = solution.y[:, -1]

    saved_radius = spherical_centers(np.asarray(saved_mesh.boundary, dtype=float))[active]
    order = np.argsort(shell_final[0])
    shell_specific_energy = (
        0.5 * shell_final[1]**2
        + 0.5 * shell_j**2 / shell_final[0]**2
        - central_mass / shell_final[0]
    )
    mapped_energy = np.interp(
        saved_radius, shell_final[0, order], shell_specific_energy[order]
    )
    saved_mass = np.asarray(saved.Mass[active], dtype=float)
    saved_total = np.sum(
        np.asarray(saved.Energy[active], dtype=float)
        + np.asarray(saved.GravitationalPotentialEnergy[active], dtype=float)
    )
    ode_total = np.sum(saved_mass * mapped_energy)
    return abs(saved_total - ode_total) / max(abs(ode_total), 1.0e-12)


def main():
    runparams, icparams = load_example_parameters(CONFIG)
    # Keep the mesh fixed so this isolates source time integration rather than
    # mixing temporal and spatial convergence errors.
    runparams['nogrid'] = 128
    dtmax_values = np.asarray((1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4), dtype=float)
    errors = []
    for dtmax in dtmax_values:
        case = dict(runparams)
        case['dtmax'] = float(dtmax)
        error = total_energy_error(case, icparams)
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
