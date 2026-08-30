"""Hydro plus centrifugal-source expansion benchmark."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/tmp/radhydropy-matplotlib')
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from radhydropy.example_config import load_example_parameters
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits


CONFIG = ROOT / 'gas_centrifugal_hydro_expansion1d.yaml'


def spherical_centers(boundary):
    return 0.75 * (
        boundary[1:]**4 - boundary[:-1]**4
    ) / (boundary[1:]**3 - boundary[:-1]**3)


class InitialCondition:
    def __init__(self, count, radius_min, radius_max, density, temperature,
                 central_mass, rotation_factor, code_units):
        boundary = np.linspace(radius_min, radius_max, count + 1)
        radius = spherical_centers(boundary)
        self.par = SimpleNamespace(
            CodeUnits=code_units, nogrid=count, noghost=2,
            coordsys='spherical', time=0.0, boxsize=np.asarray([radius_max]),
        )
        self.mesh = SimpleNamespace(
            boundary=boundary, coordinate=radius,
        )
        self.fluid = SimpleNamespace(
            rho=np.full(count, density), vel=np.zeros(count),
            temp=np.full(count, temperature), mu=np.ones(count),
            specific_angular_momentum=rotation_factor * np.sqrt(
                central_mass * radius
            ),
        )


class FixedCentralGravity:
    cosmological = False
    dark_matter = None

    def __init__(self, central_mass):
        self.central_mass = central_mass

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        radius = np.abs(np.asarray(mesh.coordinate, dtype=float))
        acceleration = np.zeros_like(radius)
        valid = radius > 0.0
        acceleration[valid] = -self.central_mass / radius[valid]**2
        return acceleration

    def potential_on(self, coordinate):
        radius = np.abs(np.asarray(coordinate, dtype=float))
        potential = np.zeros_like(radius)
        valid = radius > 0.0
        potential[valid] = -self.central_mass / radius[valid]
        return potential

    def potential_on_mesh(self, mesh):
        return self.potential_on(mesh.coordinate)


def run_simulation(runparams, icparams):
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    count = int(runparams['nogrid'])
    initial = InitialCondition(
        count, float(icparams['radius_min']), float(icparams['radius_max']),
        float(icparams['density']), float(runparams['temperature']),
        float(icparams['central_mass']), float(icparams['rotation_factor']),
        units,
    )
    filename = ROOT / runparams['ICfilename']
    filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, filename)
    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.par.gravity = FixedCentralGravity(float(icparams['central_mass']))
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    initial_mass = np.asarray(sim.fluid.Mass[active], dtype=float).copy()
    initial_energy = np.asarray(sim.fluid.Energy[active], dtype=float).copy()
    initial_radius = np.asarray(sim.mesh.coordinate[active], dtype=float).copy()
    sim.Run(outputtime=0, mode='hydro')
    final_filename = ROOT / runparams['outdir'] / 'Output_final.hdf5'
    sim.fluid.SetTemperature()
    rio.writehdf5(sim, final_filename)
    final_par = SimpleNamespace(coordsys='spherical', CodeUnits=units)
    final_mesh = SimpleNamespace()
    final_fluid = SimpleNamespace()
    rio.readhdf5(final_par, final_mesh, final_fluid, final_filename)
    return (
        sim, final_mesh, final_fluid, initial_mass, initial_energy,
        initial_radius, float(sim.cumulative_gravity_work),
        float(sim.cumulative_gravity_potential_change),
        float(sim.cumulative_gravity_potential_flux),
    )


def main(config_filename=CONFIG):
    runparams, icparams = load_example_parameters(config_filename)
    (sim, saved_mesh, saved, initial_mass, initial_energy,
     initial_radius, cumulative_gravity_work, cumulative_potential_change,
     cumulative_potential_flux) = run_simulation(runparams, icparams)
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    radius = np.asarray(sim.mesh.coordinate[active], dtype=float)
    central_mass = float(icparams['central_mass'])
    rotation_factor = float(icparams['rotation_factor'])
    shell_j = rotation_factor * np.sqrt(central_mass * radius)
    final_time = float(sim.fluid.time)

    def rhs(_time, state, specific_j):
        position, velocity = state
        safe_position = max(position, np.finfo(float).tiny)
        return velocity, -central_mass / safe_position**2 + specific_j**2 / safe_position**3

    shell_final = np.empty((2, len(radius)))
    for index, (shell_radius, specific_j) in enumerate(zip(radius, shell_j)):
        solution = solve_ivp(
            lambda time, state: rhs(time, state, specific_j),
            (0.0, final_time), (shell_radius, 0.0),
            rtol=1.0e-10, atol=1.0e-12,
        )
        shell_final[:, index] = solution.y[:, -1]
    order = np.argsort(shell_final[0])
    saved_radius = spherical_centers(np.asarray(saved_mesh.boundary, dtype=float))[active]
    ode_velocity = np.interp(saved_radius, shell_final[0, order], shell_final[1, order])
    ode_j = np.interp(saved_radius, shell_final[0, order], shell_j[order])
    saved_velocity = np.asarray(saved.vel[active], dtype=float)
    saved_j = np.asarray(saved.specific_angular_momentum[active], dtype=float)
    saved_mass = np.asarray(saved.Mass[active], dtype=float)
    saved_energy = np.asarray(saved.Energy[active], dtype=float)
    velocity_error = float(np.max(np.abs(saved_velocity - ode_velocity)))
    j_error = float(np.max(np.abs(saved_j - ode_j)))
    mass_error = float(
        abs(np.sum(saved_mass) - np.sum(initial_mass))
        / max(abs(np.sum(initial_mass)), 1.0e-300)
    )
    if velocity_error > 0.08:
        raise RuntimeError('hydro expansion velocity disagrees with shell ODE: %.6g' % velocity_error)
    if not np.all(np.isfinite(saved_j)):
        raise RuntimeError('hydro expansion produced invalid specific angular momentum')
    if mass_error > 1.0e-10:
        raise RuntimeError('closed hydro expansion lost mass: relative error %.6g' % mass_error)

    # Cell-centered potential energy uses the extensive cell mass, which
    # already contains the spherical cell volume.  This is the discrete
    # Eulerian counterpart of the shell potential energy.
    potential_initial = -central_mass * np.sum(initial_mass / initial_radius)
    potential_final = -central_mass * np.sum(saved_mass / saved_radius)
    potential_change = potential_final - potential_initial
    potential_work_residual = cumulative_gravity_work + potential_change

    # The central-gravity potential closes the gas-energy audit.  The shell
    # ODE conserves this quantity even though gas kinetic and rotational energy
    # separately exchange during the expansion.
    saved_total_energy = (
        np.sum(saved_energy) - np.sum(central_mass * saved_mass / saved_radius)
    )
    shell_specific_energy = (
        0.5 * shell_final[1]**2
        + 0.5 * shell_j**2 / shell_final[0]**2
        - central_mass / shell_final[0]
    )
    mapped_specific_energy = np.interp(
        saved_radius, shell_final[0, order], shell_specific_energy[order]
    )
    ode_total_energy = np.sum(saved_mass * mapped_specific_energy)
    energy_error = float(abs(saved_total_energy - ode_total_energy))
    energy_scale = max(abs(float(ode_total_energy)), 1.0e-12)
    if energy_error / energy_scale > 2.0e-3:
        raise RuntimeError(
            'hydro expansion total-energy audit failed: relative error %.6g'
            % (energy_error / energy_scale)
        )

    density = np.asarray(saved.rho[active], dtype=float)
    temperature = np.asarray(saved.temp[active], dtype=float)
    mu = np.asarray(saved.mu[active], dtype=float)
    pressure = np.asarray(sim.fluid.eos.pressure(density, temperature, mu), dtype=float)
    pressure_ratio = np.divide(
        pressure / np.maximum(density, np.finfo(float).tiny),
        central_mass / np.maximum(saved_radius, np.finfo(float).tiny),
    )
    figure = ROOT / runparams['savedir'] / 'GasCentrifugalHydroExpansion1D.jpg'
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flat
    axes[0].plot(saved_radius, saved_velocity, ':o', markersize=4, label='Rsim')
    axes[0].plot(saved_radius, ode_velocity, '--', label='pressureless shell ODE')
    axes[0].set_ylabel('radial velocity')
    axes[1].plot(saved_radius, density, ':o', markersize=4, label='Rsim density')
    axes[1].set_ylabel('density')
    axes[2].plot(saved_radius, saved_j, ':o', markersize=4, label='Rsim $J/M$')
    axes[2].plot(saved_radius, ode_j, '--', label='advected shell $j$')
    axes[2].set_ylabel('specific angular momentum')
    axes[3].semilogy(
        saved_radius, np.maximum(pressure_ratio, np.finfo(float).tiny),
        ':o', markersize=4, label='Rsim pressure support',
    )
    axes[3].axhline(1.0, color='k', linestyle=':')
    axes[3].set_ylabel('thermal / dynamical scale')
    axes[4].bar(
        ['ODE', 'Rsim'], [ode_total_energy, saved_total_energy],
        color=['C0', 'C1'],
    )
    axes[4].set_ylabel('gas + gravitational energy')
    axes[4].set_title('energy audit')
    axes[5].axis('off')
    for axis in axes[:4]:
        axis.set_xlabel('radius')
        axis.grid(alpha=0.25)
        axis.legend()
    axes[4].grid(axis='y', alpha=0.25)
    fig.suptitle('Centrifugal hydro expansion benchmark')
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print('centrifugal hydro expansion check passed')
    print('maximum velocity error = %.6g' % velocity_error)
    print('maximum mapped J/M error = %.6g' % j_error)
    print('relative global mass error = %.6g' % mass_error)
    print('cumulative gravity work = %.6g' % cumulative_gravity_work)
    print('cell-centered potential change = %.6g' % potential_change)
    print('reported potential change = %.6g' % cumulative_potential_change)
    print('face potential-energy boundary flux = %.6g' % cumulative_potential_flux)
    print('gravity/potential closure residual = %.6g' % potential_work_residual)
    print('total energy audit error = %.6g' % energy_error)
    print('maximum thermal/dynamical scale = %.6g' % np.max(pressure_ratio))
    print('figure = %s' % figure)


if __name__ == '__main__':
    main()
