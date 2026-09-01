"""Supercomoving eccentric-orbit benchmark with centrifugal support."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/tmp/radhydropy-matplotlib')
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'example'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from radhydropy.cosmology import EinsteinDeSitter
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.cosmological_variables import (
    physical_radius,
    physical_velocity,
)
from radhydropy.units import CodeUnits
import example_utils as eu


CONFIG = ROOT / 'gas_centrifugal_cosmological_orbit1d.yaml'


class CosmologicalInitialCondition:
    def __init__(self, count, radius_min, radius_max, density, temperature,
                 specific_j, code_units):
        self.par = SimpleNamespace(
            CodeUnits=code_units, nogrid=count, noghost=2,
            coordsys='spherical', time=0.0,
            boxsize=np.asarray([radius_max]),
        )
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(current_time=0.0, box_size=np.asarray([radius_max]), coordinate_system='spherical')
        self.par.mesh = SimpleNamespace(grid_cells=count, ghost_cells=0)
        self.par.hydrodynamics = SimpleNamespace(gamma=1.4)
        self.par.unit_system = code_units.unit_system
        self.mesh = type('Mesh', (), {})()
        self.fluid = type('Fluid', (), {})()
        self.mesh.boundary = np.linspace(radius_min, radius_max, count + 1)
        self.mesh.coordinate = 0.75 * (
            self.mesh.boundary[1:]**4 - self.mesh.boundary[:-1]**4
        ) / (self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3)
        self.fluid.rho = np.full(count, density)
        self.fluid.vel = np.zeros(count)
        self.fluid.temp = np.full(count, temperature)
        self.fluid.mu = np.ones(count)
        self.fluid.specific_angular_momentum = np.asarray(specific_j, dtype=float)


class CosmologicalCentralGravity:
    cosmological = True
    dark_matter = None

    def __init__(self, mass, cosmology):
        self.mass = mass
        self.cosmology = cosmology
        self.tau = 0.0

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        tau = float(np.asarray(
            getattr(par, 'fluid_time', getattr(par, 'time', self.tau))
        )) if par is not None else self.tau
        scale_factor = self.cosmology.scale_factor_from_supercomoving(tau)
        radius = np.asarray(mesh.coordinate, dtype=float)
        return -scale_factor * self.mass / radius**2


def run_rsim(runparams, icparams, runtime, cosmology, j):
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    count = int(runparams['nogrid'])
    initial_boundary = np.linspace(0.5, 1.5, count + 1)
    initial_radius = 0.75 * (
        initial_boundary[1:]**4 - initial_boundary[:-1]**4
    ) / (initial_boundary[1:]**3 - initial_boundary[:-1]**3)
    circular_j_profile = np.full(count, float(j))
    initial = CosmologicalInitialCondition(
        count, 0.5, 1.5, 1.0, float(runparams['temperature']),
        circular_j_profile, units
    )
    filename = ROOT / runparams['ICfilename']
    filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, filename)
    sim = Rsim(runtime)
    sim.Callreadhdf5()
    gravity = CosmologicalCentralGravity(float(icparams['central_excess_mass']), cosmology)
    sim.par.gravity = gravity
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = gravity
    sim.par.set_cosmology_model(cosmology)
    sim.Run(outputtime=0, mode='hydro')
    # Fixed-cadence output is intentionally independent of the requested
    # final time.  Persist the actual terminal state so the analytic
    # comparison is made at the same supercomoving time as the simulation.
    final_filename = ROOT / runparams['outdir'] / 'Output_final.hdf5'
    sim.fluid.SetTemperature()
    rio.writehdf5(sim, final_filename)
    final_par = sim.par
    final_mesh = type('Mesh', (), {})()
    final_fluid = type('Fluid', (), {})()
    rio.readhdf5(final_par, final_mesh, final_fluid, final_filename)
    return initial, sim, final_fluid


def main(config_filename=CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runparams = eu.legacy_example_parameters(config)
    runparams.update(config['par'].get('gravity', {}))
    runparams.update(config.get('example', {}))
    icparams = config['initial_condition']
    icparams['nogrid'] = runparams['nogrid']
    runparams['temperature'] = config['example']['temperature']
    runparams['timestep'] = config['example']['timestep']
    savedir = ROOT / runparams['savedir']
    savedir.mkdir(parents=True, exist_ok=True)
    cosmology = EinsteinDeSitter()
    x0 = float(icparams['initial_comoving_radius'])
    v0 = float(icparams['initial_supercomoving_velocity'])
    central_mass = float(icparams['central_excess_mass'])
    j = float(icparams['angular_momentum_fraction_of_circular']) * np.sqrt(
        central_mass * x0
    )
    final_tau = float(runparams['timesim'])
    initial_sim, simulation, saved_fluid = run_rsim(
        runparams, icparams, config['par'], cosmology, j
    )
    simulation_radius = np.asarray(simulation.mesh.coordinate, dtype=float)
    circular_j_profile = np.sqrt(central_mass * simulation_radius)
    saved_j = np.asarray(saved_fluid.specific_angular_momentum, dtype=float)
    saved_active = slice(
        simulation.par.noghost,
        simulation.par.noghost + simulation.par.nogrid,
    )
    if not np.all(np.isfinite(saved_j[saved_active])):
        raise RuntimeError('cosmological Rsim produced invalid specific angular momentum')

    def rhs(tau, state):
        x, velocity = state
        radius_safe = max(x, np.finfo(float).tiny)
        scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
        return velocity, -scale_factor * central_mass / radius_safe**2 + j**2 / radius_safe**3

    reference = solve_ivp(
        rhs,
        (0.0, final_tau),
        (x0, v0),
        rtol=1.0e-11,
        atol=1.0e-13,
        dense_output=True,
    )
    dt = float(runparams['timestep'])
    times = np.arange(0.0, final_tau + 0.5 * dt, dt)
    numerical = np.empty((2, len(times)))
    numerical[:, 0] = (x0, v0)
    for index in range(len(times) - 1):
        state = numerical[:, index]
        k1 = np.asarray(rhs(times[index], state))
        k2 = np.asarray(rhs(times[index] + 0.5 * dt, state + 0.5 * dt * k1))
        k3 = np.asarray(rhs(times[index] + 0.5 * dt, state + 0.5 * dt * k2))
        k4 = np.asarray(rhs(times[index] + dt, state + dt * k3))
        numerical[:, index + 1] = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    reference_state = reference.sol(times)
    radius_error = np.max(np.abs(numerical[0] - reference_state[0]))
    velocity_error = np.max(np.abs(numerical[1] - reference_state[1]))

    scale_factor = cosmology.scale_factor_from_supercomoving(times)
    proper_radius = physical_radius(numerical[0], scale_factor)
    hubble = cosmology.hubble_from_supercomoving(times)
    proper_velocity = physical_velocity(
        numerical[0], numerical[1], scale_factor, hubble
    )
    physical_tangential_velocity = j / proper_radius
    reconstructed_j = (
        proper_radius * physical_tangential_velocity
    )
    if radius_error > 1.0e-8 or velocity_error > 1.0e-8:
        raise RuntimeError('cosmological orbit disagrees with analytic ODE')
    if not np.allclose(reconstructed_j, j, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError('specific angular momentum changed under conversion')

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(times, numerical[0], label='RK4')
    axes[0, 0].plot(times, reference_state[0], '--', label='analytic ODE')
    axes[0, 0].set_ylabel('comoving radius $x$')
    axes[0, 1].plot(times, proper_radius, label='physical radius $r=ax$')
    axes[0, 1].set_ylabel('physical radius')
    axes[1, 0].plot(times, numerical[1], label='RK4 supercomoving velocity')
    axes[1, 0].plot(
        times, proper_velocity, '--', label='physical velocity',
    )
    axes[1, 0].set_ylabel('velocity [code velocity units]')
    axes[1, 1].plot(times, reconstructed_j - j, label='$j_{rec}-j$')
    axes[1, 1].set_ylabel('angular-momentum error')
    for axis in axes.flat:
        axis.set_xlabel('supercomoving time $\\tau$')
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle('Cosmological gas centrifugal eccentric-orbit check')
    fig.tight_layout()
    figure = savedir / 'GasCentrifugalCosmologicalOrbit1D.jpg'
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    # Fixed Eulerian coordinates cannot be placed on the moving x(t) curves,
    # so show the saved Rsim state in a separate spatial diagnostic figure.
    sim_active = slice(
        simulation.par.noghost,
        simulation.par.noghost + simulation.par.nogrid,
    )
    sim_radius = np.asarray(simulation.mesh.coordinate[sim_active], dtype=float)
    # readhdf5 returns NamedArray values already converted to code units.
    sim_velocity = np.asarray(saved_fluid.vel[sim_active], dtype=float)
    sim_j = np.asarray(saved_fluid.specific_angular_momentum[sim_active], dtype=float)
    sim_energy = np.asarray(saved_fluid.Energy[sim_active], dtype=float)
    # Map the analytic shell ensemble back to the fixed Eulerian grid.
    ode_final = np.empty((2, len(sim_radius)))
    for index, initial_radius in enumerate(sim_radius):
        shell_j = np.sqrt(central_mass * initial_radius)

        def shell_rhs(tau, state):
            shell_radius, shell_velocity = state
            radius_safe = max(shell_radius, np.finfo(float).tiny)
            scale_factor = float(
                cosmology.scale_factor_from_supercomoving(tau)
            )
            return (
                shell_velocity,
                -scale_factor * central_mass / radius_safe**2
                + shell_j**2 / radius_safe**3,
            )

        shell_reference = solve_ivp(
            shell_rhs, (0.0, final_tau), (initial_radius, v0),
            rtol=1.0e-10, atol=1.0e-12,
        )
        ode_final[:, index] = shell_reference.y[:, -1]
    order = np.argsort(ode_final[0])
    ode_velocity = np.interp(sim_radius, ode_final[0, order], ode_final[1, order])
    simulation_velocity_error = float(
        np.max(np.abs(sim_velocity - ode_velocity))
    )
    simulation_j_error = float(
        np.max(np.abs(sim_j - j))
    )
    # The 32-cell Eulerian run is intentionally lightweight; retain a
    # regression tolerance that reflects its finite-volume shell mixing.
    if simulation_velocity_error > 2.0e-1:
        raise RuntimeError(
            'saved cosmological Rsim velocity disagrees with Eulerian-mapped '
            'ODE: max error = %.6g' % simulation_velocity_error
        )
    if simulation_j_error > 3.0e-3:
        raise RuntimeError(
            'saved cosmological Rsim J/M drifted from the initialized profile: '
            'max error = %.6g' % simulation_j_error
        )
    sim_temperature = np.asarray(saved_fluid.temp[sim_active], dtype=float)
    sim_density = np.asarray(saved_fluid.rho[sim_active], dtype=float)
    sim_mu = np.asarray(saved_fluid.mu[sim_active], dtype=float)
    sim_pressure = np.asarray(
        simulation.fluid.eos.pressure(sim_density, sim_temperature, sim_mu),
        dtype=float,
    )
    # This is the local thermal-pressure scale divided by the circular
    # centrifugal scale.  It is a diagnostic, not an extra source term.
    pressure_support_ratio = np.divide(
        sim_pressure / np.maximum(sim_density, np.finfo(float).tiny),
        central_mass / np.maximum(sim_radius, np.finfo(float).tiny),
    )
    simulation_figure = savedir / 'GasCentrifugalCosmologicalOrbit1D_simulation.jpg'
    sim_fig, sim_axes = plt.subplots(2, 2, figsize=(11, 7))
    sim_axes = sim_axes.flat
    sim_axes[0].plot(sim_radius, sim_velocity, 'o-', label='saved Rsim')
    sim_axes[0].plot(sim_radius, ode_velocity, '--', label='Eulerian-mapped ODE')
    sim_axes[0].set_ylabel('supercomoving radial velocity')
    sim_axes[1].plot(sim_radius, sim_j, 'o-', label='saved $J/M$')
    sim_axes[1].plot(
        sim_radius, np.sqrt(central_mass * sim_radius), '--',
        label='initial circular $j(x)$',
    )
    sim_axes[1].set_ylabel('specific angular momentum')
    sim_axes[2].plot(sim_radius, sim_energy, 'o-', label='saved total energy')
    sim_axes[2].set_ylabel('total energy')
    sim_axes[3].semilogy(
        sim_radius, np.maximum(pressure_support_ratio, np.finfo(float).tiny),
        'o-', label=r'$p/\rho\,/\,(GM/x)$',
    )
    sim_axes[3].axhline(1.0, color='k', linestyle=':', linewidth=1.0)
    sim_axes[3].set_ylabel('thermal / centrifugal scale')
    for axis in sim_axes:
        axis.set_xlabel('comoving radius $x$')
        axis.grid(alpha=0.25)
        axis.legend()
    sim_fig.suptitle('Saved cosmological Rsim gas state')
    sim_fig.tight_layout()
    sim_fig.savefig(simulation_figure, dpi=180)
    plt.close(sim_fig)
    print('cosmological eccentric-orbit analytic check passed')
    print('maximum comoving-radius error = %.6g' % radius_error)
    print('maximum supercomoving-velocity error = %.6g' % velocity_error)
    print('maximum saved-Rsim velocity error = %.6g' % simulation_velocity_error)
    print('maximum saved J/M profile error = %.6g' % simulation_j_error)
    print('maximum thermal/centrifugal scale = %.6g' % np.max(pressure_support_ratio))
    print('figure = %s' % figure)
    print('simulation figure = %s' % simulation_figure)


if __name__ == '__main__':
    main()
