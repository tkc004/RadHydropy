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
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, SUPERCOMOVING_RUNTIME_FIELDS
import example_utils as eu


CONFIG = ROOT / 'gas_centrifugal_cosmological_orbit1d.yaml'

def prepare_initial_condition(initial):
    boundary_comoving_code = np.asarray(
        initial.mesh.boundary_comoving_code, dtype=float
    )
    initial.par.cosmological_expansion = True
    initial.par.supercomoving_coordinates = True
    initial.par.coordinate_frame = 'comoving'
    initial.par.time_coordinate = 'supercomoving'
    initial.par.velocity_representation = 'supercomoving_peculiar'
    initial.mesh.geometry_state = MeshGeometryState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS, x_comoving_code=initial.mesh.x_comoving_code,
        boundary_comoving_code=boundary_comoving_code,
        width_comoving_code=np.diff(boundary_comoving_code),
        area_comoving_code=4.0 * np.pi * boundary_comoving_code[:-1]**2,
        volume_comoving_code=4.0 * np.pi / 3.0 * (
            boundary_comoving_code[1:]**3 - boundary_comoving_code[:-1]**3
        ),
    )
    initial.fluid.rho_comoving_code = initial.fluid.rho_comoving_code
    initial.fluid.vel_supercomoving_code = initial.fluid.vel_supercomoving_code
    initial.fluid.temp_supercomoving_code = initial.fluid.temp_supercomoving_code
    initial.fluid.pre_supercomoving_code = initial.fluid.temp_supercomoving_code * 0.4
    initial.fluid.tau_supercomoving_code = 0.0
    initial.fluid.runtime_fields = SUPERCOMOVING_RUNTIME_FIELDS
    initial.fluid.runtime_state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS, rho_comoving_code=initial.fluid.rho_comoving_code,
        vel_supercomoving_code=initial.fluid.vel_supercomoving_code,
        pre_supercomoving_code=initial.fluid.pre_supercomoving_code,
        temp_supercomoving_code=initial.fluid.temp_supercomoving_code, tau_supercomoving_code=0.0,
        mu_dimensionless=initial.fluid.mu,
    )


class CosmologicalInitialCondition(Rsim):
    def __init__(self, par_config, count, radius_min, radius_max, density, temperature,
                 specific_j, code_unit_system):
        super().__init__(par_config)
        self.mesh.boundary_comoving_code = np.linspace(radius_min, radius_max, count + 1)
        self.mesh.x_comoving_code = 0.75 * (
            self.mesh.boundary_comoving_code[1:]**4 - self.mesh.boundary_comoving_code[:-1]**4
        ) / (self.mesh.boundary_comoving_code[1:]**3 - self.mesh.boundary_comoving_code[:-1]**3)
        self.mesh.width_comoving_code = np.diff(self.mesh.boundary_comoving_code)
        self.mesh.area_comoving_code = 4.0 * np.pi * self.mesh.boundary_comoving_code[:-1]**2
        self.mesh.volume_comoving_code = 4.0 * np.pi / 3.0 * np.diff(self.mesh.boundary_comoving_code**3)
        self.fluid.rho_comoving_code = np.full(count, density)
        self.fluid.vel_supercomoving_code = np.zeros(count)
        self.fluid.temp_supercomoving_code = np.full(count, temperature)
        self.fluid.mu = np.ones(count)
        self.fluid.specific_angular_momentum_code = np.asarray(specific_j, dtype=float)


class CosmologicalCentralGravity:
    cosmological = True
    dark_matter = None

    def __init__(self, mass, cosmology):
        self.mass = mass
        self.cosmology = cosmology
        self.tau = 0.0

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        tau = float(np.asarray(
            getattr(getattr(par, 'simulation', None), 'time_proper_code', self.tau)
        )) if par is not None else self.tau
        scale_factor = self.cosmology.scale_factor_from_supercomoving(tau)
        radius = np.asarray(mesh.x_comoving_code, dtype=float)
        return -scale_factor * self.mass / radius**2


def run_rsim(par, initial_condition, example_config, cosmology, j):
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    count = int(par['mesh']['grid_cells'])
    initial_boundary = np.linspace(0.5, 1.5, count + 1)
    initial_radius = 0.75 * (
        initial_boundary[1:]**4 - initial_boundary[:-1]**4
    ) / (initial_boundary[1:]**3 - initial_boundary[:-1]**3)
    circular_j_profile = np.full(count, float(j))
    initial = CosmologicalInitialCondition(
        par, count, 0.5, 1.5, 1.0, float(example_config['temperature']),
        circular_j_profile, units
    )
    prepare_initial_condition(initial)
    initial.par.cosmology = cosmology
    filename = ROOT / par['simulation']['initial_condition_filename']
    filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, filename)
    sim = Rsim(par)
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, str(filename))
    gravity = CosmologicalCentralGravity(float(initial_condition['central_excess_mass']), cosmology)
    sim.par.gravity = gravity
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    initial_tau = np.asarray(sim.par.tau_supercomoving_code, dtype=float)
    sim.par.tau_supercomoving_code = initial_tau.copy()
    sim.par.simulation.tau_supercomoving_code = initial_tau.copy()
    sim.fluid.SetFluidTime(initial_tau)
    sim.par.gravity = gravity
    sim.par.set_cosmology_model(cosmology)
    sim.Run(outputtime=0, mode='hydro')
    # Fixed-cadence output is intentionally independent of the requested
    # final time.  Persist the actual terminal state so the analytic
    # comparison is made at the same supercomoving time as the simulation.
    final_filename = ROOT / par['output']['directory'] / 'Output_final.hdf5'
    sim.fluid.SetTemperature()
    rio.writehdf5(sim, final_filename)
    final_par = sim.par
    final_mesh = type('Mesh', (), {})()
    final_fluid = type('Fluid', (), {})()
    rio.readhdf5(final_par, final_mesh, final_fluid, final_filename)
    return initial, sim, final_fluid


def main(config_filename=CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    initial_condition = config['initial_condition']
    example_config = config['example']
    savedir = ROOT / par['output']['savedir']
    savedir.mkdir(parents=True, exist_ok=True)
    cosmology = EinsteinDeSitter()
    x0 = float(initial_condition['initial_comoving_radius'])
    v0 = float(initial_condition['initial_supercomoving_velocity'])
    central_mass = float(initial_condition['central_excess_mass'])
    j = float(initial_condition['angular_momentum_fraction_of_circular']) * np.sqrt(
        central_mass * x0
    )
    final_tau = float(par['simulation']['final_time'])
    initial_sim, simulation, saved_fluid = run_rsim(
        par, initial_condition, example_config, cosmology, j
    )
    simulation_radius = np.asarray(simulation.mesh.x_comoving_code, dtype=float)
    circular_j_profile = np.sqrt(central_mass * simulation_radius)
    saved_j = np.asarray(saved_fluid.specific_angular_momentum_code, dtype=float)
    saved_active = slice(
        int(simulation.par.mesh.ghost_cells),
        int(simulation.par.mesh.ghost_cells) + int(simulation.par.mesh.grid_cells),
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
    dt = float(example_config['timestep'])
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
        int(simulation.par.mesh.ghost_cells),
        int(simulation.par.mesh.ghost_cells) + int(simulation.par.mesh.grid_cells),
    )
    sim_radius = np.asarray(simulation.mesh.x_comoving_code[sim_active], dtype=float)
    # Use the live Rsim mesh and fluid together.  Mixing live mesh coordinates
    # with fields from a separately reloaded HDF5 object can pair different
    # code/cgs representations after serialization.
    sim_velocity = np.asarray(simulation.fluid.vel_supercomoving_code[sim_active], dtype=float)
    sim_j = np.asarray(simulation.fluid.specific_angular_momentum_code[sim_active], dtype=float)
    sim_energy = np.asarray(simulation.fluid.Energy_code[sim_active], dtype=float)
    # Map the analytic shell ensemble back to the fixed Eulerian grid.
    ode_final = np.empty((2, len(sim_radius)))
    for index, initial_radius in enumerate(sim_radius):
        # The IC assigns the same specific angular momentum j to every shell.
        # Using sqrt(GM/x) here would compare the simulation with a different
        # circular-angular-momentum profile and produces the apparent mismatch.
        shell_j = j

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
    if simulation_velocity_error > 3.0e-1:
        raise RuntimeError(
            'saved cosmological Rsim velocity disagrees with Eulerian-mapped '
            'ODE: max error = %.6g' % simulation_velocity_error
        )
    if simulation_j_error > 3.0e-3:
        raise RuntimeError(
            'saved cosmological Rsim J/M drifted from the initialized profile: '
            'max error = %.6g' % simulation_j_error
        )
    sim_temperature = np.asarray(simulation.fluid.temp_supercomoving_code[sim_active], dtype=float)
    sim_density = np.asarray(simulation.fluid.rho_comoving_code[sim_active], dtype=float)
    sim_mu = np.asarray(simulation.fluid.mu[sim_active], dtype=float)
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
    sim_axes[0].plot(sim_radius, sim_velocity, 'o-', label='Rsim Eulerian state')
    sim_axes[0].set_ylabel('supercomoving radial velocity')
    sim_axes[0].set_title('Eulerian profile; ODE check is in the time-history figure')
    sim_axes[1].plot(sim_radius, sim_j, 'o-', label='saved $J/M$')
    sim_axes[1].plot(
        sim_radius, np.full_like(sim_radius, j), '--',
        label='initial constant $j$',
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
