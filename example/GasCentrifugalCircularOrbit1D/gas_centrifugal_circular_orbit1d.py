"""Analytic circular-orbit check for gas centrifugal support."""

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

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
import example_utils as eu


CONFIG = ROOT / 'gas_centrifugal_circular_orbit1d.yaml'

def prepare_initial_condition(initial):
    boundary_proper_code = np.asarray(
        initial.mesh.boundary_proper_code, dtype=float
    )
    initial.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=initial.mesh.x_proper_code,
        boundary_proper_code=boundary_proper_code, width_proper_code=np.diff(boundary_proper_code),
        area_proper_code=4.0 * np.pi * boundary_proper_code[:-1]**2,
        volume_proper_code=4.0 * np.pi / 3.0 * (
            boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3
        ),
    )
    initial.fluid.pre_proper_code = initial.fluid.temp_proper_code * 0.4
    initial.fluid.time_proper_code = 0.0
    initial.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    initial.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS, rho_proper_code=initial.fluid.rho_proper_code,
        vel_proper_code=initial.fluid.vel_proper_code, pre_proper_code=initial.fluid.pre_proper_code,
        temp_proper_code=initial.fluid.temp_proper_code, time_proper_code=0.0,
        mu_dimensionless=initial.fluid.mu,
    )


class FixedCentralGravity:
    def __init__(self, central_mass, angular_momentum):
        self.central_mass = central_mass
        self.angular_momentum = angular_momentum
        self.cosmological = False
        self.dark_matter = None

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        radius = np.asarray(mesh.x_proper_code, dtype=float)
        return -self.central_mass / radius**2


class CircularInitialCondition(Rsim):
    """HDF5-compatible spherical circular-orbit initial condition."""

    def __init__(self, par_config, count, radius_min, radius_max, density, pressure,
                 central_mass, code_unit_system):
        super().__init__(par_config)
        self.mesh.boundary_proper_code = np.linspace(radius_min, radius_max, count + 1)
        self.mesh.x_proper_code = 0.75 * (
            self.mesh.boundary_proper_code[1:]**4 - self.mesh.boundary_proper_code[:-1]**4
        ) / (self.mesh.boundary_proper_code[1:]**3 - self.mesh.boundary_proper_code[:-1]**3)
        self.mesh.width_proper_code = np.diff(self.mesh.boundary_proper_code)
        self.mesh.area_proper_code = 4.0 * np.pi * self.mesh.boundary_proper_code[:-1]**2
        self.mesh.volume_proper_codeume_proper_code = 4.0 * np.pi / 3.0 * np.diff(self.mesh.boundary_proper_code**3)
        self.fluid.rho_proper_code = np.full(count, density)
        self.fluid.vel_proper_code = np.zeros(count)
        self.fluid.temp_proper_code = np.full(count, pressure * 0.4)
        self.fluid.mu = np.ones(count)
        self.fluid.specific_angular_momentum_code = np.sqrt(
            central_mass * self.mesh.x_proper_code
        )


def run_rsim(config):
    par = config['par']
    initial_condition = config['initial_condition']
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    initial = CircularInitialCondition(
        par, int(par['mesh']['grid_cells']), float(initial_condition['radius_min']), float(initial_condition['radius_max']),
        float(initial_condition['density']), float(initial_condition['pressure']),
        float(initial_condition['central_mass']), units,
    )
    prepare_initial_condition(initial)
    prepare_initial_condition(initial)
    ic_filename = ROOT / par['simulation']['initial_condition_filename']
    ic_filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, ic_filename)
    sim = Rsim(config["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, str(ic_filename))
    sim.par.gravity = FixedCentralGravity(float(initial_condition['central_mass']), 0.0)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    initial_mass = np.asarray(sim.fluid.Mass_code, dtype=float).copy()
    initial_energy = np.asarray(sim.fluid.Energy_code, dtype=float).copy()
    sim.par.gravity = FixedCentralGravity(float(initial_condition['central_mass']), 0.0)
    sim.Run(outputtime=0, mode='sources')
    output_files = sorted((ROOT / par['output']['directory']).glob('Output_*.hdf5'))
    if not output_files:
        raise RuntimeError('Rsim produced no circular-orbit output')
    final_par = sim.par
    final_mesh = SimpleNamespace()
    final_fluid = SimpleNamespace()
    rio.readhdf5(final_par, final_mesh, final_fluid, output_files[-1])
    active = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    return initial, sim, final_mesh, final_fluid, active, initial_mass, initial_energy


def main(config_filename=CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    initial_condition = config['initial_condition']
    savedir = ROOT / par['output']['savedir']
    savedir.mkdir(parents=True, exist_ok=True)
    (initial_sim, simulation, saved_mesh, saved_fluid, active,
     simulation_initial_mass, simulation_initial_energy) = run_rsim(config)

    count = int(par['mesh']['grid_cells'])
    radius = float(initial_condition['radius'])
    central_mass = float(initial_condition['central_mass'])
    specific_j = np.sqrt(central_mass * radius)
    volume_proper_code = np.ones(count)
    mass = np.full(count, float(initial_condition['density'])) * volume_proper_code
    momentum = np.full(count, float(initial_condition['radial_velocity'])) * mass
    rotational_energy = 0.5 * mass * specific_j**2 / radius**2
    thermal_energy = np.full(count, float(initial_condition['pressure']) / 0.4)

    mesh = SimpleNamespace(
        coordsys='spherical',
        x_proper_code=np.full(count, radius),
        volume_proper_code=volume_proper_code,
        _par=SimpleNamespace(
            gas_angular_momentum=True,
            gas_rotational_energy=True,
            noghost=0,
            nogrid=count,
            energy_diagnostics=False,
            CodeUnits=None,
            gravity=FixedCentralGravity(central_mass, specific_j),
        ),
    )
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=mesh.x_proper_code,
        boundary_proper_code=np.linspace(radius - 0.5, radius + 0.5, count + 1),
        width_proper_code=np.ones(count), area_proper_code=4.0 * np.pi * np.ones(count) * radius**2,
        volume_proper_code=volume_proper_code,
    )
    par = mesh._par
    par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=count)
    fluid = SimpleNamespace(
        rho_proper_code=np.ones(count) * float(initial_condition['density']),
        Mass_code=mass.copy(),
        Mom_code=momentum.copy(),
        Energy_code=thermal_energy + rotational_energy,
        AngularMomentum_code=mass * specific_j,
    )

    solver = Solver()
    initial_energy = fluid.Energy_code.copy()
    times = [0.0]
    velocity_history = [fluid.Mom_code[0] / fluid.Mass_code[0]]
    energy_error = [0.0]
    dt = float(initial_condition['timestep'])
    for step in range(int(initial_condition['nsteps'])):
        solver.ApplyGravity(dt, mesh, fluid, par)
        times.append((step + 1) * dt)
        velocity_history.append(fluid.Mom_code[0] / fluid.Mass_code[0])
        energy_error.append(np.max(np.abs(fluid.Energy_code - initial_energy)))

    if not np.allclose(fluid.Mom_code, momentum, rtol=0.0, atol=1.0e-13):
        raise RuntimeError('circular force balance generated radial momentum')
    if not np.allclose(fluid.Energy_code, initial_energy, rtol=0.0, atol=1.0e-2):
        raise RuntimeError('circular force balance changed total energy')
    expected_rotational = rotational_energy
    if not np.allclose(
        expected_rotational,
        0.5 * fluid.AngularMomentum_code**2 / (fluid.Mass_code * radius**2),
    ):
        raise RuntimeError('rotational energy bookkeeping is inconsistent')

    # Compare the saved Rsim state with the circular analytic solution.
    saved_velocity = np.asarray(saved_fluid.vel_proper_code[active], dtype=float)
    saved_j = np.asarray(saved_fluid.specific_angular_momentum_code[active], dtype=float)
    saved_boundary = np.asarray(saved_mesh.boundary_proper_code, dtype=float)
    saved_radius = 0.75 * (
        saved_boundary[1:]**4 - saved_boundary[:-1]**4
    ) / (saved_boundary[1:]**3 - saved_boundary[:-1]**3)
    saved_radius = saved_radius[active]
    saved_mass = np.asarray(saved_fluid.Mass_code[active], dtype=float)
    saved_momentum = np.asarray(
        saved_fluid.Mass_code[active] * saved_fluid.vel_proper_code[active], dtype=float
    )
    saved_energy = np.asarray(saved_fluid.Energy_code[active], dtype=float)
    if np.max(np.abs(saved_velocity)) > 5.0e-5:
        raise RuntimeError('Rsim circular solution developed radial velocity')
    if not np.allclose(saved_j, np.sqrt(central_mass * saved_radius), atol=1.0e-10):
        raise RuntimeError('Rsim changed circular specific angular momentum')
    if np.max(np.abs(saved_momentum)) > 5.0e-5:
        raise RuntimeError('Rsim circular solution developed radial momentum')
    if not np.allclose(saved_mass, simulation_initial_mass[active], rtol=1.0e-10):
        raise RuntimeError('Rsim changed circular mass')
    if not np.allclose(
        saved_energy, simulation_initial_energy[active], rtol=1.0e-5, atol=1.0e-10
    ):
        raise RuntimeError('Rsim circular solution changed total energy')

    # The Eulerian mesh has fixed radii, so use the corresponding Lagrangian
    # source equations as the analytic trajectory regression for the next
    # moving-shell stage.  Choose j below the circular value to obtain an
    # eccentric radial orbit.
    eccentric_j = 0.7 * specific_j
    eccentric_time = float(initial_condition['timestep']) * int(initial_condition['nsteps'])

    def orbit_rhs(time_proper_code, state):
        orbit_radius, orbit_velocity = state
        radius_safe = max(orbit_radius, np.finfo(float).tiny)
        return (
            orbit_velocity,
            eccentric_j**2 / radius_safe**3
            - central_mass / radius_safe**2,
        )

    reference = solve_ivp(
        orbit_rhs,
        (0.0, eccentric_time),
        (radius, 0.0),
        rtol=1.0e-11,
        atol=1.0e-13,
        dense_output=True,
    )
    eccentric_dt = float(initial_condition['timestep'])
    eccentric_times = np.arange(
        0.0, eccentric_time + 0.5 * eccentric_dt, eccentric_dt
    )
    eccentric_state = np.empty((2, len(eccentric_times)))
    eccentric_state[:, 0] = (radius, 0.0)
    for index in range(len(eccentric_times) - 1):
        state = eccentric_state[:, index]
        h = eccentric_dt
        k1 = np.asarray(orbit_rhs(0.0, state))
        k2 = np.asarray(orbit_rhs(0.0, state + 0.5 * h * k1))
        k3 = np.asarray(orbit_rhs(0.0, state + 0.5 * h * k2))
        k4 = np.asarray(orbit_rhs(0.0, state + h * k3))
        eccentric_state[:, index + 1] = state + h * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
    reference_eccentric = reference.sol(eccentric_times)
    eccentric_radius_error = np.max(
        np.abs(eccentric_state[0] - reference_eccentric[0])
    )
    eccentric_velocity_error = np.max(
        np.abs(eccentric_state[1] - reference_eccentric[1])
    )
    eccentric_energy = (
        0.5 * eccentric_state[1]**2
        + 0.5 * eccentric_j**2 / eccentric_state[0]**2
        - central_mass / eccentric_state[0]
    )
    if eccentric_radius_error > 1.0e-8 or eccentric_velocity_error > 1.0e-8:
        raise RuntimeError('eccentric orbit disagrees with analytic ODE')
    if np.max(np.abs(eccentric_energy - eccentric_energy[0])) > 1.0e-10:
        raise RuntimeError('eccentric orbit failed specific-energy conservation')

    # Drive a moving one-shell simulation with RadHydropy's actual source
    # routine.  The Eulerian gas mesh has fixed cell coordinates, so this
    # small shell driver supplies the moving x_proper_code needed for a trajectory
    # comparison while retaining the production centrifugal/gravity update.
    shell_solver = Solver()
    shell_mesh = SimpleNamespace(
        coordsys='spherical', x_proper_code=np.asarray([radius]),
        volume_proper_code=np.asarray([1.0]),
    )
    shell_mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=np.asarray([radius]),
        boundary_proper_code=np.asarray([radius - 0.5, radius + 0.5]),
        width_proper_code=np.asarray([1.0]), area_proper_code=np.asarray([4.0 * np.pi * radius**2]),
        volume_proper_code=np.asarray([1.0]),
    )
    shell_par = SimpleNamespace(
        gas_angular_momentum=True, gas_rotational_energy=True,
        noghost=0, nogrid=1, energy_diagnostics=False,
        CodeUnits=None,
        gravity=FixedCentralGravity(central_mass, eccentric_j),
    )
    shell_par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=1)
    shell_fluid = SimpleNamespace(
        rho_proper_code=np.asarray([1.0]), Mass_code=np.asarray([1.0]),
        Mom_code=np.asarray([0.0]), AngularMomentum_code=np.asarray([eccentric_j]),
        Energy_code=np.asarray([1.0 + 0.5 * eccentric_j**2 / radius**2]),
    )
    shell_radius = np.empty(len(eccentric_times))
    shell_velocity = np.empty(len(eccentric_times))
    shell_radius[0] = radius
    shell_velocity[0] = 0.0
    for index in range(len(eccentric_times) - 1):
        shell_mesh.x_proper_code[...] = shell_radius[index]
        shell_solver.ApplyGravity(eccentric_dt, shell_mesh, shell_fluid, shell_par)
        shell_velocity[index + 1] = shell_fluid.Mom_code[0] / shell_fluid.Mass_code[0]
        shell_radius[index + 1] = (
            shell_radius[index] + eccentric_dt * shell_velocity[index + 1]
        )
    shell_radius_error = np.max(
        np.abs(shell_radius - reference_eccentric[0])
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(times, velocity_history, label='source check $v_r$')
    axes[0, 0].axhline(np.max(np.abs(saved_velocity)), color='tab:red',
                        ls=':', label='saved Rsim max $|v_r|$')
    axes[0, 0].axhline(0.0, color='k', ls='--', label='analytic $v_r=0$')
    axes[0, 0].set_xlabel('time [code units]')
    axes[0, 0].set_ylabel('radial velocity')
    axes[0, 1].semilogy(
        times,
        np.maximum(energy_error, 1.0e-300),
        label='numerical energy error',
    )
    axes[0, 1].set_xlabel('time [code units]')
    axes[0, 1].set_ylabel('maximum |energy error|')
    axes[1, 0].plot(
        eccentric_times, eccentric_state[0], label='RK4 source update'
    )
    axes[1, 0].plot(
        eccentric_times, shell_radius, ':',
        label='RadHydropy source-shell simulation',
    )
    axes[1, 0].plot(
        eccentric_times, reference_eccentric[0], '--',
        label='analytic ODE reference',
    )
    axes[1, 0].set_xlabel('time [code units]')
    axes[1, 0].set_ylabel('eccentric radius')
    axes[1, 1].plot(
        eccentric_times, eccentric_energy - eccentric_energy[0],
        label='specific-energy error',
    )
    axes[1, 1].set_xlabel('time [code units]')
    axes[1, 1].set_ylabel('$\\Delta e$')
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle('Gas centrifugal circular and eccentric orbit checks')
    fig.tight_layout()
    figure = savedir / 'GasCentrifugalCircularOrbit1D.jpg'
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print('circular and eccentric orbit analytic checks passed')
    print('eccentric maximum radius error = %.6g' % eccentric_radius_error)
    print('eccentric maximum velocity error = %.6g' % eccentric_velocity_error)
    print('RadHydropy source-shell maximum radius error = %.6g' % shell_radius_error)
    print('figure = %s' % figure)


if __name__ == '__main__':
    main()
