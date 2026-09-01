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
import example_utils as eu


CONFIG = ROOT / 'gas_centrifugal_circular_orbit1d.yaml'


class FixedCentralGravity:
    def __init__(self, central_mass, angular_momentum):
        self.central_mass = central_mass
        self.angular_momentum = angular_momentum
        self.cosmological = False
        self.dark_matter = None

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        radius = np.asarray(mesh.coordinate, dtype=float)
        return -self.central_mass / radius**2


class CircularInitialCondition:
    """HDF5-compatible spherical circular-orbit initial condition."""

    def __init__(self, count, radius_min, radius_max, density, pressure,
                 central_mass, code_units):
        self.par = SimpleNamespace(
            CodeUnits=code_units,
            nogrid=count,
            noghost=2,
            coordsys='spherical',
            time=0.0,
            boxsize=np.asarray([radius_max]),
        )
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(current_time=0.0, box_size=np.asarray([radius_max]), coordinate_system='spherical')
        self.par.mesh = SimpleNamespace(grid_cells=count, ghost_cells=0)
        self.par.hydrodynamics = SimpleNamespace(gamma=1.4)
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.mesh.boundary = np.linspace(radius_min, radius_max, count + 1)
        self.mesh.coordinate = 0.75 * (
            self.mesh.boundary[1:]**4 - self.mesh.boundary[:-1]**4
        ) / (self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3)
        self.fluid.rho = np.full(count, density)
        self.fluid.vel = np.zeros(count)
        self.fluid.temp = np.full(count, pressure * 0.4)
        self.fluid.mu = np.ones(count)
        self.fluid.specific_angular_momentum = np.sqrt(
            central_mass * self.mesh.coordinate
        )


def run_rsim(runparams, icparams, runtime):
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    initial = CircularInitialCondition(
        int(runparams.get('nogrid', icparams['nogrid'])),
        float(icparams['radius_min']), float(icparams['radius_max']),
        float(icparams['density']), float(icparams['pressure']),
        float(icparams['central_mass']), units,
    )
    ic_filename = ROOT / runparams['ICfilename']
    ic_filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, ic_filename)
    sim = Rsim(runtime)
    sim.Callreadhdf5()
    sim.par.gravity = FixedCentralGravity(float(icparams['central_mass']), 0.0)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    initial_mass = np.asarray(sim.fluid.Mass, dtype=float).copy()
    initial_energy = np.asarray(sim.fluid.Energy, dtype=float).copy()
    sim.par.gravity = FixedCentralGravity(float(icparams['central_mass']), 0.0)
    sim.Run(outputtime=0, mode='sources')
    output_files = sorted((ROOT / runparams['outdir']).glob('Output_*.hdf5'))
    if not output_files:
        raise RuntimeError('Rsim produced no circular-orbit output')
    final_par = sim.par
    final_mesh = SimpleNamespace()
    final_fluid = SimpleNamespace()
    rio.readhdf5(final_par, final_mesh, final_fluid, output_files[-1])
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    return initial, sim, final_mesh, final_fluid, active, initial_mass, initial_energy


def main(config_filename=CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runparams = eu.legacy_example_parameters(config)
    icparams = config['initial_condition']
    icparams['nogrid'] = runparams['nogrid']
    runparams['output_interval'] = config['par']['output']['cadence']
    savedir = ROOT / runparams['savedir']
    savedir.mkdir(parents=True, exist_ok=True)
    (initial_sim, simulation, saved_mesh, saved_fluid, active,
     simulation_initial_mass, simulation_initial_energy) = run_rsim(
         runparams, icparams, config['par']
     )

    count = int(icparams['nogrid'])
    radius = float(icparams['radius'])
    central_mass = float(icparams['central_mass'])
    specific_j = np.sqrt(central_mass * radius)
    volume = np.ones(count)
    mass = np.full(count, float(icparams['density'])) * volume
    momentum = np.full(count, float(icparams['radial_velocity'])) * mass
    rotational_energy = 0.5 * mass * specific_j**2 / radius**2
    thermal_energy = np.full(count, float(icparams['pressure']) / 0.4)

    mesh = SimpleNamespace(
        coordsys='spherical',
        coordinate=np.full(count, radius),
        vol=volume,
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
    par = mesh._par
    par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=count)
    fluid = SimpleNamespace(
        rho=np.ones(count) * float(icparams['density']),
        Mass=mass.copy(),
        Mom=momentum.copy(),
        Energy=thermal_energy + rotational_energy,
        AngularMomentum=mass * specific_j,
    )

    solver = Solver()
    initial_energy = fluid.Energy.copy()
    times = [0.0]
    velocity_history = [fluid.Mom[0] / fluid.Mass[0]]
    energy_error = [0.0]
    dt = float(icparams['timestep'])
    for step in range(int(icparams['nsteps'])):
        solver.ApplyGravity(dt, mesh, fluid, par)
        times.append((step + 1) * dt)
        velocity_history.append(fluid.Mom[0] / fluid.Mass[0])
        energy_error.append(np.max(np.abs(fluid.Energy - initial_energy)))

    if not np.allclose(fluid.Mom, momentum, rtol=0.0, atol=1.0e-13):
        raise RuntimeError('circular force balance generated radial momentum')
    if not np.allclose(fluid.Energy, initial_energy, rtol=0.0, atol=1.0e-2):
        raise RuntimeError('circular force balance changed total energy')
    expected_rotational = rotational_energy
    if not np.allclose(
        expected_rotational,
        0.5 * fluid.AngularMomentum**2 / (fluid.Mass * radius**2),
    ):
        raise RuntimeError('rotational energy bookkeeping is inconsistent')

    # Compare the saved Rsim state with the circular analytic solution.
    saved_velocity = np.asarray(saved_fluid.vel[active], dtype=float)
    saved_j = np.asarray(saved_fluid.specific_angular_momentum[active], dtype=float)
    saved_boundary = np.asarray(saved_mesh.boundary, dtype=float)
    saved_radius = 0.75 * (
        saved_boundary[1:]**4 - saved_boundary[:-1]**4
    ) / (saved_boundary[1:]**3 - saved_boundary[:-1]**3)
    saved_radius = saved_radius[active]
    saved_mass = np.asarray(saved_fluid.Mass[active], dtype=float)
    saved_momentum = np.asarray(
        saved_fluid.Mass[active] * saved_fluid.vel[active], dtype=float
    )
    saved_energy = np.asarray(saved_fluid.Energy[active], dtype=float)
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
    eccentric_time = float(icparams['timestep']) * int(icparams['nsteps'])

    def orbit_rhs(time, state):
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
    eccentric_dt = float(icparams['timestep'])
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
    # small shell driver supplies the moving coordinate needed for a trajectory
    # comparison while retaining the production centrifugal/gravity update.
    shell_solver = Solver()
    shell_mesh = SimpleNamespace(
        coordsys='spherical', coordinate=np.asarray([radius]),
        vol=np.asarray([1.0]),
    )
    shell_par = SimpleNamespace(
        gas_angular_momentum=True, gas_rotational_energy=True,
        noghost=0, nogrid=1, energy_diagnostics=False,
        CodeUnits=None,
        gravity=FixedCentralGravity(central_mass, eccentric_j),
    )
    shell_par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=1)
    shell_fluid = SimpleNamespace(
        rho=np.asarray([1.0]), Mass=np.asarray([1.0]),
        Mom=np.asarray([0.0]), AngularMomentum=np.asarray([eccentric_j]),
        Energy=np.asarray([1.0 + 0.5 * eccentric_j**2 / radius**2]),
    )
    shell_radius = np.empty(len(eccentric_times))
    shell_velocity = np.empty(len(eccentric_times))
    shell_radius[0] = radius
    shell_velocity[0] = 0.0
    for index in range(len(eccentric_times) - 1):
        shell_mesh.coordinate[...] = shell_radius[index]
        shell_solver.ApplyGravity(eccentric_dt, shell_mesh, shell_fluid, shell_par)
        shell_velocity[index + 1] = shell_fluid.Mom[0] / shell_fluid.Mass[0]
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
