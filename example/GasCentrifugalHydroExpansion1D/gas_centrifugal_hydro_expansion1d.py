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
sys.path.insert(0, str(PROJECT_ROOT / 'example'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
import example_utils as eu
from shell_remap import centrifugal_shell_reference


CONFIG = ROOT / 'gas_centrifugal_hydro_expansion1d.yaml'

def prepare_initial_condition(initial):
    boundary = np.asarray(initial.mesh.boundary, dtype=float)
    initial.mesh.boundary_proper_code = boundary
    initial.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=initial.mesh.coordinate,
        boundary=boundary, width=np.diff(boundary),
        area=4.0 * np.pi * boundary[:-1]**2,
        volume=4.0 * np.pi / 3.0 * (boundary[1:]**3 - boundary[:-1]**3),
    )
    initial.fluid.pre_proper_code = initial.fluid.temp_proper_code * 0.4
    initial.fluid.time_proper_code = 0.0
    initial.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    initial.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS, density=initial.fluid.rho_proper_code,
        velocity=initial.fluid.vel_proper_code, pressure=initial.fluid.pre_proper_code,
        temperature=initial.fluid.temp_proper_code, time=0.0,
        mu=initial.fluid.mu,
    )


def spherical_centers(boundary):
    return 0.75 * (
        boundary[1:]**4 - boundary[:-1]**4
    ) / (boundary[1:]**3 - boundary[:-1]**3)


class InitialCondition(Rsim):
    def __init__(self, par_config, count, radius_min, radius_max, density, temperature,
                 central_mass, rotation_factor, code_units):
        super().__init__(par_config)
        self.par.mesh.ghost_cells = 0
        boundary = np.linspace(radius_min, radius_max, count + 1)
        radius = spherical_centers(boundary)
        self.mesh.boundary_proper_code = boundary
        self.mesh.x_proper_code = radius
        self.mesh.width_proper_code = np.diff(boundary)
        self.mesh.area_proper_code = 4.0 * np.pi * boundary[:-1] ** 2
        self.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(boundary ** 3)
        self.fluid.rho_proper_code = np.full(count, density)
        self.fluid.vel_proper_code = np.zeros(count)
        self.fluid.temp_proper_code = np.full(count, temperature)
        self.fluid.mu = np.ones(count)
        self.fluid.specific_angular_momentum_code = rotation_factor * np.sqrt(central_mass * radius)
        self.mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS, coordinate=radius, boundary=boundary,
            width=self.mesh.width_proper_code, area=self.mesh.area_proper_code,
            volume=self.mesh.volume_proper_code,
        )


class FixedCentralGravity:
    cosmological = False
    dark_matter = None

    def __init__(self, central_mass):
        self.central_mass = central_mass

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        radius = np.abs(np.asarray(mesh.x_proper_code, dtype=float))
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
        return self.potential_on(mesh.x_proper_code)


def run_simulation(par, initial_condition, example_config):
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    count = int(par['mesh']['grid_cells'])
    initial = InitialCondition(
        par, count, float(initial_condition['radius_min']), float(initial_condition['radius_max']),
        float(initial_condition['density']), float(example_config['temperature']),
        float(initial_condition['central_mass']), float(initial_condition['rotation_factor']),
        units,
    )
    prepare_initial_condition(initial)
    prepare_initial_condition(initial)
    filename = ROOT / par['simulation']['initial_condition_filename']
    filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, filename)
    sim = Rsim(config["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, str(filename))
    sim.par.gravity = FixedCentralGravity(float(initial_condition['central_mass']))
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = FixedCentralGravity(float(initial_condition['central_mass']))
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    initial_mass = np.asarray(sim.fluid.Mass_code[active], dtype=float).copy()
    initial_energy = np.asarray(sim.fluid.Energy_code[active], dtype=float).copy()
    initial_radius = np.asarray(sim.mesh.x_proper_code[active], dtype=float).copy()
    sim.Run(outputtime=0, mode='hydro')
    final_filename = ROOT / par['output']['directory'] / 'Output_final.hdf5'
    sim.fluid.SetTemperature()
    rio.writehdf5(sim, final_filename)
    final_par = sim.par
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
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    initial_condition = config['initial_condition']
    example_config = config['example']
    (sim, saved_mesh, saved, initial_mass, initial_energy,
     initial_radius, cumulative_gravity_work, cumulative_potential_change,
     cumulative_potential_flux) = run_simulation(par, initial_condition, example_config)
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    radius = np.asarray(sim.mesh.x_proper_code[active], dtype=float)
    central_mass = float(initial_condition['central_mass'])
    rotation_factor = float(initial_condition['rotation_factor'])
    final_time = float(sim.fluid.time_proper_code)
    saved_boundary = np.asarray(saved_mesh.boundary_proper_code, dtype=float)
    source_boundary = saved_boundary[sim.par.noghost:sim.par.noghost + sim.par.nogrid + 1]
    saved_radius = spherical_centers(saved_boundary)[active]
    reference = centrifugal_shell_reference(
        source_boundary,
        source_boundary,
        final_time,
        float(initial_condition['density']),
        central_mass,
        rotation_factor,
        samples_per_cell=int(initial_condition.get('reference_samples_per_cell', 32)),
    )
    ode_velocity = reference['velocity']
    ode_j = reference['specific_angular_momentum']
    saved_velocity = np.asarray(saved.vel_proper_code[active], dtype=float)
    saved_j = np.asarray(saved.specific_angular_momentum_code[active], dtype=float)
    saved_mass = np.asarray(saved.Mass_code[active], dtype=float)
    saved_energy = np.asarray(saved.Energy_code[active], dtype=float)
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
    ode_total_energy = np.sum(reference['energy'])
    energy_error = float(abs(saved_total_energy - ode_total_energy))
    energy_scale = max(abs(float(ode_total_energy)), 1.0e-12)
    if energy_error / energy_scale > 2.0e-3:
        raise RuntimeError(
            'hydro expansion total-energy audit failed: relative error %.6g'
            % (energy_error / energy_scale)
        )

    density = np.asarray(saved.rho_proper_code[active], dtype=float)
    temperature = np.asarray(saved.temp_proper_code[active], dtype=float)
    mu = np.asarray(saved.mu[active], dtype=float)
    pressure = np.asarray(sim.fluid.eos.pressure(density, temperature, mu), dtype=float)
    pressure_ratio = np.divide(
        pressure / np.maximum(density, np.finfo(float).tiny),
        central_mass / np.maximum(saved_radius, np.finfo(float).tiny),
    )
    figure = ROOT / par['output']['savedir'] / 'GasCentrifugalHydroExpansion1D.jpg'
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
