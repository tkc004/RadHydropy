"""Source-only centrifugal work benchmark."""

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
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
import example_utils as eu


CONFIG = ROOT / 'gas_centrifugal_work_source1d.yaml'

def prepare_initial_condition(initial):
    boundary = np.asarray(initial.mesh.boundary_proper_code, dtype=float)
    initial.mesh.boundary_proper_code = boundary
    initial.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=initial.mesh.x_proper_code,
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


class InitialCondition(Rsim):
    def __init__(self, par_config, radius, density, velocity, temperature,
                 specific_j, code_units):
        super().__init__(par_config)
        self.par.mesh.grid_cells = 1
        self.par.mesh.ghost_cells = 0
        self.par.simulation.coordinate_system = 'spherical'
        self.par.simulation.time_proper_code = 0.0
        self.par.simulation.box_size = np.asarray([radius])
        self.mesh.boundary_proper_code = np.asarray([radius - 0.5, radius + 0.5])
        self.mesh.x_proper_code = np.asarray([radius])
        self.mesh.width_proper_code = np.asarray([1.0])
        self.mesh.area_proper_code = 4.0 * np.pi * self.mesh.boundary_proper_code[:-1] ** 2
        self.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(self.mesh.boundary_proper_code ** 3)
        self.fluid.rho_proper_code = np.asarray([density])
        self.fluid.vel_proper_code = np.asarray([velocity])
        self.fluid.temp_proper_code = np.asarray([temperature])
        self.fluid.mu = np.ones(1)
        self.fluid.specific_angular_momentum_code = np.asarray([specific_j])


def run_simulation(par, initial_condition, example_config):
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    radius = quantity_to_value(initial_condition['radius'], units.length_unit)
    initial = InitialCondition(
        par, radius,
        quantity_to_value(initial_condition['density'], units.density_unit),
        quantity_to_value(initial_condition['radial_velocity'], units.velocity_unit),
        quantity_to_value(example_config['temperature'], units.temperature_unit),
        quantity_to_value(
            initial_condition['specific_angular_momentum'],
            units.length_unit * units.velocity_unit,
        ), units,
    )
    prepare_initial_condition(initial)
    prepare_initial_condition(initial)
    ic_filename = ROOT / par['simulation']['initial_condition_filename']
    ic_filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, ic_filename)
    sim = Rsim(par)

    def source_backend(dt, mode='sources', **kwargs):
        sim.solver.ApplyGravity(dt, sim.mesh, sim.fluid, sim.par)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, par=sim.par)
        sim.fluid.time_proper_code += dt
        source_backend.record_source_state(dt)
        return {'dt': dt, 'hydro_steps': 0, 'source_steps': 1}

    rio.readhdf5(sim.par, sim.mesh, sim.fluid, str(ic_filename))
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    first = int(sim.par.mesh.ghost_cells)
    initial_mass = float(sim.fluid.Mass_code[first])
    initial_momentum = float(sim.fluid.Mom_code[first])
    initial_energy = float(sim.fluid.Energy_code[first])
    initial_internal = float(sim.fluid.InternalEnergy_code[first])
    source_times = [0.0]
    source_momenta = [initial_momentum]
    source_energies = [initial_energy]
    source_works = [0.0]

    def record_source_state(dt):
        source_times.append(float(sim.fluid.time_proper_code))
        source_momenta.append(float(sim.fluid.Mom_code[first]))
        source_energies.append(float(sim.fluid.Energy_code[first]))
        source_works.append(source_works[-1] + sim.solver.last_centrifugal_work)

    source_backend.record_source_state = record_source_state
    sim.Run(
        outputtime=0, mode='sources', step_backend=source_backend,
    )
    final_filename = ROOT / par['output']['directory'] / 'Output_final.hdf5'
    sim.fluid.SetTemperature()
    rio.writehdf5(sim, final_filename)
    final_par = sim.par
    final_mesh = SimpleNamespace()
    final_fluid = SimpleNamespace()
    rio.readhdf5(final_par, final_mesh, final_fluid, final_filename)
    return (
        sim, final_fluid, initial_mass, initial_momentum, initial_energy,
        initial_internal, np.asarray(source_times), np.asarray(source_momenta),
        np.asarray(source_energies), np.asarray(source_works),
    )


def main(config_filename=CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    initial_condition = config['initial_condition']
    example_config = config['example']
    (sim, saved, mass, initial_momentum, initial_energy,
     initial_internal, source_times, source_momenta, source_energies,
     source_works) = run_simulation(par, initial_condition, example_config)
    first = int(sim.par.mesh.ghost_cells)
    active = slice(first, first + int(sim.par.mesh.grid_cells))
    j = quantity_to_value(
        initial_condition['specific_angular_momentum'],
        units.length_unit * units.velocity_unit,
    )
    radius = float(sim.mesh.x_proper_code[first])
    acceleration = j**2 / radius**3
    # The generic HDF5 header stores the initial IC time for this non-cosmology
    # source driver; use the live Rsim clock for the exact source interval.
    final_time = float(sim.fluid.time_proper_code)
    time = source_times
    expected_momentum = initial_momentum + mass * acceleration * time
    # Centrifugal work is an internal transfer from rotational to radial
    # kinetic energy; the conserved total-energy field therefore stays fixed.
    expected_energy = np.full_like(time, initial_energy)
    # The reported centrifugal work is the transfer into radial kinetic
    # energy.  It is negative here because the inward radial flow is slowed;
    # total energy remains constant while rotational and radial reservoirs
    # exchange energy.
    expected_work = (
        0.5 * (expected_momentum**2 - initial_momentum**2) / mass
    )
    final_momentum = float(saved.Mass_code[active][0] * saved.vel_proper_code[active][0])
    final_energy = float(saved.Energy_code[active][0])
    final_j = float(saved.specific_angular_momentum_code[active][0])
    final_internal = float(saved.InternalEnergy_code[active][0]) if hasattr(
        saved, 'InternalEnergy') else initial_internal
    momentum_error = abs(final_momentum - expected_momentum[-1])
    energy_error = abs(final_energy - expected_energy[-1])
    if momentum_error > 1.0e-11 or energy_error > 1.0e-11:
        raise RuntimeError(
            'centrifugal source disagrees with exact work solution: '
            'momentum error=%g energy error=%g' % (momentum_error, energy_error)
        )
    if abs(final_j - j) > 1.0e-12:
        raise RuntimeError('centrifugal source changed signed specific angular momentum')
    if abs(final_internal - initial_internal) > 1.0e-11:
        raise RuntimeError('centrifugal work changed cold internal energy')

    figure = ROOT / par['output']['savedir'] / 'GasCentrifugalWorkSource1D.jpg'
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Keep the full history for validation, but sparsify plotted Rsim points
    # so the analytic reference remains visible.
    plotted = np.unique(np.r_[np.arange(0, len(source_times), 10),
                              len(source_times) - 1])
    axes[0].plot(time, expected_momentum, '--', label='analytic')
    axes[0].plot(source_times[plotted], source_momenta[plotted], ':o',
                 markersize=4, label='Rsim')
    axes[0].set_ylabel('radial momentum')
    axes[1].plot(time, expected_energy, '--', label='analytic')
    axes[1].plot(source_times[plotted], source_energies[plotted], ':o',
                 markersize=4, label='Rsim')
    axes[1].set_ylabel('total energy')
    axes[2].plot(time, expected_work, '--', label='analytic')
    axes[2].plot(source_times[plotted], source_works[plotted], ':o',
                 markersize=4, label='Rsim')
    axes[2].set_ylabel('work')
    for axis in axes:
        axis.set_xlabel('time')
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle('Centrifugal source work benchmark')
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print('centrifugal work source check passed')
    print('momentum error = %.6g' % momentum_error)
    print('energy error = %.6g' % energy_error)
    print('figure = %s' % figure)


if __name__ == '__main__':
    main()
