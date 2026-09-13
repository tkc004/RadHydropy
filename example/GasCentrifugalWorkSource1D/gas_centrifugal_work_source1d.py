"""Source-only centrifugal work benchmark."""

import os
import sys
from pathlib import Path

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
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
from radhydropy.initial_condition_writer import InitialConditionWriter
import example_utils as eu


CONFIG = ROOT / 'gas_centrifugal_work_source1d.yaml'

def build_initial_condition(config):
    initial_condition = config['initial_condition']
    par = config['par']
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    radius_proper_unyt = initial_condition['radius_proper']
    boundary_proper_unyt = radius_proper_unyt + np.array([-0.5, 0.5]) * units.length_unit
    writer = InitialConditionWriter(par_config=config['par'], code_units=units, ic_config=config["initial_condition"])
    writer.box_size = writer.radquantity(boundary_proper_unyt[-1])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(1) * initial_condition['rho_proper']
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.ones(1) * initial_condition['vel_proper']
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(1) * initial_condition['temperature_proper']
    )
    writer.simulation.fluid.mu = np.ones(1)
    writer.simulation.fluid.specific_angular_momentum_radarray = writer.radarray(
        np.atleast_1d(quantity_to_value(
            initial_condition['specific_angular_momentum'],
            units.length_unit * units.velocity_unit,
        )) * units.length_unit.units * units.velocity_unit.units,
        field_name='specific_angular_momentum_code',
        representation=None,
    )
    return writer


def run_simulation(config):
    par = config['par']
    initial_condition = config['initial_condition']
    initial = build_initial_condition(config)
    ic_filename = ROOT / par['simulation']['initial_condition_filename']
    ic_filename.parent.mkdir(parents=True, exist_ok=True)
    initial.write(ic_filename)
    sim = Rsim(config["par"])

    def source_backend(dt, mode='sources', **kwargs):
        sim.solver.ApplyGravity(dt, sim.mesh, sim.fluid, sim.par)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, par=sim.par)
        sim.fluid.time_proper_code += dt
        source_backend.record_source_state(dt)
        return {'dt': dt, 'hydro_steps': 0, 'source_steps': 1}

    sim = rio.loadhdf5(config, str(ic_filename))
    if hasattr(sim.fluid, 'specific_angular_momentum_code'):
        del sim.fluid.specific_angular_momentum_code
    if hasattr(sim.fluid, 'AngularMomentum_code'):
        del sim.fluid.AngularMomentum_code
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    ghost_cells = int(sim.par.mesh.ghost_cells)
    sim.fluid.specific_angular_momentum_code = as_named_array(np.concatenate((
        np.zeros(ghost_cells),
        np.asarray(initial.simulation.fluid.specific_angular_momentum_radarray, dtype=float),
        np.zeros(ghost_cells),
    )))
    sim.fluid.AngularMomentum_code = as_named_array(
        sim.fluid.Mass_code * sim.fluid.specific_angular_momentum_code
    )
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
    final_filename = sorted(
        (ROOT / par['output']['directory']).glob('Output_[0-9][0-9][0-9].hdf5')
    )[-1]
    final_sim = rio.loadhdf5(config, final_filename)
    return (
        sim, final_sim.fluid, initial_mass, initial_momentum, initial_energy,
        initial_internal, np.asarray(source_times), np.asarray(source_momenta),
        np.asarray(source_energies), np.asarray(source_works),
    )


def main(config_filename=CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    initial_condition = config['initial_condition']
    (sim, saved, mass, initial_momentum, initial_energy,
     initial_internal, source_times, source_momenta, source_energies,
     source_works) = run_simulation(config)
    first = int(sim.par.mesh.ghost_cells)
    active = slice(first, first + int(sim.par.mesh.grid_cells))
    j = quantity_to_value(
        initial_condition['specific_angular_momentum'],
        units.length_unit * units.velocity_unit,
    )
    radius_proper_code = float(sim.mesh.x_proper_code[first])
    acceleration_proper_code = j**2 / radius_proper_code**3
    # The generic HDF5 header stores the initial IC time for this non-cosmology
    # source driver; use the live Rsim clock for the exact source interval.
    final_time = float(sim.fluid.time_proper_code)
    time_proper_code = source_times
    expected_momentum = initial_momentum + mass * acceleration_proper_code * time_proper_code
    # Centrifugal work is an internal transfer from rotational to radial
    # kinetic energy; the conserved total-energy field therefore stays fixed.
    expected_energy = np.full_like(time_proper_code, initial_energy)
    # The reported centrifugal work is the transfer into radial kinetic
    # energy.  It is negative here because the inward radial flow is slowed;
    # total energy remains constant while rotational and radial reservoirs
    # exchange energy.
    expected_work = (
        0.5 * (expected_momentum**2 - initial_momentum**2) / mass
    )
    final_momentum = float(sim.fluid.Mom_code[first])
    final_energy = float(sim.fluid.Energy_code[first])
    final_j = float(sim.fluid.specific_angular_momentum_code[first])
    final_internal = float(sim.fluid.InternalEnergy_code[first]) if hasattr(
        sim.fluid, 'InternalEnergy_code') else initial_internal
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

    figure = ROOT / par['output']['directory'] / 'GasCentrifugalWorkSource1D.jpg'
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Keep the full history for validation, but sparsify plotted Rsim points
    # so the analytic reference remains visible.
    plotted = np.unique(np.r_[np.arange(0, len(source_times), 10),
                              len(source_times) - 1])
    axes[0].plot(time_proper_code, expected_momentum, '--', label='analytic')
    axes[0].plot(source_times[plotted], source_momenta[plotted], ':o',
                 markersize=4, label='Rsim')
    axes[0].set_ylabel('radial momentum')
    axes[1].plot(time_proper_code, expected_energy, '--', label='analytic')
    axes[1].plot(source_times[plotted], source_energies[plotted], ':o',
                 markersize=4, label='Rsim')
    axes[1].set_ylabel('total energy')
    axes[2].plot(time_proper_code, expected_work, '--', label='analytic')
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
