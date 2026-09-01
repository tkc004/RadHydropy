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
from radhydropy.units import CodeUnits
import example_utils as eu


CONFIG = ROOT / 'gas_centrifugal_work_source1d.yaml'


class InitialCondition:
    def __init__(self, radius, density, velocity, temperature, specific_j,
                 code_units):
        self.par = SimpleNamespace(
            CodeUnits=code_units, nogrid=1, noghost=2, coordsys='spherical',
            time=0.0, boxsize=np.asarray([radius]),
        )
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(current_time=0.0, box_size=np.asarray([radius]), coordinate_system='spherical')
        self.par.mesh = SimpleNamespace(grid_cells=1, ghost_cells=0)
        self.par.hydrodynamics = SimpleNamespace(gamma=1.4)
        self.mesh = SimpleNamespace(
            boundary=np.asarray([radius - 0.5, radius + 0.5]),
            coordinate=np.asarray([radius]),
        )
        self.fluid = SimpleNamespace(
            rho=np.asarray([density]), vel=np.asarray([velocity]),
            temp=np.asarray([temperature]), mu=np.ones(1),
            specific_angular_momentum=np.asarray([specific_j]),
        )


def run_simulation(runparams, icparams, runtime):
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    radius = float(icparams['radius'])
    initial = InitialCondition(
        radius, float(icparams['density']), float(icparams['radial_velocity']),
        float(runparams['temperature']),
        float(icparams['specific_angular_momentum']), units,
    )
    ic_filename = ROOT / runparams['ICfilename']
    ic_filename.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, ic_filename)
    sim = Rsim(runtime)

    def source_backend(dt, mode='sources', **kwargs):
        sim.solver.ApplyGravity(dt, sim.mesh, sim.fluid, sim.par)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, par=sim.par)
        sim.fluid.time += dt
        source_backend.record_source_state(dt)
        return {'dt': dt, 'hydro_steps': 0, 'source_steps': 1}

    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    initial_mass = float(sim.fluid.Mass[sim.par.noghost])
    initial_momentum = float(sim.fluid.Mom[sim.par.noghost])
    initial_energy = float(sim.fluid.Energy[sim.par.noghost])
    initial_internal = float(sim.fluid.InternalEnergy[sim.par.noghost])
    source_times = [0.0]
    source_momenta = [initial_momentum]
    source_energies = [initial_energy]
    source_works = [0.0]

    def record_source_state(dt):
        source_times.append(float(sim.fluid.time))
        source_momenta.append(float(sim.fluid.Mom[sim.par.noghost]))
        source_energies.append(float(sim.fluid.Energy[sim.par.noghost]))
        source_works.append(source_works[-1] + sim.solver.last_centrifugal_work)

    source_backend.record_source_state = record_source_state
    sim.Run(
        outputtime=0, mode='sources', step_backend=source_backend,
    )
    final_filename = ROOT / runparams['outdir'] / 'Output_final.hdf5'
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
    runparams = eu.legacy_example_parameters(config)
    runparams.update(config.get('example', {}))
    icparams = config['initial_condition']
    icparams['nogrid'] = runparams['nogrid']
    runparams['temperature'] = config['example']['temperature']
    icparams['timestep'] = config['example']['timestep']
    (sim, saved, mass, initial_momentum, initial_energy,
     initial_internal, source_times, source_momenta, source_energies,
     source_works) = run_simulation(runparams, icparams, config['par'])
    active = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    j = float(icparams['specific_angular_momentum'])
    radius = float(sim.mesh.coordinate[sim.par.noghost])
    acceleration = j**2 / radius**3
    # The generic HDF5 header stores the initial IC time for this non-cosmology
    # source driver; use the live Rsim clock for the exact source interval.
    final_time = float(sim.fluid.time)
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
    final_momentum = float(saved.Mass[active][0] * saved.vel[active][0])
    final_energy = float(saved.Energy[active][0])
    final_j = float(saved.specific_angular_momentum[active][0])
    final_internal = float(saved.InternalEnergy[active][0]) if hasattr(
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

    figure = ROOT / runparams['savedir'] / 'GasCentrifugalWorkSource1D.jpg'
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
