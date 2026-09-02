"""Check passive gas angular-momentum storage through a short hydro run."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join('/tmp', 'radhydropy-matplotlib'),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'passive_gas_angular_momentum1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runparams = eu.runtime_parameters(config)
    icparams = config['initial_condition']
    Path(runparams['output']['directory']).mkdir(parents=True, exist_ok=True)
    Path(runparams['output']['savedir']).mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(runparams)
    units = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    initial = et.Simwrap(icparams, units, runparams['mesh']['grid_cells'])
    rio.writehdf5(initial, runparams['simulation']['initial_condition_filename'])

    sim = Rsim(runparams)
    sim.RunAll(outputtime=0, mode='hydro')
    interior = slice(
        sim.par.mesh.ghost_cells,
        sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells,
    )

    initial_j = np.asarray(initial.fluid.specific_angular_momentum_code, dtype=float)
    final_j = np.asarray(sim.fluid.specific_angular_momentum_code[interior], dtype=float)
    final_j_from_conserved = np.asarray(
        sim.fluid.AngularMomentum_code[interior] / sim.fluid.Mass_code[interior],
        dtype=float,
    )
    if not np.allclose(
        final_j_from_conserved, final_j, rtol=1.0e-12, atol=1.0e-14
    ):
        raise RuntimeError('AngularMomentum/Mass does not reconstruct final j')
    dx = float(np.asarray(initial.mesh.boundary[1] - initial.mesh.boundary[0]))
    initial_total_j = np.sum(
        np.asarray(initial.fluid.rho_code, dtype=float) * initial_j * dx
    )
    final_total_j = np.sum(np.asarray(sim.fluid.AngularMomentum_code[interior], dtype=float))
    if not np.isclose(final_total_j, initial_total_j, rtol=1.0e-12, atol=1.0e-14):
        raise RuntimeError('periodic angular-momentum transport failed conservation')

    outputs = sorted(Path(runparams['output']['directory']).glob('Output_*.hdf5'))
    if not outputs:
        raise FileNotFoundError('no output snapshot was written')
    restart_par = type(
        'RestartPar', (), {
            'CodeUnits': units,
            'units': SimpleNamespace(CodeUnits=units),
            'simulation': SimpleNamespace(coordinate_system='cartesian'),
            'mesh': SimpleNamespace(grid_cells=runparams['mesh']['grid_cells'], ghost_cells=runparams['mesh']['ghost_cells']),
        }
    )()
    restart_mesh = type('RestartMesh', (), {})()
    restart_fluid = type('RestartFluid', (), {})()
    rio.readhdf5(restart_par, restart_mesh, restart_fluid, str(outputs[-1]))
    if not hasattr(restart_fluid, 'AngularMomentum_code'):
        raise RuntimeError('restart snapshot is missing AngularMomentum')
    restarted_j = np.asarray(
        (restart_fluid.AngularMomentum_code / restart_fluid.Mass_code)[interior],
        dtype=float,
    )
    restarted_specific_j = np.asarray(
        restart_fluid.specific_angular_momentum_code[interior], dtype=float
    )
    if not np.allclose(
        restarted_j, restarted_specific_j, rtol=1.0e-12, atol=1.0e-14
    ):
        raise RuntimeError('HDF5 restart changed J/M')

    radius = np.asarray(sim.mesh.coordinate[interior], dtype=float)
    figure = Path(runparams['output']['savedir']) / 'PassiveGasAngularMomentum1D.jpg'
    figure.parent.mkdir(parents=True, exist_ok=True)
    initial_rho_code = np.asarray(initial.fluid.rho_code, dtype=float)
    initial_vel_code = np.asarray(initial.fluid.vel_code, dtype=float)
    initial_temp_code = np.asarray(initial.fluid.temp_code, dtype=float)
    final_rho_code = np.asarray(sim.fluid.rho_code[interior], dtype=float)
    final_vel_code = np.asarray(sim.fluid.vel_code[interior], dtype=float)
    final_temp_code = np.asarray(sim.fluid.temp_code[interior], dtype=float)
    conserved_j = np.asarray(sim.fluid.AngularMomentum_code[interior], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    hydro_plots = (
        (axes[0, 0], initial_rho, final_rho, 'density [code units]'),
        (axes[0, 1], initial_vel, final_vel, 'radial velocity [code units]'),
        (axes[1, 0], initial_temp, final_temp, 'temperature [code units]'),
    )
    for axis, initial_values, final_values, ylabel in hydro_plots:
        axis.plot(radius, initial_values, '--', label='initial')
        axis.plot(radius, final_values, 'o', ms=3, label='final')
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    axes[1, 1].plot(radius, initial_j, '--', label='initial j')
    axes[1, 1].plot(radius, final_j, 'o', ms=3, label='final j = J/M')
    axes[1, 1].set_ylabel('angular momentum [code units]')
    axes[1, 1].grid(alpha=0.25)
    j_axis = axes[1, 1].twinx()
    j_axis.plot(radius, conserved_j, ':', lw=1.2, color='tab:red', label='stored J')
    j_axis.set_ylabel('extensive J [code mass·length$^2$/code time]', color='tab:red')
    j_axis.tick_params(axis='y', labelcolor='tab:red')
    for axis in axes[1, :]:
        axis.set_xlabel('cell coordinate [code length]')
    axes[0, 0].legend()
    handles, labels = axes[1, 1].get_legend_handles_labels()
    j_handles, j_labels = j_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + j_handles, labels + j_labels, fontsize='small')
    fig.suptitle('Passive gas angular-momentum storage check')
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    snapshot_times = []
    snapshot_total_j = []
    for output in outputs:
        snapshot_par = type(
            'SnapshotPar', (), {
                'CodeUnits': units,
                'units': SimpleNamespace(CodeUnits=units),
                'simulation': SimpleNamespace(coordinate_system='cartesian'),
                'mesh': SimpleNamespace(grid_cells=runparams['mesh']['grid_cells'], ghost_cells=runparams['mesh']['ghost_cells']),
            }
        )()
        snapshot_mesh = type('SnapshotMesh', (), {})()
        snapshot_fluid = type('SnapshotFluid', (), {})()
        rio.readhdf5(
            snapshot_par,
            snapshot_mesh,
            snapshot_fluid,
            str(output),
        )
        snapshot_times.append(float(np.asarray(snapshot_par.time)))
        snapshot_total_j.append(
            np.sum(
                np.asarray(snapshot_fluid.AngularMomentum_code[interior], dtype=float)
            )
        )
    snapshot_times = np.asarray(snapshot_times)
    # Older output writers may leave the header time at the initial value.
    # Do not plot duplicate timestamps as a vertical line; reconstruct the
    # configured output timeline in code units for that diagnostic only.
    if snapshot_times.size > 1 and np.allclose(snapshot_times, snapshot_times[0]):
        final_time = runparams['simulation']['final_time']
        final_time_code = float(
            final_time.to_value(units.time_unit)
            if hasattr(final_time, 'to_value')
            else final_time
        )
        snapshot_times = np.linspace(0.0, final_time_code, snapshot_times.size)
    snapshot_total_j = np.asarray(snapshot_total_j)
    relative_conservation_error = (
        snapshot_total_j - initial_total_j
    ) / max(abs(initial_total_j), np.finfo(float).tiny)
    conservation_figure = (
        Path(runparams['output']['savedir']) / 'PassiveGasAngularMomentum1D_conservation.jpg'
    )
    conservation_fig, conservation_axes = plt.subplots(1, 2, figsize=(10, 4))
    conservation_axes[0].plot(snapshot_times, snapshot_total_j, 'o-')
    conservation_axes[0].axhline(initial_total_j, color='k', ls='--', lw=1.0)
    conservation_axes[0].set_xlabel('time [code units]')
    conservation_axes[0].set_ylabel('total gas J [code units]')
    conservation_axes[1].plot(snapshot_times, relative_conservation_error, 'o-')
    conservation_axes[1].axhline(0.0, color='k', ls='--', lw=1.0)
    conservation_axes[1].set_xlabel('time [code units]')
    conservation_axes[1].set_ylabel('relative conservation error')
    for axis in conservation_axes:
        axis.grid(alpha=0.25)
    conservation_fig.suptitle('Gas angular-momentum conservation history')
    conservation_fig.tight_layout()
    conservation_fig.savefig(conservation_figure, dpi=180)
    plt.close(conservation_fig)
    print('active angular-momentum transport, conservation, and restart checks passed')
    print('figure = %s' % figure)
    print('conservation figure = %s' % conservation_figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
