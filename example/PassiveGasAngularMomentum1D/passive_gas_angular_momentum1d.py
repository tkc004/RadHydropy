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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'passive_gas_angular_momentum1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_example_parameters(config_filename)
    Path(runparams['outdir']).mkdir(parents=True, exist_ok=True)
    Path(runparams['savedir']).mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(runparams)
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    initial = et.Simwrap(icparams, units)
    rio.writehdf5(initial, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.RunAll(outputtime=0, mode='hydro')
    interior = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)

    initial_j = np.asarray(initial.fluid.specific_angular_momentum, dtype=float)
    final_j = np.asarray(sim.fluid.specific_angular_momentum[interior], dtype=float)
    final_j_from_conserved = np.asarray(
        sim.fluid.AngularMomentum[interior] / sim.fluid.Mass[interior],
        dtype=float,
    )
    if not np.allclose(
        final_j_from_conserved, final_j, rtol=1.0e-12, atol=1.0e-14
    ):
        raise RuntimeError('AngularMomentum/Mass does not reconstruct final j')
    dx = float(np.asarray(initial.mesh.boundary[1] - initial.mesh.boundary[0]))
    initial_total_j = np.sum(
        np.asarray(initial.fluid.rho, dtype=float) * initial_j * dx
    )
    final_total_j = np.sum(np.asarray(sim.fluid.AngularMomentum[interior], dtype=float))
    if not np.isclose(final_total_j, initial_total_j, rtol=1.0e-12, atol=1.0e-14):
        raise RuntimeError('periodic angular-momentum transport failed conservation')

    outputs = sorted(Path(runparams['outdir']).glob('Output_*.hdf5'))
    if not outputs:
        raise FileNotFoundError('no output snapshot was written')
    restart_par = type('RestartPar', (), {'coordsys': 'cartesian', 'CodeUnits': units})()
    restart_mesh = type('RestartMesh', (), {})()
    restart_fluid = type('RestartFluid', (), {})()
    rio.readhdf5(restart_par, restart_mesh, restart_fluid, str(outputs[-1]))
    if not hasattr(restart_fluid, 'AngularMomentum'):
        raise RuntimeError('restart snapshot is missing AngularMomentum')
    restarted_j = np.asarray(
        (restart_fluid.AngularMomentum / restart_fluid.Mass)[interior],
        dtype=float,
    )
    restarted_specific_j = np.asarray(
        restart_fluid.specific_angular_momentum[interior], dtype=float
    )
    if not np.allclose(
        restarted_j, restarted_specific_j, rtol=1.0e-12, atol=1.0e-14
    ):
        raise RuntimeError('HDF5 restart changed J/M')

    radius = np.asarray(sim.mesh.coordinate[interior], dtype=float)
    figure = Path(runparams['savedir']) / 'PassiveGasAngularMomentum1D.jpg'
    figure.parent.mkdir(parents=True, exist_ok=True)
    initial_rho = np.asarray(initial.fluid.rho, dtype=float)
    initial_vel = np.asarray(initial.fluid.vel, dtype=float)
    initial_temp = np.asarray(initial.fluid.temp, dtype=float)
    final_rho = np.asarray(sim.fluid.rho[interior], dtype=float)
    final_vel = np.asarray(sim.fluid.vel[interior], dtype=float)
    final_temp = np.asarray(sim.fluid.temp[interior], dtype=float)
    conserved_j = np.asarray(sim.fluid.AngularMomentum[interior], dtype=float)

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
            'SnapshotPar', (), {'coordsys': 'cartesian', 'CodeUnits': units}
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
                np.asarray(snapshot_fluid.AngularMomentum[interior], dtype=float)
            )
        )
    snapshot_times = np.asarray(snapshot_times)
    snapshot_total_j = np.asarray(snapshot_total_j)
    relative_conservation_error = (
        snapshot_total_j - initial_total_j
    ) / max(abs(initial_total_j), np.finfo(float).tiny)
    conservation_figure = (
        Path(runparams['savedir']) / 'PassiveGasAngularMomentum1D_conservation.jpg'
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
