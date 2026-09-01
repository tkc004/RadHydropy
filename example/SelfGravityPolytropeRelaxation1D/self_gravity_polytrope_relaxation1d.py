"""Relax a perturbed self-gravitating n=1 polytrope toward equilibrium."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'self_gravity_polytrope_relaxation1d.yaml'
)


def _profile(sim, rho, pressure):
    interior = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    radius = np.asarray(sim.mesh.coordinate[interior], dtype=float)
    code = sim.par.CodeUnits
    radius_q = radius * code.length_unit
    gravity = sim.par.gravity.acceleration_on_mesh(
        sim.mesh, rho=rho, par=sim.par
    )[interior]
    gravity_cgs = quantity_to_value(
        gravity * code.length_unit / code.time_unit**2,
        'cm/s**2',
    )
    if hasattr(pressure, 'to_value'):
        pressure_cgs = quantity_to_value(pressure, 'erg/cm**3')
    else:
        pressure_cgs = quantity_to_value(
            np.asarray(pressure, dtype=float) * code.pressure_unit,
            'erg/cm**3',
        )
    rho_cgs = quantity_to_value(
        np.asarray(rho[interior], dtype=float) * code.density_unit,
        'g/cm**3',
    )
    residual = et.hydrostatic_residual(
        quantity_to_value(radius_q, 'cm'),
        rho_cgs,
        pressure_cgs,
        gravity_cgs,
    )
    return radius_q, gravity_cgs, residual


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runtime = dict(config['par'])
    runtime['relaxation_damping_time'] = config['example']['relaxation_damping_time']
    runparams = eu.legacy_example_parameters(config)
    runparams['relaxation_damping_time'] = config['example']['relaxation_damping_time']
    icparams = {**config['initial_condition'], 'nogrid': runtime['mesh']['grid_cells'], 'coordsys': 'spherical'}
    eu.clean_previous_outputs(runparams)
    code_units = CodeUnits.from_mapping(runparams['CodeUnits'])
    initial_condition = et.Simwrap(icparams, code_units)
    rio.writehdf5(initial_condition, runparams['ICfilename'])

    runtime['simulation'] = {**runtime['simulation'], 'initial_condition_filename': runparams['ICfilename']}
    sim = Rsim(runtime)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        selfgravity=True,
        externalgravity=False,
        code_units=sim.par.CodeUnits,
    )

    def damped_step(**kwargs):
        result = sim.Step(**kwargs)
        dt = float(np.asarray(result['dt'], dtype=float))
        damping_time = float(np.asarray(sim.par.relaxation_damping_time, dtype=float))
        sim.fluid.Mom *= np.exp(-dt / damping_time)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, verbose=0)
        sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
        return result

    sim.Run(mode='hydro', step_backend=damped_step)

    outputs = sorted(Path(runparams['outdir']).glob(runparams['outfileprefix'] + '_*.hdf5'))
    output = outputs[-1] if outputs else None
    if output is None:
        raise FileNotFoundError('no output snapshots were written')
    final = et.read_output(output, runparams)
    interior = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    k_poly = et.polytropic_constant(icparams['polytropic_radius'])
    radius = np.asarray(final.mesh.coordinate[interior], dtype=float) * sim.par.CodeUnits.length_unit
    rho_final = np.asarray(final.fluid.rho[interior], dtype=float) * sim.par.CodeUnits.density_unit
    pressure_final = np.asarray(final.fluid.pre[interior], dtype=float) * sim.par.CodeUnits.pressure_unit
    rho_expected = et.equilibrium_density(
        radius, icparams['central_density'], icparams['polytropic_radius']
    )
    radius_q, gravity_cgs, residual = _profile(sim, final.fluid.rho, pressure_final)
    rho_error = np.max(np.abs((rho_final - rho_expected) / rho_expected))
    residual_scale = np.max(np.abs(rho_final * gravity_cgs))
    residual_norm = np.max(np.abs(residual)) / max(residual_scale, np.finfo(float).tiny)
    print('maximum density relative error = %.6g' % rho_error)
    print('maximum normalized hydrostatic residual = %.6g' % residual_norm)

    radius_pc = quantity_to_value(radius_q, 'pc')
    rho_final_cgs = quantity_to_value(rho_final, 'g/cm**3')
    rho_expected_cgs = quantity_to_value(rho_expected, 'g/cm**3')
    velocity_cgs = quantity_to_value(
        np.asarray(final.fluid.vel[interior], dtype=float) * sim.par.CodeUnits.velocity_unit,
        'cm/s',
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(radius_pc, rho_final_cgs, label='final')
    axes[0].plot(radius_pc, rho_expected_cgs, '--', label='analytic equilibrium')
    axes[0].set_xlabel('radius [pc]')
    axes[0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
    axes[0].legend()
    axes[1].plot(radius_pc, velocity_cgs)
    axes[1].set_xlabel('radius [pc]')
    axes[1].set_ylabel('velocity [cm s$^{-1}$]')
    axes[2].plot(radius_pc, np.abs(residual) / max(residual_scale, np.finfo(float).tiny))
    axes[2].set_xlabel('radius [pc]')
    axes[2].set_ylabel('normalized hydrostatic residual')
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(runparams['savedir']) / 'SelfGravityPolytropeRelaxation1D.jpg'
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print('figure = %s' % figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the self-gravitating n=1 polytrope relaxation example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
