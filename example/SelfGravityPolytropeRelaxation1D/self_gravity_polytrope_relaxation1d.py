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
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'self_gravity_polytrope_relaxation1d.yaml'
)


class PolytropeSolver(Solver):
    """Keep reflecting ghost temperatures consistent with the active state."""

    def SetBoundary(self, mesh, fluid, par):
        super().SetBoundary(mesh, fluid, par)
        first = int(par.mesh.ghost_cells)
        last = first + int(par.mesh.grid_cells)
        fluid.temp_proper_code[:first] = fluid.temp_proper_code[first]
        fluid.temp_proper_code[last:] = fluid.temp_proper_code[last - 1]
        fluid.SetPressure()


def _profile(sim, rho_proper_code, pre_proper_code):
    interior = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    radius_proper_code = np.asarray(sim.mesh.x_proper_code[interior], dtype=float)
    code = sim.par.units.CodeUnits
    radius_proper_cgs_cm_unyt = radius_proper_code * code.length_unit
    gravity = sim.par.gravity.acceleration_on_mesh(
        sim.mesh, rho_proper_code, sim.par
    )[interior]
    gravity_cgs = quantity_to_value(
        gravity * code.length_unit / code.time_unit**2,
        'cm/s**2',
    )
    if hasattr(pre_proper_code, 'to_value'):
        pre_proper_cgs_erg_cm3 = quantity_to_value(pre_proper_code, 'erg/cm**3')
    else:
        pre_proper_cgs_erg_cm3 = quantity_to_value(
            np.asarray(pre_proper_code, dtype=float) * code.pressure_unit,
            'erg/cm**3',
        )
    rho_proper_cgs_g_cm3 = quantity_to_value(
        np.asarray(rho_proper_code[interior], dtype=float) * code.density_unit,
        'g/cm**3',
    )
    residual = et.hydrostatic_residual(
        quantity_to_value(radius_proper_cgs_cm_unyt, 'cm'),
        rho_proper_cgs_g_cm3,
        pre_proper_cgs_erg_cm3,
        gravity_cgs,
    )
    return radius_proper_cgs_cm_unyt, gravity_cgs, residual


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    initial_mapping = config['initial_condition']
    eu.clean_previous_outputs(config)
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    config['_code_units'] = code_units
    initial_condition = et.build_initial_condition(config)
    initial_filename = Path(config['par']['simulation']['initial_condition_filename'])
    rio.writehdf5(initial_condition, initial_filename)

    config['par']['simulation']['initial_condition_filename'] = str(initial_filename)
    sim = Rsim(config['par'])
    sim.solver = PolytropeSolver()
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    # The IC header is authoritative for solver state, but it does not carry
    # this example-local relaxation control.  Restore it after the header
    # read and convert it to the active code-time representation before the
    # first damped step.
    sim.par.relaxation_damping_time = config['example']['relaxation_damping_time'].to_value(
        sim.par.units.CodeUnits.time_unit
    )
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    # The reflecting boundary fills conserved variables but leaves primitive
    # temperature unset in the ghost cells.  Copy the adjacent equilibrium
    # temperature before the timestep estimate uses the ghost zones.
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    sim.fluid.temp_proper_code[:first] = sim.fluid.temp_proper_code[first]
    sim.fluid.temp_proper_code[last:] = sim.fluid.temp_proper_code[last - 1]
    sim.fluid.SetPressure()
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    sim.par.gravity = Gravity(
        selfgravity=True,
        externalgravity=False,
        code_units=sim.par.units.CodeUnits,
    )

    def damped_step(**kwargs):
        result = sim.Step(**kwargs)
        dt_proper_code = float(np.asarray(result['dt'], dtype=float))
        damping_time_proper_code = float(sim.par.relaxation_damping_time)
        sim.fluid.Mom_code *= np.exp(-dt_proper_code / damping_time_proper_code)
        sim.solver.SetPrimitive(sim.mesh, sim.fluid, verbose=0)
        sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
        return result

    sim.Run(mode='hydro', step_backend=damped_step)

    outputs = sorted(
        Path(config['par']['output']['directory']).glob(
            config['par']['output']['filename_prefix'] + '_*.hdf5'
        )
    )
    output = outputs[-1] if outputs else None
    if output is None:
        raise FileNotFoundError('no output snapshots were written')
    final = et.read_output(output, config)
    interior = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    k_poly_cgs = et.polytropic_constant(initial_mapping['radius_polytropic_proper'])
    radius_proper_cgs_cm_unyt = np.asarray(final.mesh.x_proper_code[interior], dtype=float) * sim.par.units.CodeUnits.length_unit
    rho_final = np.asarray(final.fluid.rho_proper_code[interior], dtype=float) * sim.par.units.CodeUnits.density_unit
    pressure_final = np.asarray(final.fluid.pre_proper_code[interior], dtype=float) * sim.par.units.CodeUnits.pressure_unit
    rho_expected_proper_cgs_g_cm3_unyt = et.equilibrium_density(
        radius_proper_cgs_cm_unyt,
        initial_mapping['rho_central_proper'],
        initial_mapping['radius_polytropic_proper'],
    )
    radius_proper_cgs_cm_unyt, gravity_cgs, residual = _profile(sim, final.fluid.rho_proper_code, pressure_final)
    rho_error = np.max(np.abs((rho_final - rho_expected_proper_cgs_g_cm3_unyt) / rho_expected_proper_cgs_g_cm3_unyt))
    residual_scale = np.max(np.abs(rho_final * gravity_cgs))
    residual_norm = np.max(np.abs(residual)) / max(residual_scale, np.finfo(float).tiny)
    print('maximum density relative error = %.6g' % rho_error)
    print('maximum normalized hydrostatic residual = %.6g' % residual_norm)

    radius_proper_pc = quantity_to_value(radius_proper_cgs_cm_unyt, 'pc')
    rho_final_cgs = quantity_to_value(rho_final, 'g/cm**3')
    rho_expected_proper_cgs_g_cm3 = quantity_to_value(rho_expected_proper_cgs_g_cm3_unyt, 'g/cm**3')
    vel_proper_cgs_cm_s = quantity_to_value(
        np.asarray(final.fluid.vel_proper_code[interior], dtype=float) * sim.par.units.CodeUnits.velocity_unit,
        'cm/s',
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(radius_proper_pc, rho_final_cgs, label='final')
    axes[0].plot(radius_proper_pc, rho_expected_proper_cgs_g_cm3, '--', label='analytic equilibrium')
    axes[0].set_xlabel('radius [pc]')
    axes[0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
    axes[0].legend()
    axes[1].plot(radius_proper_pc, vel_proper_cgs_cm_s)
    axes[1].set_xlabel('radius [pc]')
    axes[1].set_ylabel('velocity [cm s$^{-1}$]')
    axes[2].plot(radius_proper_pc, np.abs(residual) / max(residual_scale, np.finfo(float).tiny))
    axes[2].set_xlabel('radius [pc]')
    axes[2].set_ylabel('normalized hydrostatic residual')
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(config['par']['output']['directory']) / 'SelfGravityPolytropeRelaxation1D.jpg'
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
