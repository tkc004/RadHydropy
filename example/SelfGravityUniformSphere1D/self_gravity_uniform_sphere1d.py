"""Analytic spherical self-gravity diagnostic for a uniform gas sphere."""

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

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
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
    'self_gravity_uniform_sphere1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runtime = config['par']
    icparams = config['initial_condition']
    eu.clean_previous_outputs(runtime['output'])
    code_units = CodeUnits.from_mapping(runtime['units']['CodeUnits'])

    config['_code_units'] = code_units
    initial_condition = et.build_initial_condition(config)
    rio.writehdf5(initial_condition, runtime['simulation']['initial_condition_filename'])

    runtime = {**runtime, 'simulation': {**runtime['simulation'], 'initial_condition_filename': runtime['simulation']['initial_condition_filename']}}
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

    numerical = sim.par.gravity.acceleration_on_mesh(
        sim.mesh,
        rho_code=sim.fluid.rho_code,
        par=sim.par,
    )
    interior = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    radius = sim.mesh.coordinate[interior]
    rho0 = icparams['rho0']
    radius_quantity = np.asarray(radius, dtype=float) * sim.par.CodeUnits.length_unit
    analytic = et.uniform_sphere_acceleration(radius_quantity, rho0)
    numerical_cgs = quantity_to_value(
        numerical[interior]
        * sim.par.CodeUnits.length_unit
        / sim.par.CodeUnits.time_unit**2,
        'cm/s**2',
    )
    analytic_cgs = quantity_to_value(analytic, 'cm/s**2')
    radius_pc = quantity_to_value(radius_quantity, 'pc')

    # The first physical cell contains the origin and is intentionally set to
    # zero by the spherical solver's symmetry convention.
    comparison = slice(1, None)
    relative_error = np.abs(
        (numerical_cgs[comparison] - analytic_cgs[comparison])
        / analytic_cgs[comparison]
    )
    max_relative_error = float(np.max(relative_error))
    if not np.all(np.isfinite(numerical_cgs)):
        raise RuntimeError('self-gravity acceleration contains non-finite values')
    if max_relative_error > 5.0e-3:
        raise RuntimeError(
            'uniform-sphere self-gravity relative error %.6g exceeds tolerance'
            % max_relative_error
        )

    figure_filename = os.path.join(
        runtime['output']['savedir'],
        'SelfGravityUniformSphere1D.jpg',
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(radius_pc, numerical_cgs, label='numerical')
    axes[0].plot(radius_pc, analytic_cgs, '--', label='analytic')
    axes[0].set_xlabel('radius [pc]')
    axes[0].set_ylabel('acceleration [cm/s$^2$]')
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(radius_pc[comparison], relative_error)
    axes[1].set_xlabel('radius [pc]')
    axes[1].set_ylabel('relative error')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)

    print('maximum relative error = %.6g' % max_relative_error)
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the uniform-sphere self-gravity diagnostic.'
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)



