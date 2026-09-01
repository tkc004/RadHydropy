"""Run a spherical gas simulation coupled to live dark-matter shells."""

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
    'gas_dark_matter_shell_coupling1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runparams = eu.runtime_parameters(config)
    icparams = config['initial_condition']
    eu.clean_previous_outputs(runparams)
    code_units = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    initial = et.Simwrap(icparams, code_units, runparams['mesh']['grid_cells'])
    rio.writehdf5(initial, runparams['simulation']['initial_condition_filename'])
    initial_density = quantity_to_value(
        initial.fluid.rho,
        'g/cm**3',
    )
    initial_gas_mass = np.sum(
        initial.fluid.rho
        * (4.0 * np.pi / 3.0)
        * (initial.mesh.boundary[1:]**3 - initial.mesh.boundary[:-1]**3)
    ).to_value('g')
    dark_matter = et.make_dark_matter(icparams, code_units)
    initial_dm_mass = dark_matter.total_mass * code_units.mass_in_cgs

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        selfgravity=True,
        dark_matter=dark_matter,
        code_units=sim.par.CodeUnits,
    )
    sim.par.dark_matter = dark_matter
    sim.Run(mode='hydro')

    interior = slice(
        sim.par.mesh.ghost_cells,
        sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells,
    )
    radius_pc = quantity_to_value(
        np.asarray(sim.mesh.coordinate[interior]) * sim.par.CodeUnits.length_unit,
        'pc',
    )
    density = quantity_to_value(
        np.asarray(sim.fluid.rho[interior]) * sim.par.CodeUnits.density_unit,
        'g/cm**3',
    )
    physical_boundaries = np.asarray(
        sim.mesh.boundary[
            sim.par.mesh.ghost_cells:
            sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells + 1
        ],
        dtype=float,
    )
    gas_mass = np.sum(
        np.asarray(sim.fluid.rho[interior], dtype=float)
        * (4.0 * np.pi / 3.0)
        * (physical_boundaries[1:]**3 - physical_boundaries[:-1]**3)
    ) * sim.par.CodeUnits.mass_unit
    final_gas_mass = float(gas_mass.to_value('g'))
    final_dm_mass = dark_matter.total_mass * sim.par.CodeUnits.mass_in_cgs
    gas_mass_error = abs(final_gas_mass - initial_gas_mass) / initial_gas_mass
    dm_mass_error = abs(final_dm_mass - initial_dm_mass) / initial_dm_mass
    if gas_mass_error > 1.0e-12 or dm_mass_error > 1.0e-12:
        raise RuntimeError(
            'mass conservation failed: gas %.6g, dark matter %.6g'
            % (gas_mass_error, dm_mass_error)
        )
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(radius_pc, initial_density, '--', label='initial')
    axis.plot(radius_pc, density, label='final')
    axis.set_xlabel('radius [pc]')
    axis.set_ylabel(r'gas density [g cm$^{-3}$]')
    axis.legend()
    axis.set_yscale('log')
    axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(runparams['output']['savedir']) / 'GasDarkMatterShellCoupling1D.jpg'
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print('dark-matter shells = %d' % dark_matter.number_of_shells)
    print('total dark-matter mass = %.6g code masses' % dark_matter.total_mass)
    print('gas mass relative error = %.6g' % gas_mass_error)
    print('dark-matter mass relative error = %.6g' % dm_mass_error)
    print('figure = %s' % figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the coupled gas/dark-matter shell example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
