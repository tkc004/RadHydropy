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

    initial_condition = config['initial_condition']
    eu.clean_previous_outputs(config)
    code_units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    initial = et.build_initial_condition(config)
    initial.write(config["par"]['simulation']['initial_condition_filename'])
    initial_state = initial.simulation
    density_cgs_g_cm3 = np.asarray(initial_state.fluid.rho_proper_code, dtype=float) * code_units.density_unit.to_value('g/cm**3')
    boundary_cgs_cm = np.asarray(initial_state.mesh.boundary_proper_code, dtype=float) * code_units.length_unit.to_value('cm')
    initial_gas_mass_cgs_g = float(np.sum(
        density_cgs_g_cm3 * (4.0 * np.pi / 3.0)
        * (boundary_cgs_cm[1:]**3 - boundary_cgs_cm[:-1]**3)
    ))
    dark_matter = et.make_dark_matter(config)
    initial_dm_mass_cgs_g = dark_matter.total_mass * code_units.mass_in_cgs
    initial_rho_proper_cgs_g_cm3 = quantity_to_value(
        initial_condition['rho_proper'], 'g/cm**3'
    )

    sim = rio.loadhdf5(
        config, config["par"]['simulation']['initial_condition_filename']
    )
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        selfgravity=True,
        dark_matter=dark_matter,
        code_units=sim.par.units.CodeUnits,
    )
    sim.par.dark_matter = dark_matter
    sim.Run(mode='hydro')

    interior = slice(
        sim.par.mesh.ghost_cells,
        sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells,
    )
    radius_proper_pc = quantity_to_value(
        np.asarray(sim.mesh.x_proper_code[interior]) * sim.par.units.CodeUnits.length_unit,
        'pc',
    )
    rho_gas_proper_cgs_g_cm3 = quantity_to_value(
        np.asarray(sim.fluid.rho_proper_code[interior]) * sim.par.units.CodeUnits.density_unit,
        'g/cm**3',
    )
    physical_boundaries = np.asarray(
        sim.mesh.boundary_proper_code[
            sim.par.mesh.ghost_cells:
            sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells + 1
        ],
        dtype=float,
    )
    gas_mass_proper_code = np.sum(
        np.asarray(sim.fluid.rho_proper_code[interior], dtype=float)
        * (4.0 * np.pi / 3.0)
        * (physical_boundaries[1:]**3 - physical_boundaries[:-1]**3)
    ) * sim.par.units.CodeUnits.mass_unit
    final_gas_mass_cgs_g = float(gas_mass_proper_code.to_value('g'))
    final_dm_mass_cgs_g = dark_matter.total_mass * sim.par.units.CodeUnits.mass_in_cgs
    gas_mass_error_dimensionless = abs(final_gas_mass_cgs_g - initial_gas_mass_cgs_g) / initial_gas_mass_cgs_g
    dm_mass_error_dimensionless = abs(final_dm_mass_cgs_g - initial_dm_mass_cgs_g) / initial_dm_mass_cgs_g
    if gas_mass_error_dimensionless > 1.0e-12 or dm_mass_error_dimensionless > 1.0e-12:
        raise RuntimeError(
            'mass conservation failed: gas %.6g, dark matter %.6g'
            % (gas_mass_error_dimensionless, dm_mass_error_dimensionless)
        )
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(radius_proper_pc, np.full_like(radius_proper_pc, initial_rho_proper_cgs_g_cm3), '--', label='initial')
    axis.plot(radius_proper_pc, rho_gas_proper_cgs_g_cm3, label='final')
    axis.set_xlabel('radius [pc]')
    axis.set_ylabel(r'gas density [g cm$^{-3}$]')
    axis.legend()
    axis.set_yscale('log')
    axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(config["par"]['output']['directory']) / 'GasDarkMatterShellCoupling1D.jpg'
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print('dark-matter shells = %d' % dark_matter.number_of_shells)
    print('total dark-matter mass = %.6g code masses' % dark_matter.total_mass)
    print('gas mass relative error = %.6g' % gas_mass_error_dimensionless)
    print('dark-matter mass relative error = %.6g' % dm_mass_error_dimensionless)
    print('figure = %s' % figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the coupled gas/dark-matter shell example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
