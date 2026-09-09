"""Analytic time-evolution benchmark for a shell in a gas+DM background."""

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
from scipy.integrate import solve_ivp

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
from example import example_utils as eu
from radhydropy.units import quantity_to_value
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'gas_dark_matter_analytic_orbit1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)

    initial_condition = config['initial_condition']
    example = config['example']
    code_units = et.code_units_from_config(config)
    shell = et.make_shell(config)
    g_code = (
        GRAVITATIONAL_CONSTANT_CGS * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )
    central_mass = float(initial_condition['central_dark_matter_mass'])
    gas_density = float(initial_condition['uniform_gas_density'])
    softening_length_code = float(initial_condition['softening'])
    specific_angular_momentum_code = float(initial_condition['specific_angular_momentum'])
    initial_radius_code = float(initial_condition['radius_initial_orbit_dimensionless'])
    initial_velocity_code = quantity_to_value(
        initial_condition['vel_proper'], code_units.velocity_unit
    )

    def rhs(time_proper_code, state):
        radius_code, velocity_code = state
        radius_safe_code = max(radius_code, np.finfo(float).tiny)
        enclosed_mass_code = central_mass + 4.0 * np.pi / 3.0 * gas_density * radius_code**3
        acceleration = (
            -g_code * enclosed_mass_code / (radius_code + softening_length_code)**2
            + specific_angular_momentum_code**2 / (radius_safe_code + softening_length_code)**3
        )
        return velocity_code, acceleration

    reference = solve_ivp(
        rhs,
        (0.0, float(config["par"]['simulation']['final_time'])),
        [initial_radius_code, initial_velocity_code],
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=float(example['output_interval']) / 4.0,
        dense_output=True,
    )

    time_code = 0.0
    numerical_time_code = [time_code]
    numerical_radius_code = [shell.radius[0]]
    numerical_velocity_code = [shell.velocity[0]]
    while time_code < reference.t[-1]:
        timestep_code = min(float(example['output_interval']) / 4.0, reference.t[-1] - time_code)
        time_code += shell.step(timestep_code)
        numerical_time_code.append(time_code)
        numerical_radius_code.append(shell.radius[0])
        numerical_velocity_code.append(shell.velocity[0])

    numerical_time_code = np.asarray(numerical_time_code)
    numerical_radius_code = np.asarray(numerical_radius_code)
    numerical_velocity_code = np.asarray(numerical_velocity_code)
    reference_state = reference.sol(numerical_time_code)
    radius_error = np.max(np.abs(numerical_radius_code - reference_state[0]))
    velocity_error = np.max(np.abs(numerical_velocity_code - reference_state[1]))
    print('maximum radius error = %.6g code lengths' % radius_error)
    print('maximum velocity error = %.6g code velocities' % velocity_error)

    time_myr = quantity_to_value(numerical_time_code * code_units.time_unit, 'Myr')
    radius_pc = quantity_to_value(numerical_radius_code * code_units.length_unit, 'pc')
    reference_pc = quantity_to_value(reference_state[0] * code_units.length_unit, 'pc')
    velocity_kms = quantity_to_value(numerical_velocity_code * code_units.velocity_unit, 'km/s')
    reference_kms = quantity_to_value(reference_state[1] * code_units.velocity_unit, 'km/s')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(time_myr, radius_pc, label='shell integrator')
    axes[0].plot(time_myr, reference_pc, '--', label='analytic ODE reference')
    axes[0].set_xlabel('time [Myr]')
    axes[0].set_ylabel('radius [pc]')
    axes[0].legend()
    axes[1].plot(time_myr, velocity_kms, label='shell integrator')
    axes[1].plot(time_myr, reference_kms, '--', label='analytic ODE reference')
    axes[1].set_xlabel('time [Myr]')
    axes[1].set_ylabel('radial velocity [km/s]')
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(config["par"]['output']['directory']) / 'GasDarkMatterAnalyticOrbit1D.jpg'
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print('figure = %s' % figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the analytic gas+DM orbit benchmark.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
