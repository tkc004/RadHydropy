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
from radhydropy.example_config import load_example_parameters
from radhydropy.units import quantity_to_value
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'gas_dark_matter_analytic_orbit1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_example_parameters(config_filename)
    code_units = et.load_units(runparams)
    shell = et.make_shell(icparams, code_units)
    g_code = (
        GRAVITATIONAL_CONSTANT_CGS * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )
    central_mass = float(icparams['central_dark_matter_mass'])
    gas_density = float(icparams['uniform_gas_density'])
    softening = float(icparams['softening'])
    angular_momentum = float(icparams['specific_angular_momentum'])
    initial_radius = float(icparams['initial_radius'])
    initial_velocity = float(icparams['initial_velocity'])

    def rhs(time, state):
        radius, velocity = state
        radius_safe = max(radius, np.finfo(float).tiny)
        enclosed = central_mass + 4.0 * np.pi / 3.0 * gas_density * radius**3
        acceleration = (
            -g_code * enclosed / (radius + softening)**2
            + angular_momentum**2 / radius_safe**3
        )
        return velocity, acceleration

    reference = solve_ivp(
        rhs,
        (0.0, float(runparams['timesim'])),
        [initial_radius, initial_velocity],
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=float(runparams['output_interval']) / 4.0,
        dense_output=True,
    )

    time = 0.0
    numerical_time = [time]
    numerical_radius = [shell.radius[0]]
    numerical_velocity = [shell.velocity[0]]
    while time < reference.t[-1]:
        dt = min(float(runparams['output_interval']) / 4.0, reference.t[-1] - time)
        time += shell.step(dt)
        numerical_time.append(time)
        numerical_radius.append(shell.radius[0])
        numerical_velocity.append(shell.velocity[0])

    numerical_time = np.asarray(numerical_time)
    numerical_radius = np.asarray(numerical_radius)
    numerical_velocity = np.asarray(numerical_velocity)
    reference_state = reference.sol(numerical_time)
    radius_error = np.max(np.abs(numerical_radius - reference_state[0]))
    velocity_error = np.max(np.abs(numerical_velocity - reference_state[1]))
    print('maximum radius error = %.6g code lengths' % radius_error)
    print('maximum velocity error = %.6g code velocities' % velocity_error)

    time_myr = quantity_to_value(numerical_time * code_units.time_unit, 'Myr')
    radius_pc = quantity_to_value(numerical_radius * code_units.length_unit, 'pc')
    reference_pc = quantity_to_value(reference_state[0] * code_units.length_unit, 'pc')
    velocity_kms = quantity_to_value(numerical_velocity * code_units.velocity_unit, 'km/s')
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
    figure = Path(runparams['savedir']) / 'GasDarkMatterAnalyticOrbit1D.jpg'
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
