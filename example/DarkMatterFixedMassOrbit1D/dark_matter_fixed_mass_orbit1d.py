"""Compare a dark-matter shell orbit with its fixed-mass analytic ODE."""

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

from radhydropy.units import quantity_to_value
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'dark_matter_fixed_mass_orbit1d.yaml'
)


def effective_potential(radius_dimensionless, mass_dimensionless,
                        angular_momentum_dimensionless,
                        softening_dimensionless, g_code):
    return (
        -g_code * mass_dimensionless / (radius_dimensionless + softening_dimensionless)
        + 0.5 * angular_momentum_dimensionless**2 / radius_dimensionless**2
    )


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)

    initial_condition = config['initial_condition']
    code_units = et.code_units_from_config(config)
    shell = et.make_shell(config)
    g_code = (
        6.67430e-8 * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )
    central_mass_dimensionless = float(initial_condition['central_mass_dimensionless'])
    softening_dimensionless = float(initial_condition['softening_dimensionless'])
    angular_momentum_dimensionless = float(initial_condition['specific_angular_momentum_dimensionless'])
    initial_radius = float(initial_condition['radius_initial_orbit_dimensionless'])
    vel_proper_code = quantity_to_value(
        initial_condition['vel_proper'], code_units.velocity_unit
    )
    energy_dimensionless = 0.5 * vel_proper_code**2 + effective_potential(
        initial_radius, central_mass_dimensionless,
        angular_momentum_dimensionless, softening_dimensionless, g_code
    )

    def rhs(time_proper_code, state):
        radius_dimensionless, velocity_dimensionless = state
        radius_safe_dimensionless = max(radius_dimensionless, np.finfo(float).tiny)
        acceleration = (
            -g_code * central_mass_dimensionless
            / (radius_dimensionless + softening_dimensionless)**2
            + angular_momentum_dimensionless**2 / radius_safe_dimensionless**3
        )
        return velocity_dimensionless, acceleration

    def event_radius_floor(time_proper_code, state):
        return state[0] - 0.02

    event_radius_floor.terminal = True
    event_radius_floor.direction = -1
    final_time_proper_code = quantity_to_value(
        config["par"]['simulation']['final_time'], code_units.time_unit
    )
    output_interval_proper_code = quantity_to_value(
        config["par"]['timestep']['output_interval'], code_units.time_unit
    )
    reference = solve_ivp(
        rhs,
        (0.0, final_time_proper_code),
        [initial_radius, vel_proper_code],
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=output_interval_proper_code / 4.0,
        events=event_radius_floor,
        dense_output=True,
    )

    numerical_time = [0.0]
    numerical_radius_dimensionless = [shell.radius[0]]
    numerical_energy = [shell.specific_energy()[0]]
    time_dimensionless = 0.0
    while time_dimensionless < reference.t[-1]:
        dt = min(
            output_interval_proper_code / 4.0,
            reference.t[-1] - time_dimensionless,
        )
        time_dimensionless += shell.step(dt)
        numerical_time.append(time_dimensionless)
        numerical_radius_dimensionless.append(shell.radius[0])
        numerical_energy.append(shell.specific_energy()[0])

    numerical_time = np.asarray(numerical_time)
    numerical_radius_dimensionless = np.asarray(numerical_radius_dimensionless)
    numerical_energy = np.asarray(numerical_energy)
    reference_radius = reference.sol(numerical_time)[0]
    radius_error = np.max(np.abs(numerical_radius_dimensionless - reference_radius))
    energy_error = np.max(np.abs(numerical_energy - numerical_energy[0]))
    print('maximum radius error = %.6g code lengths' % radius_error)
    print('maximum shell-energy drift = %.6g code velocity squared' % energy_error)

    radius_proper_pc = quantity_to_value(numerical_radius_dimensionless * code_units.length_unit, 'pc')
    reference_pc = quantity_to_value(reference_radius * code_units.length_unit, 'pc')
    time_proper_Myr = quantity_to_value(numerical_time * code_units.time_unit, 'Myr')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(time_proper_Myr, radius_proper_pc, label='shell integrator')
    axes[0].plot(time_proper_Myr, reference_pc, '--', label='fixed-mass reference')
    axes[0].set_xlabel('time [Myr]')
    axes[0].set_ylabel('radius [pc]')
    axes[0].legend()
    axes[1].plot(time_proper_Myr, np.abs(numerical_energy - numerical_energy[0]))
    axes[1].set_xlabel('time [Myr]')
    axes[1].set_ylabel('absolute shell-energy drift')
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(config["par"]['output']['directory']) / 'DarkMatterFixedMassOrbit1D.jpg'
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print('figure = %s' % figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the fixed-mass dark-matter orbit benchmark.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
