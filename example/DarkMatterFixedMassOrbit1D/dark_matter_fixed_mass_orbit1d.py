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


def effective_potential(radius, mass, angular_momentum, softening, g_code):
    return (
        -g_code * mass / (radius + softening)
        + 0.5 * angular_momentum**2 / radius**2
    )


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runtime = config['par']
    icparams = config['initial_condition']
    runparams = eu.legacy_example_parameters(config)
    runparams['timesim'] = runtime['simulation']['final_time']
    runparams['output_interval'] = runtime['timestep']['output_interval']
    runparams['savedir'] = runtime['output']['savedir']
    runparams['CodeUnits'] = runtime['units']['CodeUnits']
    code_units = et.load_units(runparams)
    shell = et.make_shell(icparams, code_units)
    g_code = (
        6.67430e-8 * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )
    central_mass = float(icparams['central_mass'])
    softening = float(icparams['softening'])
    angular_momentum = float(icparams['specific_angular_momentum'])
    initial_radius = float(icparams['initial_radius'])
    initial_velocity = float(icparams['initial_velocity'])
    energy = 0.5 * initial_velocity**2 + effective_potential(
        initial_radius, central_mass, angular_momentum, softening, g_code
    )

    def rhs(time, state):
        radius, velocity = state
        radius_safe = max(radius, np.finfo(float).tiny)
        acceleration = (
            -g_code * central_mass / (radius + softening)**2
            + angular_momentum**2 / radius_safe**3
        )
        return velocity, acceleration

    def event_radius_floor(time, state):
        return state[0] - 0.02

    event_radius_floor.terminal = True
    event_radius_floor.direction = -1
    reference = solve_ivp(
        rhs,
        (0.0, float(runparams['timesim'])),
        [initial_radius, initial_velocity],
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=float(runparams['output_interval']) / 4.0,
        events=event_radius_floor,
        dense_output=True,
    )

    numerical_time = [0.0]
    numerical_radius = [shell.radius[0]]
    numerical_energy = [shell.specific_energy()[0]]
    time = 0.0
    while time < reference.t[-1]:
        dt = min(
            float(runparams['output_interval']) / 4.0,
            reference.t[-1] - time,
        )
        time += shell.step(dt)
        numerical_time.append(time)
        numerical_radius.append(shell.radius[0])
        numerical_energy.append(shell.specific_energy()[0])

    numerical_time = np.asarray(numerical_time)
    numerical_radius = np.asarray(numerical_radius)
    numerical_energy = np.asarray(numerical_energy)
    reference_radius = reference.sol(numerical_time)[0]
    radius_error = np.max(np.abs(numerical_radius - reference_radius))
    energy_error = np.max(np.abs(numerical_energy - numerical_energy[0]))
    print('maximum radius error = %.6g code lengths' % radius_error)
    print('maximum shell-energy drift = %.6g code velocity squared' % energy_error)

    radius_pc = quantity_to_value(numerical_radius * code_units.length_unit, 'pc')
    reference_pc = quantity_to_value(reference_radius * code_units.length_unit, 'pc')
    time_myr = quantity_to_value(numerical_time * code_units.time_unit, 'Myr')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(time_myr, radius_pc, label='shell integrator')
    axes[0].plot(time_myr, reference_pc, '--', label='fixed-mass reference')
    axes[0].set_xlabel('time [Myr]')
    axes[0].set_ylabel('radius [pc]')
    axes[0].legend()
    axes[1].plot(time_myr, np.abs(numerical_energy - numerical_energy[0]))
    axes[1].set_xlabel('time [Myr]')
    axes[1].set_ylabel('absolute shell-energy drift')
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = Path(runparams['savedir']) / 'DarkMatterFixedMassOrbit1D.jpg'
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
