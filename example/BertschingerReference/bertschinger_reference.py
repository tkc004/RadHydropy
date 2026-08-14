"""Generate a collisionless Bertschinger (1985) similarity reference profile."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.cosmology import EinsteinDeSitter
import tools as et
from bertschinger_ode import plot_xi_lambda, solve_eq41_self_similar
from radhydropy.example_config import load_example_parameters


DEFAULT_CONFIG = Path(__file__).with_name('bertschinger_reference.yaml')


def main(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_example_parameters(config_filename, Path.cwd().resolve())
    units = et.load_units(runparams)
    cosmology = EinsteinDeSitter.from_code_units(
        units, t_ref=float(runparams['cosmology_t_ref']),
        a_ref=float(runparams['cosmology_a_ref']),
    )
    shells, delta_mass = et.make_scale_free_shells(icparams, units, cosmology)
    initial_time = float(icparams['initial_cosmic_time'])
    final_time = float(runparams['final_cosmic_time'])
    tau = float(cosmology.supercomoving_time(initial_time))
    final_tau = float(cosmology.supercomoving_time(final_time))
    history_time = [initial_time]
    history_rta = []
    timestep = float(runparams['supercomoving_timestep'])

    while tau < final_tau:
        dt = min(timestep, final_tau - tau)
        time_start = float(cosmology.cosmic_time_from_supercomoving(tau))
        time_end = float(cosmology.cosmic_time_from_supercomoving(tau + dt))
        a_start = float(cosmology.scale_factor(time_start))
        a_end = float(cosmology.scale_factor(time_end))
        rho_start = float(cosmology.background_density(time_start)) * a_start**3
        background = 4.0 * np.pi / 3.0 * rho_start * shells.radius**3
        shells.step(
            dt,
            crossing_safety_factor=float(runparams['crossing_safety_factor']),
            background_enclosed_mass=background,
            scale_factor=a_start,
            scale_factor_end=a_end,
            cosmological=True,
            include_shell_mass_with_fixed=True,
        )
        tau += dt
        history_time.append(time_end)

    profiles = et.similarity_profiles(shells, final_time, cosmology,
                                      bins=int(runparams['profile_bins']))
    ode_solution = solve_eq41_self_similar(
        xi_end=float(runparams['ode_xi_end']),
        points=int(runparams['ode_points']),
        similarity_exponent=float(runparams['ode_similarity_exponent']),
    )
    if not np.all(np.isfinite(profiles['density'])):
        raise RuntimeError('similarity density profile contains non-finite values')
    if not np.all(np.diff(shells.radius) >= 0.0):
        raise RuntimeError('shells are not sorted after evolution')

    output = Path(runparams['savedir']) / 'BertschingerReference.hdf5'
    et.write_reference(output, profiles, {
        'Solution': 'Bertschinger1985_collisionless_radial',
        'SimilarityEpsilon': 1.0,
        'TurnaroundExponent': 8.0 / 9.0,
        'CosmologyType': 'einstein_de_sitter',
        'CosmologyTRef': cosmology.t_ref,
        'CosmologyARef': cosmology.a_ref,
        'InitialCosmicTime': initial_time,
        'FinalCosmicTime': final_time,
        'PerturbationMass': delta_mass,
        'SimilarityEquation': 'Bertschinger1985_Eq4.1_collisionless_shell',
        'ODEInitialLambda': 1.0,
        'ODEInitialLambdaPrime': -8.0 / 9.0,
        'ODEPoints': int(runparams['ode_points']),
        'ODESimilarityExponent': float(runparams['ode_similarity_exponent']),
        'ODEMassNormalization': 9.0 * np.pi**2 / 16.0,
    })
    output_ode = Path(runparams['savedir']) / 'BertschingerEq41ODE.hdf5'
    et.write_reference(output_ode, {
        'xi': ode_solution.xi,
        'lambda': ode_solution.lam,
        'lambda_prime': ode_solution.lam_prime,
        'mass': ode_solution.mass,
        'turnaround_radius': 1.0,
    }, {
        'Solution': 'Bertschinger1985_collisionless_shell_ODE',
        'Equation': 'Bertschinger1985_Eq4.1',
        'MassClosure': 'normalized_first_stream_exp_minus_2s_xi_over_3',
        'InitialLambda': 1.0,
        'InitialLambdaPrime': -8.0 / 9.0,
        'AngularMomentum': 0.0,
        'XiEnd': float(runparams['ode_xi_end']),
        'Points': int(runparams['ode_points']),
        'SimilarityExponent': float(runparams['ode_similarity_exponent']),
        'MassNormalization': 9.0 * np.pi**2 / 16.0,
    })
    ode_figure = Path(runparams['savedir']) / 'BertschingerEq41XiLambda.jpg'
    ode_plot = plot_xi_lambda(ode_solution, filename=ode_figure)
    ode_plot.figure.clf()
    import matplotlib.pyplot as plt
    plt.close(ode_plot.figure)
    figure = Path(runparams['savedir']) / 'BertschingerReference.jpg'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(profiles['lambda'], np.maximum(profiles['density'], 1.0e-12))
    axes[0].set(xlabel=r'$\lambda=r/r_{ta}$', ylabel=r'$\rho/\rho_b$')
    axes[1].plot(profiles['lambda'], profiles['velocity'])
    axes[1].set(xlabel=r'$\lambda=r/r_{ta}$', ylabel=r'$v/(r_{ta}/t)$')
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print('Bertschinger collisionless reference generated')
    print('turnaround radius = %.8g' % profiles['turnaround_radius'])
    print('output = %s' % output)
    print('Eq. 4.1 ODE output = %s' % output_ode)
    print('Eq. 4.1 xi-lambda figure = %s' % ode_figure)
    print('figure = %s' % figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
