"""Static Stromgren sphere at constant temperature.

This benchmark keeps the gas density and temperature fixed. A central source
emits ionizing photons at a constant rate, the long-characteristic
radiative-transfer update supplies ``n_gamma``, and the hydrogen neutral
fraction is advanced with the implicit chemistry solver. Hydrodynamics,
heating, and cooling are disabled.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import unyt

from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
import radhydropy.io as rio
import stromgren_analytic as sa
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('static_stromgren_sphere1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, icparams = load_example_parameters(config_filename, rundir)
    config = {**runparams, **icparams}

    Path(runparams['outdir']).mkdir(parents=True, exist_ok=True)
    Path(runparams['savedir']).mkdir(parents=True, exist_ok=True)

    par, mesh, fluid, solver = et.build_static_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)

    rio.writehdf5(sim, runparams['ICfilename'])

    front_history = sim.EvolveStaticThermochemistry(
        config['final_time'],
        config['chemistry_timestep'],
    )

    output_filename = Path(runparams['outdir']) / f"{runparams['outfileprefix']}_000.hdf5"
    rio.writehdf5(sim, output_filename)

    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    figure_filename = Path(runparams['savedir']) / 'StaticStromgrenSphere1D.jpg'
    front_figure_filename = Path(runparams['savedir']) / 'StaticStromgrenSphere1D_IFront.jpg'
    budget_figure_filename = Path(runparams['savedir']) / 'StaticStromgrenSphere1D_PhotonBudget.jpg'

    et.save_plot(out_mesh, out_fluid, out_par, config, figure_filename)
    et.save_front_history_plot(front_history, config, front_figure_filename)
    photon_budget = et.save_photon_budget_plot(
        front_history,
        budget_figure_filename,
    )

    print('time = %s' % out_fluid.time)
    print(
        'recombination time = %s'
        % sa.recombination_time(
            config['hydrogen_number_density'],
            config['alpha_B_coefficient'],
        )
    )
    print(
        'stromgren radius = %s'
        % sa.stromgren_radius(
            config['source_photon_rate'],
            config['hydrogen_number_density'],
            config['alpha_B_coefficient'],
        ).to(unyt.kpc)
    )
    print(
        'analytic front radius = %s'
        % sa.ionization_front_radius(
            config['final_time'],
            config['source_photon_rate'],
            config['hydrogen_number_density'],
            config['alpha_B_coefficient'],
        ).to(unyt.kpc)
    )
    print(
        'injected photons = %.6e, accounted photons = %.6e'
        % (
            photon_budget['injected_photons'],
            photon_budget['accounted_photons'],
        )
    )
    print(
        'ionized H = %.6e, recombinations = %.6e, photons in volume = %.6e'
        % (
            photon_budget['ionized_atoms'],
            photon_budget['recombined_photons'],
            photon_budget['volume_photons'],
        )
    )
    print(
        'photon budget relative error = %.6e'
        % photon_budget['relative_error']
    )
    print(
        'chemistry steps = %d, radiative-transfer updates = %d'
        % (
            front_history['chemistry_steps'],
            front_history['radiative_transfer_updates'],
        )
    )
    print('IC file = %s' % runparams['ICfilename'])
    print('output file = %s' % output_filename)
    print('figure = %s' % figure_filename)
    print('front figure = %s' % front_figure_filename)
    print('photon budget figure = %s' % budget_figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the static Stromgren sphere example.',
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file containing runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
