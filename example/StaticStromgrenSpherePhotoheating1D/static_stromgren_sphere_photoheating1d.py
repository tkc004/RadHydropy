"""Static Stromgren sphere with photoheating.

This repeats the static Stromgren sphere benchmark, but lets the hydrogen
source update heat and cool the gas. Hydrodynamic motion is disabled: density
is fixed and only radiative transfer, chemistry, and thermal source terms are
advanced. The example is configured from YAML, writes HDF5 snapshots, reloads
the final snapshot, and plots from the saved output rather than live state.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
static_stromgren_dir = Path(__file__).resolve().parents[1] / 'StaticStromgrenSphere1D'
if str(static_stromgren_dir) not in sys.path:
    sys.path.append(str(static_stromgren_dir))

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


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    'static_stromgren_sphere_photoheating1d.yaml'
)


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, icparams = load_example_parameters(config_filename, rundir)
    config_dir = Path(config_filename).resolve().parent
    for key in (
        'temperature_reference_filename',
        'neutral_fraction_reference_filename',
    ):
        if key in icparams:
            value = Path(icparams[key])
            if not value.is_absolute():
                icparams[key] = str(config_dir / value)
    config = {**runparams, **icparams}

    Path(runparams['outdir']).mkdir(parents=True, exist_ok=True)
    Path(runparams['savedir']).mkdir(parents=True, exist_ok=True)

    par, mesh, fluid, solver = et.build_static_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)

    rio.writehdf5(sim, runparams['ICfilename'])

    history = sim.EvolveStaticThermochemistry(
        icparams['final_time'],
        icparams['evolution_timestep'],
        include_thermal_history=True,
        reference_time=icparams['reference_time'],
    )

    output_filename = Path(runparams['outdir']) / f"{runparams['outfileprefix']}_000.hdf5"
    rio.writehdf5(sim, output_filename)

    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    figure_filename = Path(runparams['savedir']) / 'StaticStromgrenSpherePhotoheating1D.jpg'
    et.save_plot(out_mesh, out_fluid, out_par, history, config, figure_filename)

    print('time = %s' % out_fluid.time)
    print(
        'stromgren radius = %s'
        % sa.stromgren_radius(
            icparams['source_photon_rate'],
            icparams['hydrogen_number_density'],
            icparams['alpha_B_coefficient'],
        ).to(unyt.kpc)
    )
    print(
        'analytic front radius = %s'
        % sa.ionization_front_radius(
            icparams['final_time'],
            icparams['source_photon_rate'],
            icparams['hydrogen_number_density'],
            icparams['alpha_B_coefficient'],
        ).to(unyt.kpc)
    )
    print('mean ionized temperature = %.3e K' % history['mean_ionized_temp_K'][-1])
    print('front radius = %.3e kpc' % history['front_radius_kpc'][-1])
    print('evolution steps = %d' % history['evolution_steps'])
    print('IC file = %s' % runparams['ICfilename'])
    print('output file = %s' % output_filename)
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the static Stromgren sphere photoheating example.',
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
