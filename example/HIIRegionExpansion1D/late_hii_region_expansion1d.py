"""Late phase isothermal H II region expansion in spherical 1D.

This example is from STARBENCH: The D-type expansion of an H II region
https://arxiv.org/abs/1507.05621v1

(Late phase of the expansion: note the neutral gas is at 10^3 K, not 10^2 K as in the early phase example.)

This example follows the hydrodynamic expansion of a central photoionized
region around a source at the origin. The gas is pure hydrogen, spherical,
and evolved with hydrodynamics plus hydrogen photo-chemistry. The neutral and
ionized media are both treated with a simplified isothermal closure:

* neutral gas: ``T = 10^2 K``;
* ionized gas: ``T = 10^4 K``.

The example is YAML-driven, writes HDF5 snapshots, reloads those snapshots,
and plots the ionization-front history and density profiles from the saved outputs.
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

from radhydropy.rsim import Rsim
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('late_hii_region_expansion1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, icparams = et.load_parameters(config_filename, rundir)
    config = {**runparams, **icparams}

    Path(runparams['outdir']).mkdir(parents=True, exist_ok=True)
    Path(runparams['savedir']).mkdir(parents=True, exist_ok=True)

    par, mesh, fluid, solver = et.build_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    et.apply_piecewise_isothermal_state(sim.mesh, sim.fluid, sim.par, sim.solver, config)

    output_specs = icparams['output_snapshots']
    step_backend = et.make_piecewise_isothermal_step_backend(sim, config)
    sim.Run(
        outputtime=0,
        mode='hydro_sources',
        fast_thermochemistry=True,
        step_backend=step_backend,
    )
    outputfilenames = et.output_files(runparams['outdir'], runparams['outfileprefix'])

    history = et.load_history_from_outputs(outputfilenames, config)

    figure_filename = Path(runparams['savedir']) / 'LateHIIRegionExpansion1D_IFront.jpg'
    et.save_front_plot(history, config, figure_filename)

    density_figure_filenames = []
    for label, snapshot in et.load_labeled_density_snapshots(
        outputfilenames,
        config,
        output_specs,
    ):
        density_figure_filename = Path(runparams['savedir']) / (
            f"LateHIIRegionExpansion1D_Density_{label}Myr.jpg"
        )
        et.save_density_profile_plot(snapshot, config, density_figure_filename)
        density_figure_filenames.append(density_figure_filename)

    comparison_time_myr = icparams['comparison_time'].to_value(unyt.Myr)
    simulation_radius_pc = et.front_radius_at_time(
        history,
        icparams['comparison_time'],
    ).to_value(unyt.pc)
    spitzer_radius_pc = et.spitzer_radius(
        icparams['comparison_time'],
        config,
    ).to_value(unyt.pc)
    hosokawa_inutsuka_radius_pc = et.hosokawa_inutsuka_radius(
        icparams['comparison_time'],
        config,
    ).to_value(unyt.pc)
    stagnation_radius_pc = et.stagnation_radius(config).to_value(unyt.pc)

    print('time = %s' % sim.fluid.time)
    print('stromgren radius = %.3e pc' % et.stromgren_radius(config).to_value(unyt.pc))
    print('stagnation radius = %.3e pc' % stagnation_radius_pc)
    print('output files = %d' % len(outputfilenames))
    print(
        'final ionization-front radius = %.3e pc'
        % history['front_radius_pc'][-1]
    )
    print(
        'simulation ionization-front radius at %.2f Myr = %.3e pc'
        % (comparison_time_myr, simulation_radius_pc)
    )
    print(
        'Spitzer solution at %.2f Myr = %.3e pc'
        % (comparison_time_myr, spitzer_radius_pc)
    )
    print(
        'Hosokawa-Inutsuka solution at %.2f Myr = %.3e pc'
        % (comparison_time_myr, hosokawa_inutsuka_radius_pc)
    )
    print('figure = %s' % figure_filename)
    for density_figure_filename in density_figure_filenames:
        print('density figure = %s' % density_figure_filename)
    for outputfilename in outputfilenames:
        print('output file = %s' % outputfilename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the late HII region expansion example.',
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
