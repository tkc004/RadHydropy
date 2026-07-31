"""Late phase isothermal H II region expansion in spherical 1D.

This example is from STARBENCH: The D-type expansion of an H II region
https://arxiv.org/abs/1507.05621v1

(Late phase of the expansion: note the neutral gas is at 10^3 K, not 10^2 K as in the early phase example.)

This example follows the hydrodynamic expansion of a central photoionized
region around a source at the origin. The gas is pure hydrogen, spherical,
and evolved with hydrodynamics plus hydrogen photo-chemistry. The neutral and
ionized media are both treated with a simplified isothermal closure:

* neutral gas: ``T = 10^3 K``;
* ionized gas: ``T = 10^4 K``.

The example is YAML-driven, writes HDF5 snapshots, reloads those snapshots,
and plots the ionization-front history and density profiles from the saved outputs.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import unyt

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import code_unit_scales
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('late_hii_region_expansion1d.yaml')


def write_initial_condition(sim, runparams):
    """Replace any stale late-phase initial condition snapshot."""
    Path(runparams['ICfilename']).unlink(missing_ok=True)
    rio.writehdf5(sim, runparams['ICfilename'])


def print_startup_diagnostics(sim, config, icparams):
    """Print the main physical scales before the long run starts."""
    interior = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    rho = np.asarray(sim.fluid.rho[interior], dtype=float)
    vel = np.asarray(sim.fluid.vel[interior], dtype=float)
    temp = np.asarray(sim.fluid.temp[interior], dtype=float)
    xHI = np.asarray(sim.fluid.xHI[interior], dtype=float)
    ngamma = np.asarray(sim.fluid.ngamma[interior], dtype=float) if hasattr(sim.fluid, 'ngamma') else None
    code_units = getattr(sim.par, 'code_units', getattr(sim.par, 'CodeUnits', None))
    ngamma_cgs = None
    scales = None
    if ngamma is not None and code_units is not None:
        scales = code_unit_scales(code_units)
        ngamma_cgs = ngamma * scales['number_density_cm3']

    print('--- Startup diagnostics ---')
    print('cells = %d' % sim.par.nogrid)
    print('time = %s' % sim.fluid.time)
    print('rho range = [%.3e, %.3e] g/cm^3' % (np.min(rho), np.max(rho)))
    print('vel max abs = %.3e km/s' % (np.max(np.abs(vel)) / 1.0e5))
    print('temperature range = [%.3e, %.3e] K' % (np.min(temp), np.max(temp)))
    print('neutral fraction range = [%.3e, %.3e]' % (np.min(xHI), np.max(xHI)))
    if ngamma is not None:
        print('ngamma range = [%.3e, %.3e] code units' % (np.min(ngamma), np.max(ngamma)))
        if ngamma_cgs is not None:
            print('ngamma range = [%.3e, %.3e] cm^-3' % (np.min(ngamma_cgs), np.max(ngamma_cgs)))
            boundary = np.asarray(sim.mesh.boundary[interior.start : interior.start + 2], dtype=float)
            if scales is not None:
                inner_radius_cm = 0.5 * (boundary[0] + boundary[1]) * scales['length_cm']
                thin_estimate = config['source_photon_rate'].to_value(1 / unyt.s) / (
                    4.0 * np.pi * inner_radius_cm**2 * unyt.c.to_value(unyt.cm / unyt.s)
                )
                print('optically thin inner-cell ngamma estimate = %.3e cm^-3' % thin_estimate)
    print('neutral sound speed = %.3e km/s' % et.neutral_sound_speed(config).to_value(unyt.km / unyt.s))
    print(
        'ionized sound speed (config) = %.3e km/s'
        % config['ionized_sound_speed'].to_value(unyt.km / unyt.s)
    )
    print('stromgren radius = %.3e pc' % et.stromgren_radius(config).to_value(unyt.pc))
    print('stagnation radius = %.3e pc' % et.stagnation_radius(config).to_value(unyt.pc))
    print(
        'Spitzer radius at final time = %.3e pc'
        % et.spitzer_radius(icparams['final_time'], config).to_value(unyt.pc)
    )
    print(
        'Hosokawa-Inutsuka radius at final time = %.3e pc'
        % et.hosokawa_inutsuka_radius(icparams['final_time'], config).to_value(unyt.pc)
    )
    try:
        hydro_dt = sim.solver.GetTimeStep(sim.mesh, sim.fluid, sim.par)
        hydro_dt_s = hydro_dt.to_value(unyt.s) if hasattr(hydro_dt, 'to_value') else float(hydro_dt)
        print('hydro timestep estimate = %.3e s' % hydro_dt_s)
    except Exception as exc:
        print('hydro timestep estimate failed: %s' % exc)
        hydro_dt_s = None
    try:
        source_dt, thermal_rate = sim.solver.GetSourceTimestepFast(
            sim.mesh,
            sim.fluid,
            sim.par,
            sim.par.dtmax,
        )
        source_dt_s = source_dt.to_value(unyt.s) if hasattr(source_dt, 'to_value') else float(source_dt)
        print('source timestep estimate = %.3e s' % source_dt_s)
        if hydro_dt_s is not None and source_dt_s > 0.0:
            print('estimated source substeps per hydro step = %.1f' % (hydro_dt_s / source_dt_s))
        if thermal_rate is not None:
            print('thermal rate range = [%.3e, %.3e]' % (np.min(np.asarray(thermal_rate, dtype=float)), np.max(np.asarray(thermal_rate, dtype=float))))
    except Exception as exc:
        print('source timestep estimate failed: %s' % exc)


def make_logging_step_backend(sim, config, max_logged_steps=5):
    """Wrap the isothermal step backend with a short startup trace."""
    base_step_backend = et.make_piecewise_isothermal_step_backend(sim, config)
    state = {'count': 0}
    interior = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)

    def step_backend(dt=None, mode='hydro_sources', advect_chemistry=True):
        step_index = state['count']
        should_log = step_index < max_logged_steps
        if should_log:
            print(
                '--- step %d begin: time=%s dt=%s mode=%s ---'
                % (step_index + 1, sim.fluid.time, dt, mode)
            )
        result = base_step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
        )
        if should_log:
            vel = np.asarray(sim.fluid.vel[interior], dtype=float)
            rho = np.asarray(sim.fluid.rho[interior], dtype=float)
            xHI = np.asarray(sim.fluid.xHI[interior], dtype=float)
            vmax = np.max(np.abs(vel)) / 1.0e5
            front_radius = et.ionization_front_position(sim.mesh, sim.fluid, sim.par)
            print(
                '--- step %d end: time=%s hydro_steps=%d source_steps=%d front=%.3e pc vmax=%.3e km/s rho=[%.3e, %.3e] xHI=[%.3e, %.3e] ---'
                % (
                    step_index + 1,
                    sim.fluid.time,
                    result['hydro_steps'],
                    result['source_steps'],
                    front_radius,
                    vmax,
                    np.min(rho),
                    np.max(rho),
                    np.min(xHI),
                    np.max(xHI),
                )
            )
            if step_index + 1 == max_logged_steps:
                print('--- step logging disabled after %d steps ---' % max_logged_steps)
        state['count'] += 1
        return result

    return step_backend


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
    write_initial_condition(sim, runparams)
    print_startup_diagnostics(sim, config, icparams)

    output_specs = icparams['output_snapshots']
    step_backend = make_logging_step_backend(sim, config, max_logged_steps=5)
    print('starting hydro_sources evolution; this may take a while...')
    sim.Run(
        outputtime=0,
        mode='hydro_sources',
        step_backend=step_backend,
    )
    print('finished evolution; loading saved outputs and building plots...')
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
