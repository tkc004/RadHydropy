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

The plotted ionization-front radius is defined by ``xHII = 0.5``.
"""

import os
import sys
import tempfile

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import unyt

import radhydropy.hydrogen as rh
from radhydropy.rsim import Rsim
import tools as et


rundir = os.path.dirname(os.path.abspath(__file__))
figure_filename = os.path.join(rundir, 'LateHIIRegionExpansion1D_IFront.jpg')
density_output_specs = (
    (0.05 * unyt.Myr, '0p05'),
    (0.20 * unyt.Myr, '0p20'),
    (0.80 * unyt.Myr, '0p80'),
    (3.0 * unyt.Myr, '3p00'),
)
density_figure_filenames = [
    os.path.join(
        rundir,
        'LateHIIRegionExpansion1D_Density_%sMyr.jpg' % label,
    )
    for _, label in density_output_specs
]

# Physical parameters
# This is Lyman continuum photon rate:
source_photon_rate = 1.0e49 / unyt.s
rho_initial = 5.21e-21 * unyt.g / unyt.cm**3
neutral_temperature = 1.0e3 * unyt.K
ionized_temperature = 1.0e4 * unyt.K
ionized_sound_speed = 12.85 * unyt.km / unyt.s
# Lyman continuum photoionization cross-section at 13.6 eV:
sigma_gamma = 6.3e-18 * unyt.cm**2
# Approximate recombination coefficient for hydrogen at 10^4 K, case B:
alpha_B_coefficient = 2.7e-13 * unyt.cm**3 / unyt.s
boxsize = 7.0 * unyt.pc
final_time = 3.0 * unyt.Myr
hydro_cfl = 0.5
hydro_order = 1
source_cfl = 0.1
hydro_timestep_max = 2.0e-4 * unyt.Myr
source_timestep_min = 1.0e-12 * unyt.Myr
number_of_cells = 512
comparison_time = 3.0 * unyt.Myr
show_stagnation_radius = True


def main():
    config = {
        'source_photon_rate': source_photon_rate,
        'rho_initial': rho_initial,
        'neutral_temperature': neutral_temperature,
        'ionized_temperature': ionized_temperature,
        'ionized_sound_speed': ionized_sound_speed,
        'sigma_gamma': sigma_gamma,
        'alpha_B_coefficient': alpha_B_coefficient,
        'boxsize': boxsize,
        'final_time': final_time,
        'hydro_cfl': hydro_cfl,
        'hydro_order': hydro_order,
        'source_cfl': source_cfl,
        'hydro_timestep_max': hydro_timestep_max,
        'source_timestep_min': source_timestep_min,
        'number_of_cells': number_of_cells,
        'show_stagnation_radius': show_stagnation_radius,
    }
    par, mesh, fluid, solver = et.build_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    et.apply_piecewise_isothermal_state(sim.mesh, sim.fluid, sim.par, sim.solver, config)

    history = {
        'time_Myr': [],
        'front_radius_pc': [],
    }
    density_snapshots = []
    density_snapshot_times = [time for time, _ in density_output_specs]
    next_density_snapshot = 0
    counters = {
        'hydro_steps': 0,
        'source_steps': 0,
    }
    et.append_history(history, sim.mesh, sim.fluid, sim.par)
    while sim.fluid.time < final_time:
        step_final_time = final_time
        if next_density_snapshot < len(density_snapshot_times):
            step_final_time = min(
                final_time,
                density_snapshot_times[next_density_snapshot],
            )
        dt = sim.GetStepTime(final_time=step_final_time)
        step = sim.Step(
            dt=dt,
            mode='hydro_sources',
            fast_thermochemistry=True,
        )
        counters['hydro_steps'] += step['hydro_steps']
        counters['source_steps'] += step['source_steps']
        et.apply_piecewise_isothermal_state(
            sim.mesh,
            sim.fluid,
            sim.par,
            sim.solver,
            config,
        )
        et.append_history(history, sim.mesh, sim.fluid, sim.par)
        while next_density_snapshot < len(density_snapshot_times):
            current_time_myr = sim.fluid.time.to_value(unyt.Myr)
            snapshot_time_myr = density_snapshot_times[
                next_density_snapshot
            ].to_value(unyt.Myr)
            if current_time_myr + 1.0e-12 < snapshot_time_myr:
                break
            density_snapshots.append(
                et.density_snapshot(sim.mesh, sim.fluid, sim.par)
            )
            next_density_snapshot += 1

    sim.solver.TraceSphericalPhotonDensityFast(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    et.save_front_plot(history, config, figure_filename)
    et.save_density_profile_plots(
        density_snapshots,
        config,
        density_figure_filenames,
    )

    comparison_time_myr = comparison_time.to_value(unyt.Myr)
    simulation_radius_pc = et.front_radius_at_time(
        history,
        comparison_time,
    ).to_value(unyt.pc)
    spitzer_radius_pc = et.spitzer_radius(
        comparison_time,
        config,
    ).to_value(unyt.pc)
    hosokawa_inutsuka_radius_pc = et.hosokawa_inutsuka_radius(
        comparison_time,
        config,
    ).to_value(unyt.pc)
    stagnation_radius_pc = et.stagnation_radius(config).to_value(unyt.pc)

    print('time = %s' % sim.fluid.time)
    print('stromgren radius = %.3e pc' % et.stromgren_radius(config).to_value(unyt.pc))
    print('stagnation radius = %.3e pc' % stagnation_radius_pc)
    print('hydro steps = %d' % counters['hydro_steps'])
    print('source steps = %d' % counters['source_steps'])
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


if __name__ == '__main__':
    main()
