"""Photoheated Stromgren sphere with hydrodynamic expansion.

This repeats the variable-temperature Stromgren sphere setup, but advances the
Euler equations so the ionized gas can expand as it is heated.
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

from radhydropy.rsim import Rsim
import tools as et


rundir = os.path.dirname(os.path.abspath(__file__))
figure_filename = os.path.join(
    rundir,
    'DynamicStromgrenSpherePhotoheating1D.jpg',
)
front_figure_filename = os.path.join(
    rundir,
    'DynamicStromgrenSpherePhotoheating1D_IFront.jpg',
)
density_reference_filename = os.path.join(
    rundir,
    'Stromgren3D_rhd_n_r_zeusmp_t200.csv',
)
velocity_reference_filename = os.path.join(
    rundir,
    'Stromgren3D_rhd_v_r_zeusmp_t200.csv',
)
pressure_reference_filename = os.path.join(
    rundir,
    'Stromgren3D_rhd_p_r_zeusmp_t200.csv',
)
neutral_fraction_reference_filename = os.path.join(
    rundir,
    'Stromgren3D_rhd_x_r_zeusmp_t200.csv',
)

hydrogen_number_density = 1.0e-3 / unyt.cm**3
alpha_B_coefficient = 2.59e-13 * unyt.cm**3 / unyt.s
sigma_gamma = 1.62e-18 * unyt.cm**2
source_photon_rate = 5.0e48 / unyt.s
epsilon_gamma = 6.33 * unyt.eV
boxsize = 20.0 * unyt.kpc
plot_radius_max = 7.5 * unyt.kpc
final_time = 200.0 * unyt.Myr
reference_radius_unit = 15.0 * unyt.kpc
hydro_cfl = 0.5
source_cfl = 0.1
hydro_timestep_max = 1.0 * unyt.Myr
source_timestep_min = 1.0e-3 * unyt.Myr
number_of_cells = 1024
initial_temperature = 100.0 * unyt.K


def main():
    config = {
        'hydrogen_number_density': hydrogen_number_density,
        'alpha_B_coefficient': alpha_B_coefficient,
        'sigma_gamma': sigma_gamma,
        'source_photon_rate': source_photon_rate,
        'epsilon_gamma': epsilon_gamma,
        'boxsize': boxsize,
        'plot_radius_max': plot_radius_max,
        'number_of_cells': number_of_cells,
        'initial_temperature': initial_temperature,
        'hydro_cfl': hydro_cfl,
        'source_cfl': source_cfl,
        'hydro_timestep_max': hydro_timestep_max,
        'source_timestep_min': source_timestep_min,
        'reference_radius_unit': reference_radius_unit,
        'density_reference_filename': density_reference_filename,
        'velocity_reference_filename': velocity_reference_filename,
        'pressure_reference_filename': pressure_reference_filename,
        'neutral_fraction_reference_filename': neutral_fraction_reference_filename,
    }
    par, mesh, fluid, solver = et.build_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    history = {
        'time_Myr': [],
        'front_radius_kpc': [],
        'mean_ionized_temperature_K': [],
    }
    counters = sim.Evolve(
        final_time=final_time,
        mode='hydro_sources',
        fast_thermochemistry=True,
        history_callback=lambda current_sim: et.append_history(
            history,
            current_sim.mesh,
            current_sim.fluid,
            current_sim.par,
        ),
    )
    sim.solver.TraceSphericalPhotonDensityFast(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    history.update(counters)
    et.save_plot(mesh, fluid, par, config, figure_filename)
    et.save_front_plot(history, config, front_figure_filename)

    print('time = %s' % fluid.time)
    print('hydro steps = %d' % history['hydro_steps'])
    print('source steps = %d' % history['source_steps'])
    print('figure = %s' % figure_filename)
    print('front figure = %s' % front_figure_filename)


if __name__ == '__main__':
    main()
