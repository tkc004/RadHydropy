"""Static Stromgren sphere with photoheating.

This repeats the static Stromgren sphere benchmark, but lets the hydrogen
source update heat and cool the gas. Hydrodynamic motion is disabled: density
is fixed and only radiative transfer, chemistry, and thermal source terms are
advanced.
"""

import os
import sys
import tempfile

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
static_stromgren_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'StaticStromgrenSphere1D')
)
if static_stromgren_dir not in sys.path:
    sys.path.append(static_stromgren_dir)

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import unyt

from radhydropy.rsim import Rsim
import stromgren_analytic as sa
import tools as et


rundir = os.path.dirname(os.path.abspath(__file__))
figure_filename = os.path.join(
    rundir,
    'StaticStromgrenSpherePhotoheating1D.jpg',
)
temperature_reference_filename = os.path.join(
    rundir,
    'TTT1Dthin_Stromgren100Myr.txt',
)
neutral_fraction_reference_filename = os.path.join(
    rundir,
    'xTT1Dthin_Stromgren100Myr.txt',
)

hydrogen_number_density = 1.0e-3 / unyt.cm**3
alpha_B_coefficient = 2.59e-13 * unyt.cm**3 / unyt.s
sigma_gamma = 1.62e-18 * unyt.cm**2
source_photon_rate = 5.0e48 / unyt.s
epsilon_gamma = 6.33 * unyt.eV
boxsize = 20.0 * unyt.kpc
plot_radius_max = 7.5 * unyt.kpc
final_time = 500.0 * unyt.Myr
reference_time = 100.0 * unyt.Myr
reference_radius_unit = 5.4 * unyt.kpc
evolution_timestep = 1.0 * unyt.Myr
evolution_timestep_min = 1.0e-3 * unyt.Myr
evolution_timestep_cfl = 0.1
radiative_transfer_update_interval = 5
number_of_cells = 1024
analytic_inner_radius = 0.1 * unyt.kpc
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
        'analytic_inner_radius': analytic_inner_radius,
        'initial_temperature': initial_temperature,
        'reference_time': reference_time,
        'reference_radius_unit': reference_radius_unit,
        'temperature_reference_filename': temperature_reference_filename,
        'neutral_fraction_reference_filename': neutral_fraction_reference_filename,
        'evolution_timestep_min': evolution_timestep_min,
        'evolution_timestep_cfl': evolution_timestep_cfl,
        'radiative_transfer_update_interval': radiative_transfer_update_interval,
    }
    par, mesh, fluid, solver = et.build_static_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    history = sim.EvolveStaticThermochemistry(
        final_time,
        evolution_timestep,
        include_thermal_history=True,
        reference_time=reference_time,
    )
    et.save_plot(mesh, fluid, par, history, config, figure_filename)

    print('time = %s' % fluid.time)
    print(
        'stromgren radius = %s'
        % sa.stromgren_radius(
            source_photon_rate,
            hydrogen_number_density,
            alpha_B_coefficient,
        ).to(unyt.kpc)
    )
    print(
        'analytic front radius = %s'
        % sa.ionization_front_radius(
            final_time,
            source_photon_rate,
            hydrogen_number_density,
            alpha_B_coefficient,
        ).to(unyt.kpc)
    )
    print('mean ionized temperature = %.3e K' % history['mean_ionized_temp_K'][-1])
    print('front radius = %.3e kpc' % history['front_radius_kpc'][-1])
    print('evolution steps = %d' % history['evolution_steps'])
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
