"""Static Stromgren sphere at constant temperature.

This benchmark keeps the gas density and temperature fixed. A central source
emits ionizing photons at a constant rate, the long-characteristic
radiative-transfer update supplies ``n_gamma``, and the hydrogen neutral
fraction is advanced with the implicit chemistry solver. Hydrodynamics,
heating, and cooling are disabled.
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

import stromgren_analytic as sa
import tools as et


rundir = os.path.dirname(os.path.abspath(__file__))
figure_filename = os.path.join(rundir, 'StaticStromgrenSphere1D.jpg')

hydrogen_number_density = 1.0e-3 / unyt.cm**3
alpha_B_coefficient = 2.59e-13 * unyt.cm**3 / unyt.s
sigma_gamma = 8.13e-18 * unyt.cm**2
source_photon_rate = 5.0e48 / unyt.s
boxsize = 20.0 * unyt.kpc
final_time = 500.0 * unyt.Myr
chemistry_timestep = 1.0 * unyt.Myr
number_of_cells = 256
analytic_inner_radius = 0.1 * unyt.kpc


def main():
    config = {
        'hydrogen_number_density': hydrogen_number_density,
        'alpha_B_coefficient': alpha_B_coefficient,
        'sigma_gamma': sigma_gamma,
        'source_photon_rate': source_photon_rate,
        'boxsize': boxsize,
        'number_of_cells': number_of_cells,
        'analytic_inner_radius': analytic_inner_radius,
    }
    par, mesh, fluid, solver = et.build_static_problem(config)
    et.evolve_static_chemistry(
        mesh,
        fluid,
        par,
        solver,
        final_time,
        chemistry_timestep,
    )
    et.save_plot(mesh, fluid, par, config, figure_filename)
    print('time = %s' % fluid.time)
    print(
        'recombination time = %s'
        % sa.recombination_time(hydrogen_number_density, alpha_B_coefficient)
    )
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
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
