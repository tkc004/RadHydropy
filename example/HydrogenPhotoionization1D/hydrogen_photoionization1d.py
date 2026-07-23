"""Fixed-radiation hydrogen photoionization box.

The gas starts neutral at ``T = 2e4 K`` and ``nH = 1 cm^-3``. A fixed,
spatially uniform photon number density photoionizes the gas while the
radiation-field evolution and thermal source update are disabled. The run
stops once the gas is 99 percent ionized and writes a JPG comparing the
neutral fraction against the analytic fixed-field solution.
"""

import os
import tempfile

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault(
    'MPLCONFIGDIR',
    mplconfig_dir,
)
import unyt

from radhydropy.rsim import Rsim
import radhydropy.hydrogen as rh
import radhydropy.io as rio
import tools as et


rundir = os.getcwd()

runparams = {
    'simname': 'HydrogenPhotoionization1D',
    'ICfilename': rundir + '/InitialCondition.hdf5',
    'outdir': rundir,
    'outfileprefix': 'Output',
    'outdeltatime': 5.0e2 * unyt.yr,
    'savedir': rundir,
    'coordsys': 'cartesian',
    'EOStype': 'polytropic',
    'gamma': 5.0 / 3.0,
    'timesim': 2.0e4 * unyt.yr,
    'area': 1.0 * unyt.cm**2,
    'CFL': 0.5,
    'boundcond': 'Periodic',
    'vel_inflow': 0.0 * unyt.cm / unyt.s,
    'rho_inflow': 1.0 * unyt.mp / unyt.cm**3,
    'temp_inflow': 0.0 * unyt.K,
    'mu_inflow': 1.0,
    'vel_outflow': 0.0 * unyt.cm / unyt.s,
    'rho_outflow': 1.0 * unyt.mp / unyt.cm**3,
    'temp_outflow': 0.0 * unyt.K,
    'mu_outflow': 1.0,
    'noghost': 2,
    'verbose': 0,
    'order': 0,
    'dtmin': 1.0e-6 * unyt.yr,
    'dtmax': 2.0e1 * unyt.yr,
    'hydrogen_chemistry': True,
    'hydrogen_mass_fraction': 1.0,
    'hydrogen_xHI_initial': 1.0,
    'hydrogen_xHI_inflow': 1.0,
    'hydrogen_xHI_outflow': 1.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_update_mu': False,
    'hydrogen_thermal_coupling': False,
    'hydrogen_collisional_ionization': False,
    'hydrogen_radiation_field': True,
    'hydrogen_radiation_evolution': False,
    'hydrogen_ngamma_initial': 1.0e-3 / unyt.cm**3,
    'hydrogen_ngamma_inflow': 1.0e-3 / unyt.cm**3,
    'hydrogen_ngamma_outflow': 1.0e-3 / unyt.cm**3,
    'hydrogen_sigma_gamma': rh.DEFAULT_SIGMA_GAMMA,
    'hydrogen_epsilon_gamma': 0.0 * unyt.erg,
}

ICparams = {
    'nogrid': 16,
    'coordsys': 'cartesian',
    'boxsize': 1.0 * unyt.kpc,
    'time': 0.0 * unyt.yr,
    'nHini': 1.0 / unyt.cm**3,
    'tempini': 2.0e4 * unyt.K,
    'xHIini': 1.0,
    'ngammaini': runparams['hydrogen_ngamma_initial'],
    'muini': 1.0,
}

target_neutral_fraction = 0.01


def main():
    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    history = {'time_yr': [], 'temperature_K': [], 'xHI': [], 'ngamma': []}
    outindex = 0
    output_interval = sim.par.outdeltatime.copy()
    next_output_time = output_interval.copy()
    last_output_time = et.write_output(sim, outindex)
    et.append_history(sim, history)
    outindex += 1

    while (
        et.mean_neutral_fraction(sim) > target_neutral_fraction
        and et.time_value(sim, unyt.s) < float(sim.par.timesim.to_value(unyt.s))
    ):
        sim.Step(mode='hydro_sources')
        et.append_history(sim, history)
        if sim.fluid.time >= next_output_time:
            last_output_time = et.write_output(sim, outindex)
            outindex += 1
            next_output_time += output_interval

    if et.time_value(sim, unyt.s) != last_output_time:
        et.write_output(sim, outindex)

    figure_filename = rundir + '/HydrogenPhotoionization1D.jpg'
    et.save_history_plot(
        history,
        figure_filename,
        ICparams,
        runparams,
        target_neutral_fraction,
    )

    print('Hydrogen photoionization example finished')
    print('time = %.3e yr' % et.time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % et.mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % et.mean_neutral_fraction(sim))
    print(
        'mean photon number density = %.3e cm^-3'
        % et.mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
    )
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
