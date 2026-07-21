"""Optically thin hydrogen photoheating and recombination parcel.

An initially neutral pure-hydrogen parcel with fixed total density is exposed
to a spatially uniform ionizing radiation field. The radiation is treated as
optically thin, so the photon density is fixed while the source is on and set
to zero when the source switches off.
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

photon_flux = 1.0e12 / (unyt.s * unyt.cm**2)
hydrogen_number_density = 1.0 / unyt.cm**3
excess_photoionization_energy = 6.33 * unyt.eV
sigma_gamma = rh.DEFAULT_SIGMA_GAMMA
thermal_equilibrium_timescale = 10.0**9.3 * unyt.yr
source_switch_time = 5.0e7 * unyt.yr
final_time = 1.0e8 * unyt.yr
reference = et.reference_values(
    photon_flux,
    hydrogen_number_density,
    excess_photoionization_energy,
    sigma_gamma,
    thermal_equilibrium_timescale,
)
photon_density_on = reference['photon_density_on']
photoionization_equilibrium_temperature = reference['photoionization_temperature']
thermal_equilibrium_temperature = reference['thermal_temperature']

runparams = {
    'simname': 'HydrogenPhotoheating1D',
    'ICfilename': rundir + '/InitialCondition.hdf5',
    'outdir': rundir,
    'outfileprefix': 'Output',
    'outdeltatime': 1.0e7 * unyt.yr,
    'savedir': rundir,
    'coordsys': 'cartesian',
    'EOStype': 'polytropic',
    'gamma': 5.0 / 3.0,
    'timesim': final_time,
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
    'dtmin': 1.0e-9 * unyt.yr,
    'dtmax': 1.0e6 * unyt.yr,
    'hydrogen_chemistry': True,
    'hydrogen_mass_fraction': 1.0,
    'hydrogen_xHI_initial': 1.0,
    'hydrogen_xHI_inflow': 1.0,
    'hydrogen_xHI_outflow': 1.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_update_mu': True,
    'hydrogen_thermal_coupling': True,
    'hydrogen_recombination': True,
    'hydrogen_collisional_ionization': True,
    'hydrogen_radiation_field': True,
    'hydrogen_radiation_evolution': False,
    'hydrogen_ngamma_initial': photon_density_on,
    'hydrogen_ngamma_inflow': photon_density_on,
    'hydrogen_ngamma_outflow': photon_density_on,
    'hydrogen_sigma_gamma': sigma_gamma,
    'hydrogen_epsilon_gamma': 6.33 * unyt.eV,
}

ICparams = {
    'nogrid': 16,
    'coordsys': 'cartesian',
    'boxsize': 1.0 * unyt.kpc,
    'time': 0.0 * unyt.yr,
    'nHini': hydrogen_number_density,
    'tempini': 100.0 * unyt.K,
    'xHIini': 1.0,
    'ngammaini': photon_density_on,
    'muini': 1.0,
}


def main():
    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    history = {'time_yr': [], 'temperature_K': [], 'xHI': [], 'ngamma': []}
    et.append_history(sim, history)
    et.write_output(sim, 0)

    output_interval = sim.par.outdeltatime.copy()
    next_output_time = output_interval.copy()
    outindex = 1
    last_output_time = et.time_value(sim, unyt.s)

    for target_time in et.sample_times(source_switch_time, final_time)[1:]:
        while et.current_time(sim) < target_time:
            start_time = et.current_time(sim)
            dt = target_time - start_time
            if start_time < source_switch_time < start_time + dt:
                dt = source_switch_time - start_time
            et.advance_sources(sim, dt, source_switch_time, photon_density_on)
            if et.current_time(sim) >= next_output_time:
                last_output_time = et.write_output(sim, outindex)
                outindex += 1
                next_output_time += output_interval
        et.append_history(sim, history)

    if et.time_value(sim, unyt.s) != last_output_time:
        et.write_output(sim, outindex)

    figure_filename = rundir + '/HydrogenPhotoheating1D.jpg'
    xHI_reference = et.save_history_plot(history, figure_filename, reference)

    print('Hydrogen photoheating example finished')
    print('time = %.3e yr' % et.time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % et.mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % et.mean_neutral_fraction(sim))
    print(
        'mean photon number density = %.3e cm^-3'
        % et.mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
    )
    print(
        'sigma_gamma = %.3e cm^2'
        % runparams['hydrogen_sigma_gamma'].to_value(unyt.cm**2)
    )
    print(
        'epsilon_gamma = %.3e eV'
        % runparams['hydrogen_epsilon_gamma'].to_value(unyt.eV)
    )
    print(
        'photoionization equilibrium temperature = %.3e K'
        % photoionization_equilibrium_temperature.to_value(unyt.K)
    )
    print(
        'thermal equilibrium reference temperature = %.3e K'
        % thermal_equilibrium_temperature.to_value(unyt.K)
    )
    print(
        'ionization time = %.3e yr'
        % xHI_reference['ionization_timescale'].to_value(unyt.yr)
    )
    print(
        'recombination time at T_ion = %.3e yr'
        % xHI_reference['recombination_timescale'].to_value(unyt.yr)
    )
    print(
        'ionization equilibrium neutral fraction = %.3e'
        % xHI_reference['xHI']
    )
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
