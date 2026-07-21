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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.rsim import Rsim
import radhydropy.hydrogen as rh
import radhydropy.io as rio


rundir = os.getcwd()

photon_flux = 1.0e12 / (unyt.s * unyt.cm**2)
ionization_timescale = 10.0**-2.3 * unyt.yr
recombination_timescale = 10.0**5.1 * unyt.yr
thermal_equilibrium_timescale = 10.0**9.3 * unyt.yr
source_switch_time = 5.0e7 * unyt.yr
final_time = 1.0e8 * unyt.yr
photon_density_on = (photon_flux / rh.SPEED_OF_LIGHT).to(1.0 / unyt.cm**3)
sigma_gamma = (1.0 / (photon_flux * ionization_timescale)).to(unyt.cm**2)
photoionization_equilibrium_temperature = (
    6.33 * unyt.eV / (3.0 * unyt.kb)
).to(unyt.K)
thermal_equilibrium_temperature = 2.0 * photoionization_equilibrium_temperature

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
    'nHini': 1.0 / unyt.cm**3,
    'tempini': 100.0 * unyt.K,
    'xHIini': 1.0,
    'ngammaini': photon_density_on,
    'muini': 1.0,
}


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()

        self.par.nogrid = ICparams['nogrid']
        self.par.coordsys = ICparams['coordsys']
        self.par.boxsize = np.ones(1) * ICparams['boxsize']
        self.par.time = np.ones(1) * ICparams['time']

        self.mesh.boundary = np.linspace(
            0.0,
            1.0,
            self.par.nogrid + 1,
        ) * ICparams['boxsize']

        self.fluid.rho = (
            np.ones(self.par.nogrid)
            * ICparams['nHini']
            * unyt.mp
        ).to(unyt.g / unyt.cm**3)
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.temp = np.ones(self.par.nogrid) * ICparams['tempini']
        self.fluid.xHI = np.ones(self.par.nogrid) * ICparams['xHIini']
        self.fluid.ngamma = np.ones(self.par.nogrid) * ICparams['ngammaini']
        self.fluid.mu = np.ones(self.par.nogrid) * ICparams['muini']


def interior_slice(sim):
    first = sim.par.noghost
    return slice(first, first + sim.par.nogrid)


def mean_temperature(sim):
    interior = interior_slice(sim)
    return np.mean(sim.fluid.temp[interior].to_value(unyt.K)) * unyt.K


def mean_neutral_fraction(sim):
    interior = interior_slice(sim)
    return float(np.mean(sim.fluid.xHI[interior]))


def mean_photon_number_density(sim):
    interior = interior_slice(sim)
    return (
        np.mean(sim.fluid.ngamma[interior].to_value(1.0 / unyt.cm**3))
        / unyt.cm**3
    )


def time_value(sim, units):
    return float(np.ravel(sim.fluid.time.to_value(units))[0])


def current_time(sim):
    return time_value(sim, unyt.s) * unyt.s


def set_source_state(sim):
    if current_time(sim) < source_switch_time:
        ngamma = photon_density_on
    else:
        ngamma = 0.0 / unyt.cm**3
    sim.fluid.ngamma[:] = ngamma.to(sim.fluid.ngamma.units)


def sample_times():
    switch_yr = source_switch_time.to_value(unyt.yr)
    final_yr = final_time.to_value(unyt.yr)
    early = np.logspace(-6.0, np.log10(switch_yr), 420)
    late = np.logspace(np.log10(switch_yr), np.log10(final_yr), 120)
    values = np.concatenate(([0.0], early, late, [switch_yr, final_yr]))
    values = values[np.logical_and(values >= 0.0, values <= final_yr)]
    return np.unique(values) * unyt.yr


def write_output(sim, outindex):
    sim.fluid.SetTemperature()
    sim.par.time = sim.fluid.time
    filename = (
        sim.par.outdir
        + '/'
        + sim.par.outfileprefix
        + '_%03d' % outindex
        + '.hdf5'
    )
    rio.writehdf5(sim, filename)
    return time_value(sim, unyt.s)


def append_history(sim, history):
    history['time_yr'].append(time_value(sim, unyt.yr))
    history['temperature_K'].append(mean_temperature(sim).to_value(unyt.K))
    history['xHI'].append(mean_neutral_fraction(sim))
    history['ngamma'].append(
        mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
    )


def advance_sources(sim, dt):
    set_source_state(sim)
    sim.solver.AddHydrogenSources(dt, sim.mesh, sim.fluid, sim.par)
    sim.solver.SetPrimitive(sim.mesh, sim.fluid)
    sim.fluid.time += dt
    sim.fluid.SetTemperature()


def save_history_plot(history, filename):
    time_yr = np.asarray(history['time_yr'])
    temperature_K = np.asarray(history['temperature_K'])
    plot_time_yr = np.maximum(time_yr, 1.0e-6)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        plot_time_yr,
        temperature_K,
        color='tab:red',
        lw=2.0,
        label='Temperature',
    )
    ax.axhline(
        photoionization_equilibrium_temperature.to_value(unyt.K),
        color='0.25',
        lw=1.2,
        ls=':',
        label=r'$T_{\rm ion}=6.33\,{\rm eV}/(3k_{\rm B})$',
    )
    ax.axhline(
        thermal_equilibrium_temperature.to_value(unyt.K),
        color='0.45',
        lw=1.2,
        ls='-.',
        label=r'$T_{\rm therm}\approx2T_{\rm ion}$',
    )
    ax.text(
        1.7e8,
        photoionization_equilibrium_temperature.to_value(unyt.K) * 1.04,
        r'$10^{4.39}\ {\rm K}$',
        color='0.25',
        va='bottom',
    )
    ax.text(
        1.7e8,
        thermal_equilibrium_temperature.to_value(unyt.K) * 1.04,
        r'$\approx2\times10^{4.39}\ {\rm K}$',
        color='0.45',
        va='bottom',
    )

    timescales = [
        (ionization_timescale.to_value(unyt.yr), r'$\tau_i=10^{-2.3}\ {\rm yr}$'),
        (
            recombination_timescale.to_value(unyt.yr),
            r'$\tau_r=10^{5.1}\ {\rm yr}$',
        ),
        (
            thermal_equilibrium_timescale.to_value(unyt.yr),
            r'$\tau_e=10^{9.3}\ {\rm yr}$',
        ),
    ]
    colors = ['tab:blue', 'tab:green', 'tab:purple']
    for (time_scale, label), color in zip(timescales, colors):
        ax.axvline(time_scale, color=color, lw=1.2, ls='--')
        ax.text(
            time_scale,
            0.97,
            label,
            color=color,
            rotation=90,
            va='top',
            ha='right',
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Temperature [K]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.0e-6, 4.0e9)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def main():
    ric = Simwrap()
    rio.writehdf5(ric, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    history = {'time_yr': [], 'temperature_K': [], 'xHI': [], 'ngamma': []}
    append_history(sim, history)
    write_output(sim, 0)

    output_interval = sim.par.outdeltatime.copy()
    next_output_time = output_interval.copy()
    outindex = 1
    last_output_time = time_value(sim, unyt.s)

    for target_time in sample_times()[1:]:
        while current_time(sim) < target_time:
            start_time = current_time(sim)
            dt = target_time - start_time
            if start_time < source_switch_time < start_time + dt:
                dt = source_switch_time - start_time
            advance_sources(sim, dt)
            if current_time(sim) >= next_output_time:
                last_output_time = write_output(sim, outindex)
                outindex += 1
                next_output_time += output_interval
        append_history(sim, history)

    if time_value(sim, unyt.s) != last_output_time:
        write_output(sim, outindex)

    figure_filename = rundir + '/HydrogenPhotoheating1D.jpg'
    save_history_plot(history, figure_filename)

    print('Hydrogen photoheating example finished')
    print('time = %.3e yr' % time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % mean_neutral_fraction(sim))
    print(
        'mean photon number density = %.3e cm^-3'
        % mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
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
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
