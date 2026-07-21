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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.rsim import Rsim
import radhydropy.hydrogen as rh
import radhydropy.io as rio


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


def recombination_rate():
    alpha_B = rh.alpha_B(ICparams['tempini']).to(unyt.cm**3 / unyt.s)
    nH = ICparams['nHini'].to(1.0 / unyt.cm**3)
    return (alpha_B * nH).to(1.0 / unyt.s)


def photoionization_rate():
    return rh.photoionization_frequency(
        ICparams['ngammaini'],
        runparams['hydrogen_sigma_gamma'],
    )


def analytic_neutral_fraction(time_yr):
    time = np.asarray(time_yr) * unyt.yr
    rate_rec = recombination_rate().to_value(1.0 / unyt.s)
    rate_photo = photoionization_rate().to_value(1.0 / unyt.s)
    time_s = time.to_value(unyt.s)
    x0 = ICparams['xHIini']

    if rate_rec == 0.0:
        return x0 * np.exp(-rate_photo * time_s)

    discriminant = np.sqrt(rate_photo**2 + 4.0 * rate_rec * rate_photo)
    root_low = (
        rate_photo + 2.0 * rate_rec - discriminant
    ) / (2.0 * rate_rec)
    root_high = (
        rate_photo + 2.0 * rate_rec + discriminant
    ) / (2.0 * rate_rec)
    ratio_initial = (x0 - root_low) / (x0 - root_high)
    ratio = ratio_initial * np.exp(-discriminant * time_s)
    return (root_low - ratio * root_high) / (1.0 - ratio)


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


def save_history_plot(history, filename):
    time_yr = np.asarray(history['time_yr'])
    xHI = np.asarray(history['xHI'])
    analytic = analytic_neutral_fraction(time_yr)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        time_yr,
        xHI,
        color='tab:blue',
        marker='o',
        ms=3.0,
        lw=0.0,
        label='RadHydropy',
    )
    ax.plot(
        time_yr,
        analytic,
        color='black',
        lw=2.0,
        label='Fixed-field analytic',
    )
    ax.axhline(
        target_neutral_fraction,
        color='tab:red',
        lw=1.0,
        ls='--',
    )

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Neutral fraction')
    ax.set_yscale('log')
    ax.set_ylim(2.0e-3, 1.2)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
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
    outindex = 0
    output_interval = sim.par.outdeltatime.copy()
    next_output_time = output_interval.copy()
    last_output_time = write_output(sim, outindex)
    append_history(sim, history)
    outindex += 1

    while (
        mean_neutral_fraction(sim) > target_neutral_fraction
        and time_value(sim, unyt.s) < float(sim.par.timesim.to_value(unyt.s))
    ):
        sim.RunOneStep()
        append_history(sim, history)
        if sim.fluid.time >= next_output_time:
            last_output_time = write_output(sim, outindex)
            outindex += 1
            next_output_time += output_interval

    if time_value(sim, unyt.s) != last_output_time:
        write_output(sim, outindex)

    figure_filename = rundir + '/HydrogenPhotoionization1D.jpg'
    save_history_plot(history, figure_filename)

    print('Hydrogen photoionization example finished')
    print('time = %.3e yr' % time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % mean_neutral_fraction(sim))
    print(
        'mean photon number density = %.3e cm^-3'
        % mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
    )
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
