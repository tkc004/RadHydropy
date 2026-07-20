"""Fixed-temperature case-B hydrogen recombination box.

The gas starts fully ionized at ``T = 2e4 K``. Hydrogen cooling/heating terms
and collisional ionization are disabled, leaving pure case-B recombination.
The run stops once the gas is 99 percent neutral and writes a JPG comparing
the ionized fraction against the analytic case-B expectation.
"""

import os
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
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
    'simname': 'HydrogenRecombination1D',
    'ICfilename': rundir + '/InitialCondition.hdf5',
    'outdir': rundir,
    'outfileprefix': 'Output',
    'outdeltatime': 5.0e4 * unyt.yr,
    'savedir': rundir,
    'coordsys': 'cartesian',
    'EOStype': 'polytropic',
    'gamma': 5.0 / 3.0,
    'timesim': 5.0e5 * unyt.yr,
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
    'dtmax': 2.0e3 * unyt.yr,
    'hydrogen_chemistry': True,
    'hydrogen_mass_fraction': 1.0,
    'hydrogen_xHI_initial': 0.0,
    'hydrogen_xHI_inflow': 0.0,
    'hydrogen_xHI_outflow': 0.0,
    'hydrogen_source_CFL': 0.2,
    'hydrogen_update_mu': False,
    'hydrogen_thermal_coupling': False,
    'hydrogen_collisional_ionization': False,
}

ICparams = {
    'nogrid': 16,
    'coordsys': 'cartesian',
    'boxsize': 1.0 * unyt.kpc,
    'time': 0.0 * unyt.yr,
    'nHini': 100.0 / unyt.cm**3,
    'tempini': 2.0e4 * unyt.K,
    'xHIini': 0.0,
    'muini': 0.5,
}

target_neutral_fraction = 0.99


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


def mean_ionized_fraction(sim):
    return 1.0 - mean_neutral_fraction(sim)


def time_value(sim, units):
    return float(np.ravel(sim.fluid.time.to_value(units))[0])


def recombination_rate():
    alpha_B = rh.alpha_B(ICparams['tempini']).to_value(unyt.cm**3 / unyt.s)
    nH = ICparams['nHini'].to_value(1.0 / unyt.cm**3)
    return alpha_B * nH / unyt.s


def analytic_ionized_fraction(time_yr):
    time = np.asarray(time_yr) * unyt.yr
    rate_time = (recombination_rate() * time).value
    y0 = 1.0 - ICparams['xHIini']
    return y0 / (1.0 + y0 * rate_time)


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
    history['ionized_fraction'].append(mean_ionized_fraction(sim))


def save_history_plot(history, filename):
    time_yr = np.asarray(history['time_yr'])
    ionized_fraction = np.asarray(history['ionized_fraction'])
    analytic = analytic_ionized_fraction(time_yr)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        time_yr,
        ionized_fraction,
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
        label='Case-B analytic',
    )
    ax.axhline(
        1.0 - target_neutral_fraction,
        color='tab:red',
        lw=1.0,
        ls='--',
    )

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Ionized fraction')
    ax.set_yscale('log')
    ax.set_ylim(7.0e-3, 1.2)
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

    history = {'time_yr': [], 'temperature_K': [], 'ionized_fraction': []}
    outindex = 0
    output_interval = sim.par.outdeltatime.copy()
    next_output_time = output_interval.copy()
    last_output_time = write_output(sim, outindex)
    append_history(sim, history)
    outindex += 1

    while (
        mean_neutral_fraction(sim) < target_neutral_fraction
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

    figure_filename = rundir + '/HydrogenRecombination1D.jpg'
    save_history_plot(history, figure_filename)

    print('Hydrogen recombination example finished')
    print('time = %.3e yr' % time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % mean_neutral_fraction(sim))
    print('mean ionized fraction = %.3e' % mean_ionized_fraction(sim))
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
