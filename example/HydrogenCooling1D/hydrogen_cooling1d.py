"""Uniform ionized hydrogen cooling box.

This example starts with a static, uniform box of pure ionized hydrogen at
``T = 1e6 K``. Hydrogen cooling and chemistry are enabled, and the run stops
once the mean temperature reaches roughly ``2e4 K``.
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
import radhydropy.io as rio


rundir = os.getcwd()

runparams = {
    'simname': 'HydrogenCooling1D',
    'ICfilename': rundir + '/InitialCondition.hdf5',
    'outdir': rundir,
    'outfileprefix': 'Output',
    'outdeltatime': 1.0e4 * unyt.yr,
    'savedir': rundir,
    'coordsys': 'cartesian',
    'EOStype': 'polytropic',
    'gamma': 5.0 / 3.0,
    'timesim': 1.0e6 * unyt.yr,
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
    'dtmax': 1.0e3 * unyt.yr,
    'hydrogen_chemistry': True,
    'hydrogen_mass_fraction': 1.0,
    'hydrogen_xHI_initial': 0.0,
    'hydrogen_xHI_inflow': 0.0,
    'hydrogen_xHI_outflow': 0.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_update_mu': True,
}

ICparams = {
    'nogrid': 16,
    'coordsys': 'cartesian',
    'boxsize': 1.0 * unyt.kpc,
    'time': 0.0 * unyt.yr,
    'nHini': 100.0 / unyt.cm**3,
    'tempini': 1.0e6 * unyt.K,
    'xHIini': 0.0,
}

target_temperature = 2.0e4 * unyt.K


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
        self.fluid.mu = np.ones(self.par.nogrid) * 0.5


def mean_temperature(sim):
    return np.mean(sim.fluid.temp.to_value(unyt.K)) * unyt.K


def mean_neutral_fraction(sim):
    return float(np.mean(sim.fluid.xHI))


def time_value(sim, units):
    return float(np.ravel(sim.fluid.time.to_value(units))[0])


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


def save_history_plot(history, filename):
    fig, ax_temp = plt.subplots(figsize=(7.0, 4.5))
    ax_xHI = ax_temp.twinx()

    ax_temp.plot(
        history['time_yr'],
        history['temperature_K'],
        color='tab:red',
        lw=2.0,
        label='Temperature',
    )
    ax_xHI.plot(
        history['time_yr'],
        history['xHI'],
        color='tab:blue',
        lw=2.0,
        label='Neutral fraction',
    )

    ax_temp.set_xlabel('Time [yr]')
    ax_temp.set_ylabel('Temperature [K]', color='tab:red')
    ax_xHI.set_ylabel('Neutral fraction', color='tab:blue')
    ax_temp.set_yscale('log')
    ax_xHI.set_ylim(0.0, 1.0)
    ax_temp.tick_params(axis='y', labelcolor='tab:red')
    ax_xHI.tick_params(axis='y', labelcolor='tab:blue')
    ax_temp.grid(True, which='both', alpha=0.25)
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

    history = {'time_yr': [], 'temperature_K': [], 'xHI': []}
    outindex = 0
    output_interval = sim.par.outdeltatime.copy()
    next_output_time = output_interval.copy()
    last_output_time = write_output(sim, outindex)
    append_history(sim, history)
    outindex += 1

    while (
        mean_temperature(sim) > target_temperature
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

    figure_filename = rundir + '/HydrogenCooling1D.jpg'
    save_history_plot(history, figure_filename)

    print('Hydrogen cooling example finished')
    print('time = %.3e yr' % time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % mean_neutral_fraction(sim))
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
