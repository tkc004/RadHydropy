"""Helper utilities for the hydrogen cooling example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.io as rio


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']

        self.mesh.boundary = np.linspace(
            0.0,
            1.0,
            self.par.nogrid + 1,
        ) * icparams['boxsize']

        self.fluid.rho = (
            np.ones(self.par.nogrid)
            * icparams['nHini']
            * unyt.mp
        ).to(unyt.g / unyt.cm**3)
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['tempini']
        self.fluid.xHI = np.ones(self.par.nogrid) * icparams['xHIini']
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
