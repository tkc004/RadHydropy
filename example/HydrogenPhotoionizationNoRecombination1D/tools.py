"""Helper utilities for fixed-field photoionization without recombination."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.io as rio
import hydrogen_photoionization_no_recombination_analytic as hpna


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
        self.fluid.ngamma = np.ones(self.par.nogrid) * icparams['ngammaini']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']


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


def save_history_plot(history, filename, icparams, runparams, target_xHI):
    time_yr = np.asarray(history['time_yr'])
    xHI = np.asarray(history['xHI'])
    analytic = hpna.neutral_fraction(
        time_yr,
        icparams['xHIini'],
        icparams['ngammaini'],
        runparams['hydrogen_sigma_gamma'],
    )

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
        label='Exponential analytic',
    )
    ax.axhline(target_xHI, color='tab:red', lw=1.0, ls='--')

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Neutral fraction')
    ax.set_yscale('log')
    ax.set_ylim(2.0e-3, 1.2)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
