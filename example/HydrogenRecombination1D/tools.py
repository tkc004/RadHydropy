"""Helper utilities for the hydrogen recombination example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import unyt

import radhydropy.io as rio
import hydrogen_recombination_analytic as hra


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


def mean_ionized_fraction(sim):
    return 1.0 - mean_neutral_fraction(sim)


def time_value(sim, units):
    return float(np.ravel(sim.fluid.time.to_value(units))[0])


def _interior_slice(noghost, nogrid):
    return slice(noghost, noghost + nogrid)


def load_history_from_outputs(outputfiles, icparams, noghost):
    history = {'time_yr': [], 'temperature_K': [], 'ionized_fraction': []}
    interior = _interior_slice(noghost, icparams['nogrid'])

    for outfilename in sorted(outputfiles):
        rout = Simwrap(icparams)
        rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
        history['time_yr'].append(time_value(rout, unyt.yr))
        history['temperature_K'].append(
            np.mean(rout.fluid.temp[interior].to_value(unyt.K))
        )
        history['ionized_fraction'].append(
            1.0 - float(np.mean(rout.fluid.xHI[interior]))
        )

    return history


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


def output_files(outdir, outfileprefix):
    pattern = outdir + '/' + outfileprefix + '_*.hdf5'
    return glob.glob(pattern)


def append_history(sim, history):
    history['time_yr'].append(time_value(sim, unyt.yr))
    history['temperature_K'].append(mean_temperature(sim).to_value(unyt.K))
    history['ionized_fraction'].append(mean_ionized_fraction(sim))


def save_history_plot(history, filename, icparams, target_neutral_fraction):
    time_yr = np.asarray(history['time_yr'])
    ionized_fraction = np.asarray(history['ionized_fraction'])
    analytic = hra.ionized_fraction(
        time_yr,
        icparams['xHIini'],
        icparams['tempini'],
        icparams['nHini'],
    )

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
