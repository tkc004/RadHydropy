"""Helper utilities for the fixed-field photoionization example."""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs, time_seconds
import hydrogen_photoionization_analytic as hpa


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
    code_units = getattr(sim.par, 'CodeUnits', None)
    temp_values = code_quantity_to_cgs(
        sim.fluid.temp[interior],
        code_units,
        'temperature_K',
    )
    return np.mean(temp_values) * unyt.K


def mean_neutral_fraction(sim):
    interior = interior_slice(sim)
    return float(np.mean(sim.fluid.xHI[interior]))


def mean_photon_number_density(sim):
    interior = interior_slice(sim)
    return (
        np.mean(
            code_quantity_to_cgs(
                sim.fluid.ngamma[interior],
                getattr(sim.par, 'CodeUnits', None),
                'number_density_cm3',
            )
        )
        / unyt.cm**3
    )


def time_value(sim, units):
    code = getattr(sim.par, 'CodeUnits', None)
    time_s = time_seconds(sim.fluid.time, code)
    unit_seconds = float((1.0 * units).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config, noghost):
    history = {'time_yr': [], 'temperature_K': [], 'xHI': [], 'ngamma': []}
    interior = slice(noghost, noghost + config['nogrid'])
    code_units = CodeUnits.from_mapping(config.get('CodeUnits'))

    for outfilename in sorted(outputfiles):
        rout = Simwrap(config)
        rout.par.CodeUnits = code_units
        rout.par.unit_system = code_units.unit_system
        rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
        history['time_yr'].append(time_value(rout, unyt.yr))
        history['temperature_K'].append(
            np.mean(
                code_quantity_to_cgs(
                    rout.fluid.temp[interior],
                    code_units,
                    'temperature_K',
                )
            )
        )
        history['xHI'].append(float(np.mean(rout.fluid.xHI[interior])))
        history['ngamma'].append(
            np.mean(
                code_quantity_to_cgs(
                    rout.fluid.ngamma[interior],
                    code_units,
                    'number_density_cm3',
                )
            )
        )

    return history


def output_files(outdir, outfileprefix):
    return sorted(glob.glob(outdir + '/' + outfileprefix + '_*.hdf5'))


def save_history_plot(history, filename, icparams, runparams, target_xHI):
    time_yr = np.asarray(history['time_yr'])
    xHI = np.asarray(history['xHI'])
    positive_time_yr = time_yr[time_yr > 0.0]
    if positive_time_yr.size > 0:
        dense_time_yr = np.logspace(
            np.log10(max(positive_time_yr.min() * 0.1, 1.0e-6)),
            np.log10(positive_time_yr.max()),
            400,
        )
    else:
        dense_time_yr = np.maximum(time_yr, 1.0e-6)
    analytic = hpa.neutral_fraction(
        dense_time_yr,
        icparams['xHIini'],
        icparams['tempini'],
        icparams['nHini'],
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
        dense_time_yr,
        analytic,
        color='black',
        lw=2.0,
        label='Fixed-field analytic',
    )
    ax.axhline(target_xHI, color='tab:red', lw=1.0, ls='--')

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Neutral fraction')
    ax.set_yscale('log')
    lower_ylim = min(analytic.min(), np.min(xHI), target_xHI) * 0.2
    ax.set_ylim(max(lower_ylim, 1.0e-6), 1.2)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
