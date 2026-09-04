"""Helper utilities for the hydrogen recombination example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import unyt
from types import SimpleNamespace

import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs, time_seconds
import hydrogen_recombination_analytic as hra


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        if code_units is not None:
            self.par.unit_system = code_units.unit_system

        grid_cells = icparams['grid_cells']
        box_size = np.ones(1) * icparams['box_size']
        self.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordinate_system'],
            current_time=np.ones(1) * icparams['current_time'],
            box_size=box_size,
        )

        self.mesh.boundary = np.linspace(
            0.0 * box_size[0], box_size[0], grid_cells + 1,
        )

        self.fluid.rho_code = (
            np.ones(grid_cells)
            * icparams['hydrogen_number_density']
            * unyt.mp
        ).to(unyt.g / unyt.cm**3)
        self.fluid.vel_code = np.zeros(grid_cells) * unyt.cm / unyt.s
        self.fluid.temp_code = np.ones(grid_cells) * icparams['temperature']
        self.fluid.xHI = np.ones(grid_cells) * icparams['neutral_fraction']
        self.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']


def interior_slice(sim):
    first = sim.par.mesh.ghost_cells
    return slice(first, first + sim.par.mesh.grid_cells)


def mean_temperature(sim):
    interior = interior_slice(sim)
    return (
        np.mean(
            code_quantity_to_cgs(
                sim.fluid.temp_code[interior],
                getattr(sim.par.units, 'CodeUnits', None),
                'temperature_cgs_K',
            )
        )
        * unyt.K
    )


def mean_neutral_fraction(sim):
    interior = interior_slice(sim)
    return float(np.mean(sim.fluid.xHI[interior]))


def mean_ionized_fraction(sim):
    return 1.0 - mean_neutral_fraction(sim)


def time_value(sim, units):
    code = getattr(sim.par.units, 'CodeUnits', None)
    time_s = time_seconds(sim.fluid.time, code)
    unit_seconds = float((1.0 * units).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config):
    history = {'time_yr': [], 'temperature_cgs_K': [], 'ionized_fraction': []}
    icparams = config['initial_condition']
    runparams = config['par']
    interior = slice(0, icparams['grid_cells'])
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])

    for outfilename in sorted(outputfiles):
        rout = Simwrap(icparams, code_units=code_units_obj)
        rout.par.unit_system = code_units_obj.unit_system
        rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
        history['time_yr'].append(time_value(rout, unyt.yr))
        history['temperature_cgs_K'].append(
            np.mean(
                code_quantity_to_cgs(
                    rout.fluid.temp_code[interior],
                    code_units_obj,
                    'temperature_cgs_K',
                )
            )
        )
        history['ionized_fraction'].append(
            1.0 - float(np.mean(rout.fluid.xHI[interior]))
        )

    return history


def output_files(outdir, outfileprefix):
    pattern = outdir + '/' + outfileprefix + '_*.hdf5'
    return glob.glob(pattern)


def run_hydrogen_recombination(sim, target_neutral_fraction, outputtime=0):
    return sim.RunAll(
        outputtime=outputtime,
        mode='sources',
        stop_condition=lambda runner: (
            mean_neutral_fraction(runner) >= target_neutral_fraction
        ),
    )


def save_history_plot(history, filename, icparams, target_neutral_fraction):
    time_yr = np.asarray(history['time_yr'])
    ionized_fraction = np.asarray(history['ionized_fraction'])
    if time_yr.size > 1:
        dense_time_yr = np.linspace(time_yr.min(), time_yr.max(), 400)
    else:
        dense_time_yr = time_yr
    dense_analytic = hra.ionized_fraction(
        dense_time_yr,
        icparams['neutral_fraction'],
        icparams['temperature'],
        icparams['hydrogen_number_density'],
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
        dense_time_yr,
        dense_analytic,
        color='black',
        lw=2.2,
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
