"""Helper utilities for the hydrogen recombination example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value, time_seconds
import hydrogen_recombination_analytic as hra


def build_initial_condition(config):
    """Build the initial state directly from the nested example mapping."""
    initial = config['initial_condition']
    units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    grid_cells = int(config['par']['mesh']['grid_cells'])
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(
        np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial['hydrogen_number_density'] * unyt.mp
    )
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.simulation.fluid.xHI = as_named_array(np.full(grid_cells, initial['neutral_fraction']))
    writer.fluid.mu = as_named_array(
        np.full(grid_cells, initial['mean_molecular_weight'])
    )
    return writer


def interior_slice(sim):
    first = sim.par.mesh.ghost_cells
    return slice(first, first + sim.par.mesh.grid_cells)


def mean_temperature(sim):
    interior = interior_slice(sim)
    return np.mean(sim.fluid.temp_radarray[interior].to_cgs().value) * unyt.K


def mean_neutral_fraction(sim):
    interior = interior_slice(sim)
    return float(np.mean(sim.fluid.xHI[interior]))


def mean_ionized_fraction(sim):
    return 1.0 - mean_neutral_fraction(sim)


def time_value(sim, code_unit_system):
    code = getattr(sim.par.units, 'CodeUnits', None)
    time_s = time_seconds(sim.fluid.time_proper_code, code)
    unit_seconds = float((1.0 * code_unit_system).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config):
    history = {'time_proper_yr': [], 'temperature_proper_cgs_K': [], 'ionized_fraction': []}
    initial = config['initial_condition']

    interior = slice(0, config['par']['mesh']['grid_cells'])
    code_units_obj = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])

    for outfilename in sorted(outputfiles):
        rout = rio.loadhdf5(config, outfilename)
        history['time_proper_yr'].append(time_value(rout, unyt.yr))
        history['temperature_proper_cgs_K'].append(
            np.mean(
                rout.fluid.temp_radarray[interior].to_cgs().value
            )
        )
        history['ionized_fraction'].append(
            1.0 - float(np.mean(rout.fluid.xHI[interior]))
        )

    return history


def output_files(output_directory, output_filename_prefix):
    pattern = output_directory + '/' + output_filename_prefix + '_*.hdf5'
    return glob.glob(pattern)


def run_hydrogen_recombination(sim, target_neutral_fraction, outputtime=0):
    return sim.RunAll(
        outputtime=outputtime,
        mode='sources',
        stop_condition=lambda runner: (
            mean_neutral_fraction(runner) >= target_neutral_fraction
        ),
    )


def save_history_plot(history, filename, config, target_neutral_fraction):
    initial = config['initial_condition']
    time_proper_yr = np.asarray(history['time_proper_yr'])
    ionized_fraction = np.asarray(history['ionized_fraction'])
    if time_proper_yr.size > 1:
        dense_time_yr = np.linspace(time_proper_yr.min(), time_proper_yr.max(), 400)
    else:
        dense_time_yr = time_proper_yr
    dense_analytic = hra.ionized_fraction_dimensionless(
        dense_time_yr,
        initial['neutral_fraction'],
        initial['temperature_proper'],
        initial['hydrogen_number_density'],
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        time_proper_yr,
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
