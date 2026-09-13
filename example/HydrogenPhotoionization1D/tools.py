"""Helper utilities for the fixed-field photoionization example."""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value, time_seconds
import hydrogen_photoionization_analytic as hpa


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    grid_cells = int(config['par']['mesh']['grid_cells'])
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(
        np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial['hydrogen_number_density'] * unyt.mp
    )
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.simulation.fluid.xHI = as_named_array(np.full(grid_cells, initial['neutral_fraction']))
    writer.fluid.ngamma_radarray = writer.radarray(
        np.ones(grid_cells) * initial['photon_number_density']
    )
    writer.fluid.mu = as_named_array(
        np.full(grid_cells, initial['mean_molecular_weight'])
    )
    return writer

def interior_slice(sim):
    first = sim.par.mesh.ghost_cells
    return slice(first, first + sim.par.mesh.grid_cells)


def mean_temperature(sim):
    interior = interior_slice(sim)
    temp_values = sim.fluid.temp_radarray[interior].to_cgs().value
    return np.mean(temp_values) * unyt.K


def mean_neutral_fraction(sim):
    interior = interior_slice(sim)
    return float(np.mean(sim.fluid.xHI[interior]))


def mean_photon_number_density(sim):
    interior = interior_slice(sim)
    return (
        np.mean(sim.fluid.ngamma_radarray[interior].to_cgs().value)
        / unyt.cm**3
    )


def time_value(sim, code_unit_system):
    code = getattr(sim.par.units, 'CodeUnits', None)
    time_s = time_seconds(sim.fluid.time_proper_code, code)
    unit_seconds = float((1.0 * code_unit_system).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config):
    history = {'time_proper_yr': [], 'temperature_proper_cgs_K': [], 'xHI': [], 'ngamma_proper_cgs_cm3': []}
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
        history['xHI'].append(float(np.mean(rout.fluid.xHI[interior])))
        history['ngamma_proper_cgs_cm3'].append(
            np.mean(
                rout.fluid.ngamma_radarray[interior].to_cgs().value
            )
        )

    return history


def output_files(output_directory, output_filename_prefix):
    return sorted(glob.glob(output_directory + '/' + output_filename_prefix + '_*.hdf5'))


def save_history_plot(history, filename, config, target_xHI):
    initial = config['initial_condition']

    time_proper_yr = np.asarray(history['time_proper_yr'])
    xHI = np.asarray(history['xHI'])
    positive_time_yr = time_proper_yr[time_proper_yr > 0.0]
    if positive_time_yr.size > 0:
        dense_time_yr = np.logspace(
            np.log10(max(positive_time_yr.min() * 0.1, 1.0e-6)),
            np.log10(positive_time_yr.max()),
            400,
        )
    else:
        dense_time_yr = np.maximum(time_proper_yr, 1.0e-6)
    analytic = hpa.neutral_fraction(
        dense_time_yr,
        initial['neutral_fraction'],
        initial['temperature_proper'],
        initial['hydrogen_number_density'],
        initial['photon_number_density'],
        config["par"]['radiation']['hydrogen_sigma_gamma'],
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        time_proper_yr,
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
