"""Helper utilities for the fixed-field photoionization example."""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value, time_seconds
import hydrogen_photoionization_analytic as hpa


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    result = Rsim(config['par'])
    grid_cells = int(initial['grid_cells'])
    result.par.simulation.coordinate_system = initial['coordinate_system']
    result.par.simulation.time_proper_code = initial['current_time'].to_value(code_units.time_unit)
    result.par.simulation.box_size = initial['box_size'].to_value(code_units.length_unit)
    result.par.mesh.ghost_cells = 1
    result.mesh.boundary_proper_code = as_named_array(np.linspace(
        0.0, result.par.simulation.box_size, grid_cells + 1
    ))
    result.fluid.rho_proper_code = as_named_array(
        (np.ones(grid_cells) * initial['hydrogen_number_density'] * unyt.mp)
        .to_value(code_units.density_unit)
    )
    result.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells))
    result.fluid.temp_proper_code = as_named_array(
        (np.ones(grid_cells) * initial['temperature']).to_value(code_units.temperature_unit)
    )
    result.fluid.xHI = as_named_array(np.ones(grid_cells) * initial['neutral_fraction'])
    result.fluid.ngamma_code = as_named_array(
        (np.ones(grid_cells) * initial['photon_number_density']).to_value(
            code_units.number_density_unit
        )
    )
    result.fluid.mu = as_named_array(np.ones(grid_cells) * initial['mean_molecular_weight'])
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    first, last = 1, 1 + grid_cells
    result.mesh.boundary_proper_code = as_named_array(
        result.mesh.boundary_proper_code[first:last + 1]
    )
    for field in ('rho_proper_code', 'vel_proper_code', 'temp_proper_code', 'xHI', 'mu', 'ngamma_code'):
        setattr(result.fluid, field, as_named_array(getattr(result.fluid, field)[first:last]))
    result.par.mesh.ghost_cells = 0
    boundary = result.mesh.boundary_proper_code
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=0.5 * (boundary[1:] + boundary[:-1]),
        boundary=boundary,
        width=np.diff(boundary),
        area=np.ones(grid_cells) * quantity_to_value(result.par.mesh.area, code_units.area_unit),
        volume=np.ones(grid_cells) * quantity_to_value(result.par.mesh.area, code_units.area_unit) * np.diff(boundary),
    )
    result.fluid.SetPressure()
    result.fluid.SetFluidTime(result.par.simulation.time_proper_code)
    result.fluid.SetEnergyDensity()
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    return result

def interior_slice(sim):
    first = sim.par.mesh.ghost_cells
    return slice(first, first + sim.par.mesh.grid_cells)


def mean_temperature(sim):
    interior = interior_slice(sim)
    code_units_obj = getattr(sim.par.units, 'CodeUnits', None)
    temp_values = code_quantity_to_cgs(
        sim.fluid.temp_proper_code[interior],
        code_units_obj,
        'temperature_cgs_K',
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
                sim.fluid.ngamma_code[interior],
                getattr(sim.par.units, 'CodeUnits', None),
                'number_density_cgs_cm3',
            )
        )
        / unyt.cm**3
    )


def time_value(sim, units):
    code = getattr(sim.par.units, 'CodeUnits', None)
    time_s = time_seconds(sim.fluid.time_proper_code, code)
    unit_seconds = float((1.0 * units).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config):
    history = {'time_yr': [], 'temperature_cgs_K': [], 'xHI': [], 'ngamma_cgs_cm3': []}
    initial = config['initial_condition']
    par_config = config['par']
    interior = slice(0, initial['grid_cells'])
    code_units_obj = CodeUnits.from_mapping(par_config['units']['CodeUnits'])

    for outfilename in sorted(outputfiles):
        rout = Rsim(config['par'])
        rout.par.unit_system = code_units_obj.unit_system
        rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
        history['time_yr'].append(time_value(rout, unyt.yr))
        history['temperature_cgs_K'].append(
            np.mean(
                code_quantity_to_cgs(
                rout.fluid.temp_proper_code[interior],
                    code_units_obj,
                    'temperature_cgs_K',
                )
            )
        )
        history['xHI'].append(float(np.mean(rout.fluid.xHI[interior])))
        history['ngamma_cgs_cm3'].append(
            np.mean(
                code_quantity_to_cgs(
                rout.fluid.ngamma_code[interior],
                    code_units_obj,
                    'number_density_cgs_cm3',
                )
            )
        )

    return history


def output_files(outdir, outfileprefix):
    return sorted(glob.glob(outdir + '/' + outfileprefix + '_*.hdf5'))


def save_history_plot(history, filename, config, target_xHI):
    initial = config['initial_condition']
    par_config = config['par']
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
        initial['neutral_fraction'],
        initial['temperature'],
        initial['hydrogen_number_density'],
        initial['photon_number_density'],
        par_config['radiation']['hydrogen_sigma_gamma'],
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
