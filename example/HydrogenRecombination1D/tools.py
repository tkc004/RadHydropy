"""Helper utilities for the hydrogen recombination example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value, time_seconds
import hydrogen_recombination_analytic as hra


def build_initial_condition(config):
    """Build the initial state directly from the nested example mapping."""
    initial = config['initial_condition']
    units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    grid_cells = int(initial['grid_cells'])
    result = Rsim(config['par'])
    result.par.simulation.coordinate_system = initial['coordinate_system']
    result.par.simulation.time_proper_code = initial['current_time'].to_value(units.time_unit)
    result.par.simulation.box_size = initial['box_size'].to_value(units.length_unit)
    result.par.mesh.ghost_cells = 1
    result.mesh.boundary_proper_code = as_named_array(np.linspace(
        0.0, result.par.simulation.box_size, grid_cells + 1
    ))
    result.fluid.rho_proper_code = as_named_array(
        (np.ones(grid_cells) * initial['hydrogen_number_density'] * unyt.mp)
        .to_value(units.density_unit)
    )
    result.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells))
    result.fluid.temp_proper_code = as_named_array(
        (np.ones(grid_cells) * initial['temperature']).to_value(units.temperature_unit)
    )
    result.fluid.xHI = as_named_array(np.ones(grid_cells) * initial['neutral_fraction'])
    result.fluid.mu = as_named_array(np.ones(grid_cells) * initial['mean_molecular_weight'])
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    first, last = 1, 1 + grid_cells
    result.mesh.boundary_proper_code = as_named_array(
        result.mesh.boundary_proper_code[first:last + 1]
    )
    for field in ('rho_proper_code', 'vel_proper_code', 'temp_proper_code', 'xHI', 'mu'):
        setattr(result.fluid, field, as_named_array(getattr(result.fluid, field)[first:last]))
    result.par.mesh.ghost_cells = 0
    boundary = result.mesh.boundary_proper_code
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=0.5 * (boundary[1:] + boundary[:-1]),
        boundary=boundary,
        width=np.diff(boundary),
        area=np.ones(grid_cells) * quantity_to_value(result.par.mesh.area, units.area_unit),
        volume=np.ones(grid_cells) * quantity_to_value(result.par.mesh.area, units.area_unit) * np.diff(boundary),
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
    return (
        np.mean(
            code_quantity_to_cgs(
                sim.fluid.temp_proper_code[interior],
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


def time_value(sim, code_unit_system):
    code = getattr(sim.par.code_unit_system, 'CodeUnits', None)
    time_s = time_seconds(sim.fluid.time_proper_code, code)
    unit_seconds = float((1.0 * code_unit_system).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config):
    history = {'time_yr': [], 'temperature_cgs_K': [], 'ionized_fraction': []}
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


def save_history_plot(history, filename, config, target_neutral_fraction):
    initial = config['initial_condition']
    time_yr = np.asarray(history['time_yr'])
    ionized_fraction = np.asarray(history['ionized_fraction'])
    if time_yr.size > 1:
        dense_time_yr = np.linspace(time_yr.min(), time_yr.max(), 400)
    else:
        dense_time_yr = time_yr
    dense_analytic = hra.ionized_fraction(
        dense_time_yr,
        initial['neutral_fraction'],
        initial['temperature'],
        initial['hydrogen_number_density'],
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
