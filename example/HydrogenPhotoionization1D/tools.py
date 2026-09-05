"""Helper utilities for the fixed-field photoionization example."""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from types import SimpleNamespace

import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs, time_seconds
import hydrogen_photoionization_analytic as hpa


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    icparams = config['initial_condition']
    code_units = config['_code_units']
    sim = SimpleNamespace()
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    if code_units is not None:
        sim.par.unit_system = code_units.unit_system

    grid_cells = icparams['grid_cells']
    box_size = np.ones(1) * icparams['box_size']
    sim.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
    sim.par.simulation = SimpleNamespace(
        coordinate_system=icparams['coordinate_system'],
        current_time=np.ones(1) * icparams['current_time'],
        box_size=box_size,
    )

    sim.mesh.boundary = np.linspace(
        0.0 * box_size[0], box_size[0], grid_cells + 1,
    )

    sim.fluid.rho_code = (
        np.ones(grid_cells)
        * icparams['hydrogen_number_density']
        * unyt.mp
    ).to(unyt.g / unyt.cm**3)
    sim.fluid.vel_code = np.zeros(grid_cells) * unyt.cm / unyt.s
    sim.fluid.temp_code = np.ones(grid_cells) * icparams['temperature']
    sim.fluid.xHI = np.ones(grid_cells) * icparams['neutral_fraction']
    sim.fluid.ngamma_code = np.ones(grid_cells) * icparams['photon_number_density']
    sim.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']


    return sim

def interior_slice(sim):
    first = sim.par.mesh.ghost_cells
    return slice(first, first + sim.par.mesh.grid_cells)


def mean_temperature(sim):
    interior = interior_slice(sim)
    code_units_obj = getattr(sim.par.units, 'CodeUnits', None)
    temp_values = code_quantity_to_cgs(
        sim.fluid.temp_code[interior],
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
    time_s = time_seconds(sim.fluid.time_code, code)
    unit_seconds = float((1.0 * units).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config):
    history = {'time_yr': [], 'temperature_cgs_K': [], 'xHI': [], 'ngamma_cgs_cm3': []}
    icparams = config['initial_condition']
    runparams = config['par']
    interior = slice(0, icparams['grid_cells'])
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])

    for outfilename in sorted(outputfiles):
        nested_config = dict(config)
        nested_config['_code_units'] = code_units_obj
        rout = build_initial_condition(nested_config)
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
    icparams = config['initial_condition']
    runparams = config['par']
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
            icparams['neutral_fraction'],
            icparams['temperature'],
            icparams['hydrogen_number_density'],
            icparams['photon_number_density'],
            runparams['radiation']['hydrogen_sigma_gamma'],
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



