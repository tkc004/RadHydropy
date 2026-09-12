"""Helper utilities for the optically thin photoheating example."""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import time

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value, time_seconds
import hydrogen_photoheating_reference as hpr


start_time = time.time()


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    grid_cells = int(config['par']['mesh']['grid_cells'])
    writer = InitialConditionWriter(par_config=config['par'], code_units=code_units)
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial['hydrogen_number_density'] * unyt.mp
    )
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.simulation.par.simulation.time_proper_code = quantity_to_value(
        initial['time_proper'], code_units.time_unit
    )
    writer.simulation.fluid.xHI = as_named_array(np.full(grid_cells, initial['neutral_fraction']))
    writer.fluid.ngamma_radarray = writer.radarray(
        np.ones(grid_cells) * initial['photon_number_density']
    )
    writer.simulation.fluid.mu = as_named_array(
        np.full(grid_cells, initial['mean_molecular_weight'])
    )
    return writer
def reference_values(
    photon_flux_cgs_cm2_s,
    hydrogen_number_density_cgs_cm3_unyt,
    excess_photoionization_energy_cgs_erg,
    sigma_gamma_cgs_cm2,
    time_thermal_equilibrium_proper_unyt,
):
    photon_number_density_cgs_cm3_unyt = hpr.photon_number_density_from_flux(
        photon_flux_cgs_cm2_s
    )
    temperature_photoionization_cgs_K_unyt = (
        hpr.photoionization_equilibrium_temperature(
            excess_photoionization_energy_cgs_erg,
        )
    )
    temperature_thermal_equilibrium_cgs_K_unyt = hpr.thermal_equilibrium_temperature(
        temperature_photoionization_cgs_K_unyt,
    )
    time_ionization_proper_unyt = hpr.photoionization_timescale(
        sigma_gamma_cgs_cm2,
        photon_number_density_cgs_cm3_unyt,
    )
    time_recombination_proper_unyt = hpr.recombination_timescale_at_temperature(
        hydrogen_number_density_cgs_cm3_unyt,
        temperature_photoionization_cgs_K_unyt,
    )
    return {
        'photon_number_density_cgs_cm3_unyt': photon_number_density_cgs_cm3_unyt,
        'temperature_photoionization_cgs_K_unyt': temperature_photoionization_cgs_K_unyt,
        'temperature_thermal_equilibrium_cgs_K_unyt': temperature_thermal_equilibrium_cgs_K_unyt,
        'time_ionization_proper_unyt': time_ionization_proper_unyt,
        'time_recombination_proper_unyt': time_recombination_proper_unyt,
        'time_thermal_equilibrium_proper_unyt': time_thermal_equilibrium_proper_unyt,
        'hydrogen_number_density_cgs_cm3_unyt': hydrogen_number_density_cgs_cm3_unyt,
        'sigma_gamma_cgs_cm2': sigma_gamma_cgs_cm2,
    }


def interior_slice(sim):
    first = sim.par.mesh.ghost_cells
    return slice(first, first + sim.par.mesh.grid_cells)


def mean_temperature(sim):
    interior = interior_slice(sim)
    return np.mean(sim.fluid.temp_radarray[interior].to_cgs().value) * unyt.K


def mean_neutral_fraction(sim):
    interior = interior_slice(sim)
    return float(np.mean(sim.fluid.xHI[interior]))


def mean_photon_number_density(sim):
    interior = interior_slice(sim)
    return np.mean(sim.fluid.ngamma_radarray[interior].to_cgs().value) / unyt.cm**3


def time_value(sim, code_unit_system):
    code = getattr(sim.par.units, 'CodeUnits', None)
    time_s = time_seconds(sim.fluid.time_proper_code, code)
    unit_seconds = float((1.0 * code_unit_system).to_value(unyt.s))
    return float(time_s / unit_seconds)


def load_history_from_outputs(outputfiles, config):
    history = {'time_proper_yr': [], 'temperature_proper_cgs_K': [], 'xHI': [], 'ngamma_proper_cgs_cm3': []}
    initial = config['initial_condition']

    code_units_obj = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])

    for outfilename in sorted(outputfiles):
        rout = rio.loadhdf5(config, outfilename)
        first = int(rout.par.mesh.ghost_cells)
        interior = slice(first, first + int(rout.par.mesh.grid_cells))
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


def RunHydrogenPhotoheating(sim, source_switch_time, photon_density_on, outputtime=0):
    """Run the optically thin photoheating example with source switching."""
    print("--- Initization finished. Start running ... ---")
    print("--- %s seconds ---" % (time.time() - start_time))
    rio.write_numbered_hdf5(sim, 0)

    time_unit = sim.par.units.CodeUnits.time_unit

    def _as_time_quantity(value, unit=time_unit):
        if hasattr(value, 'to'):
            return value
        value_array = np.asarray(value, dtype=float)
        if value_array.shape == ():
            magnitude = float(value_array)
        else:
            magnitude = float(value_array.reshape(-1)[0])
        return magnitude * unit

    final_time = _as_time_quantity(sim.par.simulation.final_time)
    source_switch_time = _as_time_quantity(source_switch_time, final_time.units)
    sim.fluid.time_proper_code = float(sim.fluid.time_proper_code)
    output_times = rio.load_output_time_list(
        getattr(sim.par.output, 'time_list_filename', None)
    )
    if output_times is not None:
        target_unit = final_time.units
        output_times = np.unique(
            np.asarray(output_times.to_value(target_unit), dtype=float)
        )
        output_times = [
            value * target_unit
            for value in output_times
            if value * target_unit > sim.fluid.time_proper_code * time_unit and value * target_unit <= final_time
        ]
    else:
        output_interval = getattr(sim.par.output, 'cadence', None)
        next_output_time = (
            _as_time_quantity(output_interval, final_time.units)
            if output_interval is not None
            else None
        )
    last_output_time = sim.fluid.time_proper_code * time_unit
    outindex = 1
    next_output_index = 0

    while sim.fluid.time_proper_code * time_unit < final_time:
        time_proper = sim.fluid.time_proper_code * time_unit
        dt = final_time - time_proper
        if output_times is not None and next_output_index < len(output_times):
            target_output_time = output_times[next_output_index]
            if time_proper < target_output_time < time_proper + dt:
                dt = target_output_time - time_proper
        elif (
            next_output_time is not None
            and time_proper < next_output_time < time_proper + dt
        ):
            dt = next_output_time - time_proper
        if time_proper < source_switch_time < time_proper + dt:
            dt = source_switch_time - time_proper

        if time_proper < source_switch_time:
            ngamma_proper_cgs_cm3 = float(
                np.asarray(photon_density_on.to_value(1.0 / unyt.cm**3), dtype=float)
            )
        else:
            ngamma_proper_cgs_cm3 = 0.0
        sim.fluid.ngamma_code[:] = (
            ngamma_proper_cgs_cm3 * unyt.cm**-3
        ).to_value(sim.par.units.CodeUnits.number_density_unit)

        sim.solver.ApplyThermochemistryFast(dt, sim.mesh, sim.fluid, sim.par)
        sim.fluid.time_proper_code += dt.to_value(time_unit)

        if getattr(sim.par, 'verbose', 0) >= 1:
            print("time, dt", sim.fluid.time_proper_code * time_unit, dt)

        if output_times is not None:
            while (
                next_output_index < len(output_times)
                and sim.fluid.time_proper_code * time_unit >= output_times[next_output_index]
            ):
                rio.write_numbered_hdf5(sim, outindex)
                last_output_time = sim.fluid.time_proper_code * time_unit
                outindex += 1
                next_output_index += 1
        elif next_output_time is not None and sim.fluid.time_proper_code * time_unit >= next_output_time:
            rio.write_numbered_hdf5(sim, outindex)
            last_output_time = sim.fluid.time_proper_code * time_unit
            outindex += 1
            next_output_time += output_interval

    if sim.fluid.time_proper_code * time_unit != last_output_time:
        rio.write_numbered_hdf5(sim, outindex)

    print("--- Simulation finished. ---")
    print("--- %s seconds ---" % (time.time() - start_time))


def save_history_plot(history, filename, reference):
    time_proper_yr = np.asarray(history['time_proper_yr'])
    temperature_proper_cgs_K = np.asarray(history['temperature_proper_cgs_K'])
    xHI = np.maximum(np.asarray(history['xHI']), 1.0e-12)
    plot_time_yr = np.maximum(time_proper_yr, 1.0e-6)
    xHI_reference = hpr.neutral_fraction_reference(
        reference['hydrogen_number_density_cgs_cm3_unyt'],
        reference['sigma_gamma_cgs_cm2'],
        reference['photon_number_density_cgs_cm3_unyt'],
        reference['temperature_photoionization_cgs_K_unyt'],
    )
    xHI_reference_log = np.log10(xHI_reference['xHI'])

    fig, (ax_temp, ax_xHI) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.4),
        sharex=True,
        gridspec_kw={'height_ratios': [2.0, 1.0], 'hspace': 0.08},
    )
    ax_temp.plot(
        plot_time_yr,
        temperature_proper_cgs_K,
        color='tab:red',
        lw=2.0,
        label='Temperature',
    )
    ax_temp.axhline(
        reference['temperature_photoionization_cgs_K_unyt'].to_value(unyt.K),
        color='0.25',
        lw=1.2,
        ls=':',
        label=r'$T_{\rm ion}=6.33\,{\rm eV}/(3k_{\rm B})$',
    )
    ax_temp.axhline(
        reference['temperature_thermal_equilibrium_cgs_K_unyt'].to_value(unyt.K),
        color='0.45',
        lw=1.2,
        ls='-.',
        label=r'$T_{\rm therm}\approx2T_{\rm ion}$',
    )
    ax_temp.text(
        1.7e8,
        reference['temperature_photoionization_cgs_K_unyt'].to_value(unyt.K) * 1.04,
        r'$10^{4.39}\ {\rm K}$',
        color='0.25',
        va='bottom',
    )
    ax_temp.text(
        2.0e7,
        reference['temperature_thermal_equilibrium_cgs_K_unyt'].to_value(unyt.K) * 1.04,
        r'$\approx2\times10^{4.39}\ {\rm K}$',
        color='0.45',
        va='bottom',
    )

    ax_xHI.plot(
        plot_time_yr,
        xHI,
        color='tab:blue',
        lw=2.0,
        label='Neutral fraction',
    )
    ax_xHI.axhline(
        xHI_reference['xHI'],
        color='black',
        lw=1.2,
        ls=':',
        label=r'$x_{\rm HI}=\tau_i/\tau_r(T_{\rm ion})$',
    )
    ax_xHI.text(
        1.0e2,
        xHI_reference['xHI'] * 1.25,
        r'$\tau_i/\tau_r=10^{%.2f}$' % xHI_reference_log,
        color='black',
        va='bottom',
    )

    timescales = [
        (
            reference['time_ionization_proper_unyt'].to_value(unyt.yr),
            hpr.timescale_label('i', reference['time_ionization_proper_unyt']),
        ),
        (
            reference['time_recombination_proper_unyt'].to_value(unyt.yr),
            hpr.timescale_label('r', reference['time_recombination_proper_unyt']),
        ),
        (
            reference['time_thermal_equilibrium_proper_unyt'].to_value(unyt.yr),
            hpr.timescale_label(
                'e',
                reference['time_thermal_equilibrium_proper_unyt'],
            ),
        ),
    ]
    colors = ['tab:blue', 'tab:green', 'tab:purple']
    for (time_scale, label), color in zip(timescales, colors):
        ax_temp.axvline(time_scale, color=color, lw=1.2, ls='--')
        ax_xHI.axvline(time_scale, color=color, lw=1.0, ls='--', alpha=0.65)
        ax_temp.text(
            time_scale,
            0.97,
            label,
            color=color,
            rotation=90,
            va='top',
            ha='right',
            transform=ax_temp.get_xaxis_transform(),
        )

    ax_xHI.set_xlabel('Time [yr]')
    ax_temp.set_ylabel('Temperature [K]')
    ax_xHI.set_ylabel('Neutral fraction')
    ax_xHI.set_xscale('log')
    ax_temp.set_yscale('log')
    ax_xHI.set_yscale('log')
    ax_xHI.set_xlim(1.0e-6, 4.0e9)
    ax_temp.set_ylim(70.0, reference['temperature_thermal_equilibrium_cgs_K_unyt'].to_value(unyt.K) * 1.55)
    ax_xHI.set_ylim(1.0e-9, 1.5)
    ax_temp.grid(True, which='both', alpha=0.25)
    ax_xHI.grid(True, which='both', alpha=0.25)
    ax_temp.legend(frameon=False, loc='lower left')
    ax_xHI.legend(frameon=False, loc='lower left')
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.10,
        top=0.98,
        hspace=0.08,
    )
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return xHI_reference
