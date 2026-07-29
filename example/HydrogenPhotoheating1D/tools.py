"""Helper utilities for the optically thin photoheating example."""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import time

import radhydropy.io as rio
import hydrogen_photoheating_reference as hpr


start_time = time.time()


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


def reference_values(
    photon_flux,
    hydrogen_number_density,
    excess_photoionization_energy,
    sigma_gamma,
    thermal_equilibrium_timescale,
):
    photon_density_on = hpr.photon_number_density_from_flux(photon_flux)
    photoionization_temperature = (
        hpr.photoionization_equilibrium_temperature(
            excess_photoionization_energy,
        )
    )
    thermal_temperature = hpr.thermal_equilibrium_temperature(
        photoionization_temperature,
    )
    ionization_timescale = hpr.photoionization_timescale(
        sigma_gamma,
        photon_density_on,
    )
    recombination_timescale = hpr.recombination_timescale_at_temperature(
        hydrogen_number_density,
        photoionization_temperature,
    )
    return {
        'photon_density_on': photon_density_on,
        'photoionization_temperature': photoionization_temperature,
        'thermal_temperature': thermal_temperature,
        'ionization_timescale': ionization_timescale,
        'recombination_timescale': recombination_timescale,
        'thermal_equilibrium_timescale': thermal_equilibrium_timescale,
        'hydrogen_number_density': hydrogen_number_density,
        'sigma_gamma': sigma_gamma,
    }


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


def load_history_from_outputs(outputfiles, icparams, noghost):
    history = {'time_yr': [], 'temperature_K': [], 'xHI': [], 'ngamma': []}
    interior = slice(noghost, noghost + icparams['nogrid'])

    for outfilename in sorted(outputfiles):
        rout = Simwrap(icparams)
        rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
        history['time_yr'].append(time_value(rout, unyt.yr))
        history['temperature_K'].append(
            np.mean(rout.fluid.temp[interior].to_value(unyt.K))
        )
        history['xHI'].append(float(np.mean(rout.fluid.xHI[interior])))
        history['ngamma'].append(
            np.mean(rout.fluid.ngamma[interior].to_value(1.0 / unyt.cm**3))
        )

    return history


def output_files(outdir, outfileprefix):
    return sorted(glob.glob(outdir + '/' + outfileprefix + '_*.hdf5'))


def RunHydrogenPhotoheating(sim, source_switch_time, photon_density_on, outputtime=0):
    """Run the optically thin photoheating example with source switching."""
    print("--- Initization finished. Start running ... ---")
    print("--- %s seconds ---" % (time.time() - start_time))
    sim._write_numbered_hdf5(0)

    final_time = sim.par.timesim
    output_times = sim._load_output_time_list()
    if output_times is not None:
        target_unit = final_time.units
        output_times = np.unique(
            np.asarray(output_times.to_value(target_unit), dtype=float)
        )
        output_times = [
            value * target_unit
            for value in output_times
            if value * target_unit > sim.fluid.time and value * target_unit <= final_time
        ]
    else:
        output_interval = getattr(sim.par, 'outdeltatime', None)
        next_output_time = (
            output_interval.copy() if output_interval is not None else None
        )
    last_output_time = sim.fluid.time.copy()
    outindex = 1
    next_output_index = 0

    while sim.fluid.time < final_time:
        current_time = sim.fluid.time
        dt = final_time - current_time
        if output_times is not None and next_output_index < len(output_times):
            target_output_time = output_times[next_output_index]
            if current_time < target_output_time < current_time + dt:
                dt = target_output_time - current_time
        elif (
            next_output_time is not None
            and current_time < next_output_time < current_time + dt
        ):
            dt = next_output_time - current_time
        if current_time < source_switch_time < current_time + dt:
            dt = source_switch_time - current_time

        if current_time < source_switch_time:
            ngamma = photon_density_on
        else:
            ngamma = 0.0 * sim.fluid.ngamma.units
        sim.fluid.ngamma[:] = ngamma.to(sim.fluid.ngamma.units)

        sim.solver.ApplyThermochemistryFast(dt, sim.mesh, sim.fluid, sim.par)
        sim.fluid.time += dt

        if outputtime == 1:
            print("time, dt", sim.fluid.time, dt)

        if output_times is not None:
            while (
                next_output_index < len(output_times)
                and sim.fluid.time >= output_times[next_output_index]
            ):
                sim._write_numbered_hdf5(outindex)
                last_output_time = sim.fluid.time.copy()
                outindex += 1
                next_output_index += 1
        elif next_output_time is not None and sim.fluid.time >= next_output_time:
            sim._write_numbered_hdf5(outindex)
            last_output_time = sim.fluid.time.copy()
            outindex += 1
            next_output_time += output_interval

    if sim.fluid.time != last_output_time:
        sim._write_numbered_hdf5(outindex)

    print("--- Simulation finished. ---")
    print("--- %s seconds ---" % (time.time() - start_time))


def save_history_plot(history, filename, reference):
    time_yr = np.asarray(history['time_yr'])
    temperature_K = np.asarray(history['temperature_K'])
    xHI = np.maximum(np.asarray(history['xHI']), 1.0e-12)
    plot_time_yr = np.maximum(time_yr, 1.0e-6)
    xHI_reference = hpr.neutral_fraction_reference(
        reference['hydrogen_number_density'],
        reference['sigma_gamma'],
        reference['photon_density_on'],
        reference['photoionization_temperature'],
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
        temperature_K,
        color='tab:red',
        lw=2.0,
        label='Temperature',
    )
    ax_temp.axhline(
        reference['photoionization_temperature'].to_value(unyt.K),
        color='0.25',
        lw=1.2,
        ls=':',
        label=r'$T_{\rm ion}=6.33\,{\rm eV}/(3k_{\rm B})$',
    )
    ax_temp.axhline(
        reference['thermal_temperature'].to_value(unyt.K),
        color='0.45',
        lw=1.2,
        ls='-.',
        label=r'$T_{\rm therm}\approx2T_{\rm ion}$',
    )
    ax_temp.text(
        1.7e8,
        reference['photoionization_temperature'].to_value(unyt.K) * 1.04,
        r'$10^{4.39}\ {\rm K}$',
        color='0.25',
        va='bottom',
    )
    ax_temp.text(
        2.0e7,
        reference['thermal_temperature'].to_value(unyt.K) * 1.04,
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
            reference['ionization_timescale'].to_value(unyt.yr),
            hpr.timescale_label('i', reference['ionization_timescale']),
        ),
        (
            reference['recombination_timescale'].to_value(unyt.yr),
            hpr.timescale_label('r', reference['recombination_timescale']),
        ),
        (
            reference['thermal_equilibrium_timescale'].to_value(unyt.yr),
            hpr.timescale_label(
                'e',
                reference['thermal_equilibrium_timescale'],
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
    ax_temp.set_ylim(70.0, reference['thermal_temperature'].to_value(unyt.K) * 1.55)
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
