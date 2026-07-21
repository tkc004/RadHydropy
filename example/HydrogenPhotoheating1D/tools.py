"""Helper utilities for the optically thin photoheating example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.io as rio
import hydrogen_photoheating_reference as hpr


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


def current_time(sim):
    return time_value(sim, unyt.s) * unyt.s


def set_source_state(sim, source_switch_time, photon_density_on):
    if current_time(sim) < source_switch_time:
        ngamma = photon_density_on
    else:
        ngamma = 0.0 / unyt.cm**3
    sim.fluid.ngamma[:] = ngamma.to(sim.fluid.ngamma.units)


def sample_times(source_switch_time, final_time):
    switch_yr = source_switch_time.to_value(unyt.yr)
    final_yr = final_time.to_value(unyt.yr)
    early = np.logspace(-6.0, np.log10(switch_yr), 420)
    late = np.logspace(np.log10(switch_yr), np.log10(final_yr), 120)
    values = np.concatenate(([0.0], early, late, [switch_yr, final_yr]))
    values = values[np.logical_and(values >= 0.0, values <= final_yr)]
    return np.unique(values) * unyt.yr


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


def append_history(sim, history):
    history['time_yr'].append(time_value(sim, unyt.yr))
    history['temperature_K'].append(mean_temperature(sim).to_value(unyt.K))
    history['xHI'].append(mean_neutral_fraction(sim))
    history['ngamma'].append(
        mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
    )


def advance_sources(sim, dt, source_switch_time, photon_density_on):
    set_source_state(sim, source_switch_time, photon_density_on)
    sim.solver.AddHydrogenSources(dt, sim.mesh, sim.fluid, sim.par)
    sim.solver.SetPrimitive(sim.mesh, sim.fluid)
    sim.fluid.time += dt
    sim.fluid.SetTemperature()


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
