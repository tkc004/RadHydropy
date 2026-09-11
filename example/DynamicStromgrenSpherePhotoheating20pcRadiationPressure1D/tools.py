"""Helpers for the 20 pc radiation-pressure Stromgren example."""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from DynamicStromgrenSpherePhotoheating20pc1D.tools import (
    _pressure_from_radarrays, _to_km_s, _to_kpc, _to_number_density,
    _to_pressure, _to_temperature,
    interior_slice, ionization_front_position, load_history_from_outputs,
    load_output_state,
    load_reference_profile, output_files, save_front_plot, scatter_reference,
    stromgren_radius, write_initial_condition,
)


def save_plot(mesh, fluid, config, figure_filename):
    """Save the inherited profile plot with a linear velocity axis."""
    par = config['_output_par']
    interior = interior_slice(config)
    radius_proper_pc = _to_kpc(
        0.5 * (mesh.boundary_radarray[:-1] + mesh.boundary_radarray[1:])[interior],
        config,
    ) * (1.0 * unyt.kpc).to_value(unyt.pc)
    number_density_cgs_cm3 = _to_number_density(fluid.rho_radarray[interior], config)
    vel_peculiar_proper_km_s = _to_km_s(fluid.vel_radarray[interior], config)
    neutral_fraction = np.asarray(fluid.xHI[interior], dtype=float)
    pre_proper_cgs_erg_cm3 = _pressure_from_radarrays(fluid, config)[interior]
    temperature_proper_cgs_K = _to_temperature(fluid.temp_radarray[interior], config)
    example_config = config['example']
    plot_radius_max = example_config['plot_radius_max'].to_value(unyt.pc)
    radius_unit = example_config.get('reference_radius_unit', 15.0 * unyt.kpc)
    density_reference = load_reference_profile(
        config.get('density_reference_filename', None), radius_unit, log_value=True
    )
    velocity_reference = load_reference_profile(
        config.get('velocity_reference_filename', None), radius_unit, log_value=False
    )
    pressure_reference = load_reference_profile(
        config.get('pressure_reference_filename', None), radius_unit, log_value=True
    )
    neutral_fraction_reference = load_reference_profile(
        config.get('neutral_fraction_reference_filename', None), radius_unit, log_value=True
    )
    reference_radius_scale = (1.0 * unyt.kpc).to_value(unyt.pc)
    for reference in (
        density_reference,
        velocity_reference,
        pressure_reference,
        neutral_fraction_reference,
    ):
        if reference is not None:
            reference['radius_proper_kpc'] *= reference_radius_scale

    fig, axes = plt.subplots(5, 1, figsize=(7.4, 11.0), sharex=True)
    axes[0].plot(radius_proper_pc, number_density_cgs_cm3, color='tab:blue', lw=1.8, label='RadHydropy')
    scatter_reference(axes[0], density_reference)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$n$ [cm$^{-3}$]')
    axes[0].legend(frameon=False, loc='best')

    axes[1].plot(radius_proper_pc, vel_peculiar_proper_km_s, color='tab:orange', lw=1.8, label='RadHydropy')
    scatter_reference(axes[1], velocity_reference)
    axes[1].set_yscale('linear')
    axes[1].set_ylabel(r'$v_r$ [km s$^{-1}$]')
    axes[1].legend(frameon=False, loc='best')

    axes[2].plot(
        radius_proper_pc,
        np.clip(neutral_fraction, 1.0e-8, 1.0),
        color='tab:green',
        lw=1.8,
        label='RadHydropy',
    )
    scatter_reference(axes[2], neutral_fraction_reference)
    axes[2].set_yscale('log')
    axes[2].set_ylabel(r'$x_{\rm HI}$')
    axes[2].legend(frameon=False, loc='best')

    axes[3].plot(radius_proper_pc, pre_proper_cgs_erg_cm3, color='tab:red', lw=1.8, label='RadHydropy')
    scatter_reference(axes[3], pressure_reference)
    axes[3].set_yscale('log')
    axes[3].set_ylabel(r'$P$ [g cm$^{-1}$ s$^{-2}$]')
    axes[3].legend(frameon=False, loc='best')

    axes[4].plot(radius_proper_pc, temperature_proper_cgs_K, color='tab:purple', lw=1.8, label='RadHydropy')
    axes[4].set_yscale('log')
    axes[4].set_ylabel(r'$T$ [K]')
    axes[4].set_xlabel('Radius [pc]')
    axes[4].legend(frameon=False, loc='best')

    for ax in axes:
        ax.set_xlim(0.0, plot_radius_max)
        ax.grid(True, which='both', alpha=0.25)
    final_time_myr = config['par']['simulation']['final_time'].to_value(unyt.Myr)
    fig.suptitle('Dynamic photoheated Stromgren sphere at %.3g Myr' % final_time_myr)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
