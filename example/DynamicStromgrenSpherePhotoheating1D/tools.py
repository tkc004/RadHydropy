"""Utilities for the dynamic photoheated Stromgren sphere example."""

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import example_utils as eu

import radhydropy.io as rio
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value
from radhydropy.rsim import Rsim
from basic_hydro_utils import make_initial_condition

IONIZATION_FRONT_NEUTRAL_FRACTION = 0.5


def _to_kpc(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.kpc), dtype=float)
    code = getattr(getattr(par, 'units', None), 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm),
        dtype=float,
    )


def _to_myr(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.Myr), dtype=float)
    code = getattr(getattr(par, 'units', None), 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'time_s') / (1.0 * unyt.Myr).to_value(unyt.s),
        dtype=float,
    )


def _to_km_s(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.km / unyt.s), dtype=float)
    code = getattr(getattr(par, 'units', None), 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'velocity_cgs_cm_s') / (1.0 * unyt.km).to_value(unyt.cm),
        dtype=float,
    )


def _to_number_density(values, par):
    if hasattr(values, 'to_value'):
        density_proper_cgs_g_cm3 = np.asarray(values.to_value(unyt.g / unyt.cm**3), dtype=float)
        return density_proper_cgs_g_cm3 / (1.0 * unyt.mp).to_value(unyt.g)
    code = getattr(getattr(par, 'units', None), 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'density_cgs_g_cm3') / (1.0 * unyt.mp).to_value(unyt.g),
        dtype=float,
    )


def _to_pressure(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.g / unyt.cm / unyt.s**2), dtype=float)
    code = getattr(getattr(par, 'units', None), 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(code_quantity_to_cgs(values, code, 'pressure_cgs_erg_cm3'), dtype=float)


def _to_temperature(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.K), dtype=float)
    code = getattr(getattr(par, 'units', None), 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(code_quantity_to_cgs(values, code, 'temperature_cgs_K'), dtype=float)


def _attach_proper_runtime_states(par, mesh, fluid):
    """Attach unitless proper-code geometry and fluid states for HDF5."""
    boundary_proper_code = np.asarray(mesh.boundary_proper_code, dtype=float)
    width_proper_code = np.diff(boundary_proper_code)
    area_proper_code = 4.0 * np.pi * boundary_proper_code[:-1] ** 2
    volume_proper_code = (
        4.0 * np.pi / 3.0
        * np.abs(boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    )
    coordinate_proper_code = 0.5 * (
        boundary_proper_code[1:] + boundary_proper_code[:-1]
    )
    denominator = boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
    valid = denominator != 0.0
    coordinate_proper_code[valid] = 0.75 * (
        boundary_proper_code[1:][valid] ** 4
        - boundary_proper_code[:-1][valid] ** 4
    ) / denominator[valid]
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=coordinate_proper_code,
        boundary_proper_code=boundary_proper_code,
        width_proper_code=width_proper_code,
        area_proper_code=area_proper_code,
        volume_proper_code=volume_proper_code,
    )
    fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    fluid.SetPressure()
    fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=fluid.rho_proper_code,
        vel_proper_code=fluid.vel_proper_code,
        pre_proper_code=fluid.pre_proper_code,
        temp_proper_code=fluid.temp_proper_code,
        time_proper_code=fluid.time_proper_code,
        mu_dimensionless=fluid.mu,
        xHI_dimensionless=fluid.xHI if hasattr(fluid, 'xHI') else None,
    )


def build_static_problem(config):
    """Build the proper-code IC using the configured ``Rsim`` object."""

    initial = config['initial_condition']
    units = config['_code_units']
    grid_cells = int(config["par"]['mesh']['grid_cells'])
    box_size_proper_code = quantity_to_value(
        initial['box_size_proper'], units.length_unit
    )
    boundary_proper_code = np.linspace(0.0, box_size_proper_code, grid_cells + 1)
    density_proper_code = np.full(
        grid_cells,
        quantity_to_value(
            initial['hydrogen_number_density'] * unyt.mp,
            units.density_unit,
        ),
    )
    temperature_proper_code = np.full(
        grid_cells,
        quantity_to_value(initial['temperature_proper'], units.temperature_unit),
    )
    sim = make_initial_condition(
        config, boundary_proper_code=boundary_proper_code,
        rho_proper_code=density_proper_code,
        vel_proper_code=np.zeros(grid_cells),
        temp_proper_code=temperature_proper_code,
        mu_dimensionless=np.ones(grid_cells),
    )
    radiation = config["par"]['radiation']
    sim.fluid.xHI = np.ones(grid_cells)
    sim.fluid.ngamma_code = np.full(
        grid_cells,
        quantity_to_value(radiation['hydrogen_ngamma_initial'], units.number_density_unit),
    )
    sim.fluid.SetFluidTime(0.0)
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=sim.fluid.rho_proper_code,
        vel_proper_code=sim.fluid.vel_proper_code,
        pre_proper_code=sim.fluid.pre_proper_code,
        temp_proper_code=sim.fluid.temp_proper_code,
        time_proper_code=sim.fluid.time_proper_code,
        mu_dimensionless=sim.fluid.mu,
        xHI_dimensionless=sim.fluid.xHI,
    )
    return sim

build_problem = build_static_problem


def write_initial_condition(config):
    """Build and write the initial-condition snapshot."""
    sim = build_static_problem(config)
    filename = config['par']['simulation']['initial_condition_filename']
    Path(filename).unlink(missing_ok=True)
    rio.writehdf5(sim, filename)


def load_output_state(outputfilename, config):
    sim = Rsim(config['par'])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, outputfilename)
    # Output snapshots contain the canonical ghosted boundary array.  Rebuild
    # the typed mesh geometry from its physical boundaries for diagnostics and
    # plotting; the runtime runner performs this step during RunAll.
    ghost_cells = int(sim.par.mesh.ghost_cells)
    grid_cells = int(sim.par.mesh.grid_cells)
    boundary_proper_code = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    if boundary_proper_code.size == grid_cells + 1 + 2 * ghost_cells:
        sim.mesh.boundary_proper_code = boundary_proper_code[ghost_cells:-ghost_cells]
        sim.SetMesh()
    sim.fluid.SetPressure()
    return sim.par, sim.mesh, sim.fluid


def output_files(outdir, outfileprefix):
    pattern = os.path.join(outdir, f'{outfileprefix}_*.hdf5')
    filenames = []
    for filename in glob.glob(pattern):
        stem = Path(filename).stem
        suffix = stem[len(outfileprefix) + 1:]
        if suffix.isdigit():
            filenames.append(filename)
    return sorted(filenames)


def interior_slice(par):
    first = int(par.mesh.ghost_cells)
    return slice(first, first + int(par.mesh.grid_cells))


def ionization_front_position(
    mesh,
    fluid,
    config,
    neutral_fraction=IONIZATION_FRONT_NEUTRAL_FRACTION,
):
    par = config['_output_par']
    interior = interior_slice(par)
    radius_proper_kpc = _to_kpc(mesh.x_proper_code[interior], par)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)

    if np.all(xHI > neutral_fraction):
        return 0.0
    if np.all(xHI <= neutral_fraction):
        return radius_proper_kpc[-1]

    # The front is the first outward transition from ionized gas
    # (xHI <= 0.5) to neutral gas (xHI > 0.5). This remains well-defined
    # when the profile contains additional ionized pockets farther out.
    crossings = np.where(
        (xHI[:-1] <= neutral_fraction) & (xHI[1:] > neutral_fraction)
    )[0]
    if crossings.size == 0:
        return radius_proper_kpc[np.where(xHI <= neutral_fraction)[0][-1]]

    left = int(crossings[0])
    right = left + 1
    weight = (neutral_fraction - xHI[left]) / (xHI[right] - xHI[left])
    return radius_proper_kpc[left] + weight * (radius_proper_kpc[right] - radius_proper_kpc[left])


def mean_ionized_temperature(fluid, config):
    par = config['_output_par']
    interior = interior_slice(par)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)
    temperature_proper_cgs_K = _to_temperature(fluid.temp_proper_code[interior], par)
    ionized_weight = 1.0 - xHI
    if np.sum(ionized_weight) <= 0.0:
        return 0.0
    return float(np.sum(ionized_weight * temperature_proper_cgs_K) / np.sum(ionized_weight))


def append_history(history, mesh, fluid, config):
    par = config['_output_par']
    history['time_Myr'].append(_to_myr(fluid.time_proper_code, par))
    history['front_radius_kpc'].append(
        ionization_front_position(
            mesh,
            fluid,
            config,
            neutral_fraction=IONIZATION_FRONT_NEUTRAL_FRACTION,
        )
    )
    history['mean_ionized_temperature_cgs_K'].append(mean_ionized_temperature(fluid, config))


def load_history_from_outputs(outputfilenames, config):
    history = {
        'time_Myr': [],
        'front_radius_kpc': [],
        'mean_ionized_temperature_cgs_K': [],
    }
    for outputfilename in outputfilenames:
        par, mesh, fluid = load_output_state(outputfilename, config)
        config['_output_par'] = par
        append_history(history, mesh, fluid, config)
    return history


def stromgren_radius(config):
    initial = config['initial_condition']
    chemistry = config['par']['chemistry']
    radiation = config['par']['radiation']
    radius_proper_kpc = (
        3.0
        * radiation['source_photon_rate']
        / (
            4.0
            * np.pi
            * chemistry['hydrogen_alpha_B']
            * initial['hydrogen_number_density']**2
        )
    ) ** (1.0 / 3.0)
    return radius_proper_kpc.to(unyt.kpc)


def recombination_time(config):
    initial = config['initial_condition']
    alpha_B = config['par']['chemistry']['hydrogen_alpha_B']
    return (
        1.0 / (initial['hydrogen_number_density'] * alpha_B)
    ).to(unyt.Myr)


def ionized_sound_speed(gamma):
    """Return the Spitzer ionized-gas sound speed at 10^4 K."""
    temperature_proper_cgs_K = 1.0e4 * unyt.K
    mu_ionized = 0.5
    return np.sqrt(gamma * unyt.kboltz * temperature_proper_cgs_K / (mu_ionized * unyt.mp)).to(
        unyt.km / unyt.s
    )


def spitzer_radius(time_proper_code, config, ci):
    radius_stromgren = stromgren_radius(config)
    factor = (
        1.0
        + 7.0
        * ci.to(unyt.cm / unyt.s)
        * time_proper_code.to(unyt.s)
        / (4.0 * radius_stromgren.to(unyt.cm))
    )
    return (radius_stromgren * factor**(4.0 / 7.0)).to(unyt.kpc)


def shifted_spitzer_radius(time_proper_code, config, ci):
    time_since_recombination = time_proper_code - recombination_time(config)
    return spitzer_radius(time_since_recombination, config, ci)


def load_reference_profile(filename, radius_unit, log_value=False):
    if filename is None or not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    value = data[:, 1]
    if log_value:
        value = 10.0**value
    return {
        'radius_kpc': data[:, 0] * radius_unit.to_value(unyt.kpc),
        'value': value,
    }


def scatter_reference(ax, reference, label='ZEUS-MP'):
    if reference is None:
        return
    ax.scatter(
        reference['radius_kpc'],
        reference['value'],
        s=20,
        color='black',
        marker='o',
        facecolors='none',
        label=label,
    )


def save_front_plot(history, config, figure_filename):
    example = config.get('example', {})
    time_proper_Myr = np.asarray(history['time_Myr']) * unyt.Myr
    front_radius = np.asarray(history['front_radius_kpc'])
    radius_stromgren = stromgren_radius(config)
    tau_recombination = recombination_time(config)
    ci = ionized_sound_speed(5.0 / 3.0)
    spitzer_valid = time_proper_Myr >= tau_recombination
    radius_spitzer = None
    if np.any(spitzer_valid):
        radius_spitzer = shifted_spitzer_radius(
            time_proper_Myr[spitzer_valid],
            config,
            ci,
        ).to_value(unyt.kpc)
    plot_radius_max = example['plot_radius_max'].to_value(unyt.kpc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time_proper_Myr.to_value(unyt.Myr),
        front_radius,
        color='tab:blue',
        lw=2.0,
        label=r'RadHydropy $x_{\rm HI}=0.5$',
    )
    if radius_spitzer is not None:
        ax.plot(
            time_proper_Myr[spitzer_valid].to_value(unyt.Myr),
            radius_spitzer,
            color='black',
            lw=1.7,
            ls='--',
            label=(
                r'Spitzer after $\tau_{\rm rec}$, '
                r'$c_i=%.1f$ km s$^{-1}$'
                % ci.to_value(unyt.km / unyt.s)
            ),
        )
    ax.axvline(
        tau_recombination.to_value(unyt.Myr),
        color='0.45',
        lw=1.2,
        ls='-.',
        label=r'$\tau_{\rm rec}=%.1f$ Myr' % tau_recombination.to_value(unyt.Myr),
    )
    ax.axhline(
        radius_stromgren.to_value(unyt.kpc),
        color='0.3',
        lw=1.4,
        ls=':',
        label=r'$R_{\rm S}$',
    )
    ax.set_xlim(0.0, time_proper_Myr[-1].to_value(unyt.Myr))
    if radius_spitzer is not None:
        ax.set_ylim(0.0, max(plot_radius_max, 1.05 * np.nanmax(radius_spitzer)))
    else:
        ax.set_ylim(0.0, plot_radius_max)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Ionization-front radius [kpc]')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_plot(mesh, fluid, config, figure_filename):
    par = config['_output_par']
    example = config.get('example', {})
    interior = interior_slice(par)
    radius_pc = _to_kpc(mesh.x_proper_code[interior], par) * (1.0 * unyt.kpc).to_value(unyt.pc)
    number_density = _to_number_density(fluid.rho_proper_code[interior], par)
    vel_peculiar_proper_km_s = _to_km_s(fluid.vel_proper_code[interior], par)
    neutral_fraction = np.asarray(fluid.xHI[interior], dtype=float)
    pressure_proper_cgs_erg_cm3 = _to_pressure(fluid.pre_proper_code[interior], par)
    temperature_proper_cgs_K = _to_temperature(fluid.temp_proper_code[interior], par)
    plot_radius_max = example['plot_radius_max'].to_value(unyt.pc)
    radius_unit = example.get('reference_radius_unit', 15.0 * unyt.kpc)
    density_reference = load_reference_profile(
        example.get('density_reference_filename', None),
        radius_unit,
        log_value=True,
    )
    velocity_reference = load_reference_profile(
        example.get('velocity_reference_filename', None),
        radius_unit,
        log_value=False,
    )
    pressure_reference = load_reference_profile(
        example.get('pressure_reference_filename', None),
        radius_unit,
        log_value=True,
    )
    neutral_fraction_reference = load_reference_profile(
        example.get('neutral_fraction_reference_filename', None),
        radius_unit,
        log_value=True,
    )

    reference_radius_scale = (1.0 * unyt.kpc).to_value(unyt.pc)
    for reference in (
        density_reference,
        velocity_reference,
        pressure_reference,
        neutral_fraction_reference,
    ):
        if reference is not None:
            reference['radius_kpc'] *= reference_radius_scale

    fig, axes = plt.subplots(5, 1, figsize=(7.4, 11.0), sharex=True)
    axes[0].plot(radius_pc, number_density, color='tab:blue', lw=1.8, label='RadHydropy')
    scatter_reference(axes[0], density_reference)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$n$ [cm$^{-3}$]')
    axes[0].legend(frameon=False, loc='best')

    positive_velocity = np.where(vel_peculiar_proper_km_s > 0.0, vel_peculiar_proper_km_s, np.nan)
    axes[1].plot(radius_pc, positive_velocity, color='tab:orange', lw=1.8, label='RadHydropy')
    scatter_reference(axes[1], velocity_reference)
    axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=0.5)
    axes[1].set_ylabel(r'$v_r$ [km s$^{-1}$]')
    axes[1].legend(frameon=False, loc='best')

    axes[2].plot(
        radius_pc,
        np.clip(neutral_fraction, 1.0e-8, 1.0),
        color='tab:green',
        lw=1.8,
        label='RadHydropy',
    )
    scatter_reference(axes[2], neutral_fraction_reference)
    axes[2].set_yscale('log')
    axes[2].set_ylabel(r'$x_{\rm HI}$')
    axes[2].legend(frameon=False, loc='best')

    axes[3].plot(radius_pc, pressure_proper_cgs_erg_cm3, color='tab:red', lw=1.8, label='RadHydropy')
    scatter_reference(axes[3], pressure_reference)
    axes[3].set_yscale('log')
    axes[3].set_ylabel(r'$P$ [g cm$^{-1}$ s$^{-2}$]')
    axes[3].legend(frameon=False, loc='best')

    axes[4].plot(radius_pc, temperature_proper_cgs_K, color='tab:purple', lw=1.8, label='RadHydropy')
    axes[4].set_yscale('log')
    axes[4].set_ylabel(r'$T$ [K]')
    axes[4].set_xlabel('Radius [pc]')
    axes[4].legend(frameon=False, loc='best')

    for ax in axes:
        ax.set_xlim(0.0, plot_radius_max)
        ax.grid(True, which='both', alpha=0.25)
    final_time_myr = config['par']['simulation']['final_time'].to_value(unyt.Myr)
    fig.suptitle(
        'Dynamic photoheated Stromgren sphere at %.3g Myr' % final_time_myr
    )
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
