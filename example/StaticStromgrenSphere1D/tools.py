"""Helper utilities for the static Stromgren sphere example."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.thermo_networks.hydrogen as rth
import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.units import code_quantity_to_cgs, quantity_to_value
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)
import stromgren_analytic as sa


def _attach_proper_runtime_states(mesh, fluid):
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
    """Build the initial state using the canonical nested runtime objects."""

    initial = config['initial_condition']
    grid_cells = int(config["par"]['mesh']['grid_cells'])
    sim = Rsim(config["par"])
    code_units = sim.par.units.CodeUnits
    sim.par.simulation.box_size_proper_code = float(
        quantity_to_value(initial['box_size_proper'], code_units.length_unit)
    )
    sim.par.simulation.time_proper_code = 0.0
    sim.mesh.boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(0.0, initial['box_size_proper'].to_value(unyt.cm), grid_cells + 1) * unyt.cm,
        code_units.length_unit,
    ))
    boundary_proper_code = sim.mesh.boundary_proper_code
    width = np.diff(boundary_proper_code)
    volume_proper_code = 4.0 * np.pi / 3.0 * (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    x_proper_code = 0.75 * (boundary_proper_code[1:] ** 4 - boundary_proper_code[:-1] ** 4) / (
        boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
    )
    area_proper_code = 4.0 * np.pi * boundary_proper_code[:-1] ** 2
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=x_proper_code,
        boundary_proper_code=boundary_proper_code,
        width_proper_code=width,
        area_proper_code=area_proper_code,
        volume_proper_code=volume_proper_code,
    )
    sim.fluid.rho_proper_code = as_named_array(quantity_to_value((
        np.ones(grid_cells)
        * initial['hydrogen_number_density']
        * unyt.mp
    ).to(unyt.g / unyt.cm**3), code_units.density_unit))
    sim.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells, dtype=float))
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * 1.0e4 * unyt.K, code_units.temperature_unit
    ))
    sim.fluid.mu = np.ones(grid_cells)
    sim.fluid.xHI = np.ones(grid_cells)
    sim.fluid.ngamma_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * sim.par.radiation.hydrogen_ngamma_initial,
        code_units.number_density_unit,
    ))
    sim.fluid.SetFluidTime(0.0)
    _attach_proper_runtime_states(sim.mesh, sim.fluid)
    return sim.par, sim.mesh, sim.fluid, sim.solver


def _refresh_mesh_geometry(mesh, par):
    """Recompute derived mesh geometry from an already ghosted boundary."""
    mesh.width_proper_code = mesh.boundary_proper_code[1:] - mesh.boundary_proper_code[:-1]
    mesh.coordinate_inverse_proper_code = 1.0 / mesh.width_proper_code
    code_units = par.units.CodeUnits
    if par.simulation.coordinate_system == 'cartesian':
        mesh.x_proper_code = 0.5 * (mesh.boundary_proper_code[1:] + mesh.boundary_proper_code[:-1])
        mesh.area_proper_code = np.ones(len(mesh.width_proper_code)) * quantity_to_value(
            par.mesh.area_proper, code_units.area_unit
        )
        mesh.volume_proper_code = mesh.width_proper_code * mesh.area_proper_code
    elif par.simulation.coordinate_system == 'spherical':
        mesh.area_proper_code = (mesh.boundary_proper_code[:-1] ** 2) * 4.0 * np.pi
        mesh.volume_proper_code = np.absolute(
            mesh.boundary_proper_code[1:] ** 3 - mesh.boundary_proper_code[:-1] ** 3
        ) * 4.0 * np.pi / 3.0
        vol_denom = mesh.boundary_proper_code[1:] ** 3 - mesh.boundary_proper_code[:-1] ** 3
        mesh.x_proper_code = 0.5 * (mesh.boundary_proper_code[1:] + mesh.boundary_proper_code[:-1])
        nonzero_vol_denom = vol_denom != 0.0
        mesh.x_proper_code[nonzero_vol_denom] = 0.75 * (
            mesh.boundary_proper_code[1:][nonzero_vol_denom] ** 4 - mesh.boundary_proper_code[:-1][nonzero_vol_denom] ** 4
        ) / vol_denom[nonzero_vol_denom]
        for ig in range(len(mesh.volume_proper_code)):
            if (mesh.boundary_proper_code[ig] < 0.0) and (mesh.boundary_proper_code[ig + 1] > 0.0):
                mesh.volume_proper_code[ig] = (mesh.boundary_proper_code[ig + 1] ** 3) * 4.0 * np.pi / 3.0
                mesh.x_proper_code[ig] = 0.75 * mesh.boundary_proper_code[ig + 1]
                mesh.area_proper_code[ig] = 0.0
    else:
        raise ValueError("coordinate system unknown: %s" % par.simulation.coordinate_system)
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=mesh.x_proper_code,
        boundary_proper_code=mesh.boundary_proper_code,
        width_proper_code=mesh.width_proper_code,
        area_proper_code=mesh.area_proper_code,
        volume_proper_code=mesh.volume_proper_code,
    )


def write_initial_condition(config):
    """Build the raw IC state and write it to ``ICfilename``."""
    par, mesh, fluid, solver = build_static_problem(config)
    filename = config['par']['simulation']['initial_condition_filename']
    rio.writehdf5(Rsim.FromComponents(par, mesh, fluid, solver), filename)


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_static_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    code_units_obj = par.units.CodeUnits
    par.time_proper_code = float(np.asarray(par.time_proper_code, dtype=float))
    par.simulation.box_size_proper_code = float(np.asarray(par.box_size_proper_code, dtype=float))
    mesh.boundary_proper_code = np.asarray(mesh.boundary_proper_code, dtype=float)
    fluid.rho_proper_code = np.asarray(fluid.rho_proper_code, dtype=float)
    fluid.vel_proper_code = np.asarray(fluid.vel_proper_code, dtype=float)
    fluid.temp_proper_code = np.asarray(fluid.temp_proper_code, dtype=float)
    fluid.time_proper_code = par.time_proper_code
    if hasattr(fluid, 'ngamma_code'):
        fluid.ngamma_code = np.asarray(fluid.ngamma_code, dtype=float)
    _refresh_mesh_geometry(mesh, par)
    fluid.SetPressure()
    fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=fluid.rho_proper_code,
        vel_proper_code=fluid.vel_proper_code,
        pre_proper_code=fluid.pre_proper_code,
        temp_proper_code=fluid.temp_proper_code,
        time_proper_code=fluid.time_proper_code,
        mu_dimensionless=fluid.mu,
        xHI_dimensionless=fluid.xHI,
    )
    return par, mesh, fluid


def interior_slice(par):
    first = par.mesh.ghost_cells
    return slice(first, first + par.mesh.grid_cells)


def _density_cgs_g_cm3(values, par):
    return code_quantity_to_cgs(values, par.units.CodeUnits, 'density_cgs_g_cm3')


def _volume_cgs_cm3(values, par):
    return code_quantity_to_cgs(values, par.units.CodeUnits, 'volume_cgs_cm3')


def _radius_kpc(values, par):
    return code_quantity_to_cgs(values, par.units.CodeUnits, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm)


def ionization_front_position(mesh, fluid, par, neutral_fraction=0.5):
    interior = interior_slice(par)
    radius = _radius_kpc(mesh.x_proper_code[interior], par) * unyt.kpc
    xHI = np.asarray(fluid.xHI[interior])

    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0 * unyt.kpc
    if np.all(ionized):
        return _radius_kpc(mesh.boundary_proper_code[interior.stop], par) * unyt.kpc

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHI[left]
    x_right = xHI[right]
    if x_right == x_left:
        return radius[left]

    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return radius[left] + weight * (radius[right] - radius[left])


def ionized_hydrogen_atoms(mesh, fluid, par):
    interior = interior_slice(par)
    nH = rth._cgs_hydrogen_number_density(
        _density_cgs_g_cm3(fluid.rho_proper_code[interior], par),
        par.chemistry.hydrogen_mass_fraction,
    )
    ionized_fraction = 1.0 - np.asarray(fluid.xHI[interior])
    volume_cgs_cm3 = _volume_cgs_cm3(mesh.volume_proper_code[interior], par)
    return float(np.sum(ionized_fraction * nH * volume_cgs_cm3))


def photons_in_volume(mesh, fluid, par):
    interior = interior_slice(par)
    photon_density_cgs_cm3 = code_quantity_to_cgs(
        fluid.ngamma_code[interior], par.units.CodeUnits, 'number_density_cgs_cm3'
    )
    volume_cgs_cm3 = _volume_cgs_cm3(mesh.volume_proper_code[interior], par)
    return float(np.sum(photon_density_cgs_cm3 * volume_cgs_cm3))


def total_recombination_rate(mesh, fluid, par):
    interior = interior_slice(par)
    nH = rth._cgs_hydrogen_number_density(
        _density_cgs_g_cm3(fluid.rho_proper_code[interior], par),
        par.chemistry.hydrogen_mass_fraction,
    )
    ionized_fraction = 1.0 - np.asarray(fluid.xHI[interior])
    rate = np.sum(
        par.chemistry.alpha_B
        * ionized_fraction**2
        * nH**2
        * _volume_cgs_cm3(mesh.volume_proper_code[interior], par)
    )
    return float(rate) / unyt.s


def append_history(history, mesh, fluid, par, config, recombined_photons):
    radiation = config['par']['radiation']
    time_Myr = float(fluid.time_proper_code * par.units.CodeUnits.time_unit.to_value(unyt.Myr))
    history['time_Myr'].append(time_Myr)
    history['front_radius_kpc'].append(
        ionization_front_position(mesh, fluid, par).to_value(unyt.kpc)
    )
    history['injected_photons'].append(
        (radiation['radiative_transfer_source_photon_rate'] * time_Myr * unyt.Myr).to_value('')
    )
    history['ionized_atoms'].append(ionized_hydrogen_atoms(mesh, fluid, par))
    history['recombined_photons'].append(recombined_photons)
    history['volume_photons'].append(photons_in_volume(mesh, fluid, par))
    history['accounted_photons'].append(
        history['ionized_atoms'][-1]
        + history['recombined_photons'][-1]
        + history['volume_photons'][-1]
    )


def save_plot(mesh, fluid, par, config, figure_filename):
    radiation = config['par']['radiation']
    initial = config['initial_condition']
    thermo = config['par']['thermochemistry']
    example = config.get('example', {})
    interior = interior_slice(par)
    radius_kpc = _radius_kpc(mesh.x_proper_code[interior], par)
    radius = radius_kpc * unyt.kpc
    plot_radius_max = example.get('plot_radius_max', initial['box_size_proper']).to_value(unyt.kpc)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)
    xHII = 1.0 - xHI
    xHI_analytic = sa.neutral_fraction_profile(
        radius,
        initial['hydrogen_number_density'],
        thermo['hydrogen_sigma_gamma'],
        thermo['hydrogen_alpha_B'],
        radiation['radiative_transfer_source_photon_rate'],
        inner_radius=example['analytic_inner_radius'],
    )
    xHII_analytic = 1.0 - xHI_analytic
    radius_stromgren = sa.stromgren_radius(
        radiation['radiative_transfer_source_photon_rate'],
        initial['hydrogen_number_density'],
        thermo['hydrogen_alpha_B'],
    ).to(unyt.kpc)
    plot_floor = 1.0e-6

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        radius_kpc,
        np.clip(xHI, plot_floor, 1.0),
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}$ numerical',
    )
    ax.plot(
        radius_kpc,
        np.clip(xHII, plot_floor, 1.0),
        color='tab:red',
        lw=2.0,
        label=r'$x_{\rm HII}$ numerical',
    )
    ax.plot(
        radius_kpc,
        np.clip(xHI_analytic, plot_floor, 1.0),
        color='tab:blue',
        lw=1.6,
        ls='--',
        label=r'$x_{\rm HI}$ analytic',
    )
    ax.plot(
        radius_kpc,
        np.clip(xHII_analytic, plot_floor, 1.0),
        color='tab:red',
        lw=1.6,
        ls='--',
        label=r'$x_{\rm HII}$ analytic',
    )
    ax.axvline(
        radius_stromgren.to_value(unyt.kpc),
        color='black',
        lw=2.0,
        label=r'$R_{\rm S}=%.2f\ {\rm kpc}$' % radius_stromgren.to_value(unyt.kpc),
    )
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Hydrogen fraction')
    ax.set_xlim(0.0, plot_radius_max)
    ax.set_yscale('log')
    ax.set_ylim(plot_floor, 1.2)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, loc='center right')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_front_history_plot(history, config, figure_filename):
    radiation = config['par']['radiation']
    initial = config['initial_condition']
    thermo = config['par']['thermochemistry']
    example = config.get('example', {})
    time_Myr = np.asarray(history['time_Myr'])
    front_radius_kpc = np.asarray(history['front_radius_kpc'])
    plot_radius_max = example.get('plot_radius_max', initial['box_size_proper']).to_value(unyt.kpc)
    time = time_Myr * unyt.Myr
    analytic_front = sa.ionization_front_radius(
        time,
        radiation['radiative_transfer_source_photon_rate'],
        initial['hydrogen_number_density'],
        thermo['hydrogen_alpha_B'],
    ).to_value(unyt.kpc)
    radius_stromgren = sa.stromgren_radius(
        radiation['radiative_transfer_source_photon_rate'],
        initial['hydrogen_number_density'],
        thermo['hydrogen_alpha_B'],
    ).to_value(unyt.kpc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time_Myr,
        front_radius_kpc,
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}=0.5$ numerical',
    )
    ax.plot(
        time_Myr,
        analytic_front,
        color='black',
        lw=1.8,
        ls='--',
        label=r'$R_I(t)=R_S[1-\exp(-t/\tau_r)]^{1/3}$',
    )
    ax.axhline(
        radius_stromgren,
        color='0.25',
        lw=1.2,
        ls=':',
        label=r'$R_{\rm S}=%.2f\ {\rm kpc}$' % radius_stromgren,
    )
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Ionization-front radius [kpc]')
    ax.set_xlim(0.0, time_Myr[-1])
    ax.set_ylim(0.0, plot_radius_max)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc='lower right')
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.10, top=0.97, hspace=0.08)
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_photon_budget_plot(history, figure_filename):
    time_Myr = np.asarray(history['time_Myr'])
    injected = np.asarray(history['injected_photons'])
    ionized = np.asarray(history['ionized_atoms'])
    recombined = np.asarray(history['recombined_photons'])
    volume_photons = np.asarray(history['volume_photons'])
    accounted = np.asarray(history['accounted_photons'])
    residual = np.zeros_like(injected)
    valid = injected > 0.0
    residual[valid] = (accounted[valid] - injected[valid]) / injected[valid]

    fig, (ax_budget, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        gridspec_kw={'height_ratios': [2.0, 1.0], 'hspace': 0.08},
    )
    ax_budget.plot(
        time_Myr,
        injected,
        color='black',
        lw=2.0,
        label=r'injected photons, $\dot{N}_\gamma t$',
    )
    ax_budget.plot(
        time_Myr,
        accounted,
        color='tab:blue',
        lw=1.8,
        ls='--',
        label=r'$N_{\rm HII}+N_{\rm rec}+N_{\gamma,\rm vol}$',
    )
    ax_budget.plot(
        time_Myr,
        ionized,
        color='tab:red',
        lw=1.2,
        ls=':',
        label=r'$N_{\rm HII}$',
    )
    ax_budget.plot(
        time_Myr,
        recombined,
        color='tab:green',
        lw=1.2,
        ls='-.',
        label=r'$N_{\rm rec}$',
    )
    ax_budget.plot(
        time_Myr,
        volume_photons,
        color='tab:orange',
        lw=1.2,
        ls=(0, (3, 1, 1, 1)),
        label=r'$N_{\gamma,\rm vol}$',
    )
    ax_residual.axhline(0.0, color='black', lw=1.0)
    ax_residual.plot(
        time_Myr,
        residual,
        color='tab:purple',
        lw=1.8,
        label=(
            r'$(N_{\rm HII}+N_{\rm rec}+N_{\gamma,\rm vol}'
            r'-\dot{N}_\gamma t)/\dot{N}_\gamma t$'
        ),
    )

    ax_budget.set_ylabel('Photon count')
    ax_residual.set_xlabel('Time [Myr]')
    ax_residual.set_ylabel('Relative error')
    ax_budget.set_yscale('log')
    ax_budget.grid(True, which='both', alpha=0.25)
    ax_residual.grid(True, alpha=0.25)
    ax_budget.legend(frameon=False, loc='lower right')
    ax_residual.legend(frameon=False, loc='best')
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.10, top=0.97, hspace=0.08)
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
    return {
        'injected_photons': injected[-1],
        'accounted_photons': accounted[-1],
        'ionized_atoms': ionized[-1],
        'recombined_photons': recombined[-1],
        'volume_photons': volume_photons[-1],
        'relative_error': residual[-1],
    }
