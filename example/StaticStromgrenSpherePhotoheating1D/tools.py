"""Helper utilities for the photoheated static Stromgren sphere example."""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import example_utils as eu

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.thermo_networks.hydrogen import collisional_equilibrium_neutral_fraction
from radhydropy.units import (
    code_quantity_to_cgs,
    quantity_to_value,
)
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)

static_stromgren_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'StaticStromgrenSphere1D')
)
if static_stromgren_dir not in sys.path:
    sys.path.append(static_stromgren_dir)

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
    """Build the photoheating IC with the canonical nested runtime objects."""

    chemistry = config["par"].get('chemistry', {})
    thermo = config["par"].get('thermochemistry', {})
    radiation = config["par"].get('radiation', {})
    initial = config['initial_condition']
    grid_cells = int(config["par"]['mesh']['grid_cells'])
    sim = Rsim(config["par"])
    code_units_obj = sim.par.units.CodeUnits
    sim.par.simulation.box_size_proper_code = quantity_to_value(
        initial['box_size_proper'], code_units_obj.length_unit
    )
    sim.par.simulation.time_proper_code = quantity_to_value(
        initial.get('time_proper', 0.0 * unyt.Myr), code_units_obj.time_unit
    )
    sim.mesh.boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(0.0, initial['box_size_proper'].to_value(unyt.cm), grid_cells + 1) * unyt.cm,
        code_units_obj.length_unit,
    ))
    boundary_proper_code = sim.mesh.boundary_proper_code
    width_proper_code = np.diff(boundary_proper_code)
    volume_proper_code = 4.0 * np.pi / 3.0 * (
        boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
    )
    coordinate_proper_code = 0.75 * (
        boundary_proper_code[1:] ** 4 - boundary_proper_code[:-1] ** 4
    ) / (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    area_proper_code = 4.0 * np.pi * boundary_proper_code[:-1] ** 2
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=coordinate_proper_code,
        boundary_proper_code=boundary_proper_code,
        width_proper_code=width_proper_code,
        area_proper_code=area_proper_code,
        volume_proper_code=volume_proper_code,
    )
    sim.fluid.rho_proper_code = as_named_array(quantity_to_value((
        np.ones(grid_cells)
        * initial['hydrogen_number_density']
        * unyt.mp
        / chemistry.get('hydrogen_mass_fraction', 1.0)
    ).to(unyt.g / unyt.cm**3), code_units_obj.density_unit))
    sim.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells, dtype=float))
    temperature_proper_unyt = np.ones(grid_cells) * initial.get(
        'temperature_proper', 1.0e4 * unyt.K
    )
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(
        temperature_proper_unyt, code_units_obj.temperature_unit
    ))
    sim.fluid.mu = np.ones(grid_cells)
    if initial.get('hydrogen_initial_collisional_equilibrium', False):
        sim.fluid.xHI = np.ones(grid_cells) * collisional_equilibrium_neutral_fraction(
            initial.get('temperature_proper', 1.0e4 * unyt.K).to_value(unyt.K)
        )
    else:
        sim.fluid.xHI = np.ones(grid_cells) * chemistry.get(
            'hydrogen_xHI_initial',
            1.0,
        )
    if sim.par.thermochemistry.network == 'hydrogen_helium':
        sim.fluid.xHeI = np.ones(grid_cells) * chemistry.get(
            'hydrogen_helium_xHeI_initial', 1.0
        )
        sim.fluid.xHeII = np.ones(grid_cells) * chemistry.get(
            'hydrogen_helium_xHeII_initial', 0.0
        )
        sim.fluid.xHeIII = np.ones(grid_cells) * chemistry.get(
            'hydrogen_helium_xHeIII_initial', 0.0
        )
        sim.fluid.mu = np.ones(grid_cells) / (
            chemistry.get('hydrogen_mass_fraction', 1.0)
            + chemistry.get('helium_mass_fraction', 0.0) / 4.0
        )
    group_edges = radiation.get('radiation_group_edges_eV')
    if group_edges is not None:
        ngroup = len(group_edges) - 1
        photon_number_density_cgs_cm3_unyt = np.zeros((ngroup, grid_cells)) / unyt.cm**3
        sim.fluid.ngamma_code = as_named_array(quantity_to_value(
            photon_number_density_cgs_cm3_unyt,
            code_units_obj.number_density_unit,
        ))
    else:
        photon_number_density_cgs_cm3_unyt = np.ones(grid_cells) * radiation.get(
            'hydrogen_ngamma_initial', 0.0 / unyt.cm**3
        )
        sim.fluid.ngamma_code = as_named_array(quantity_to_value(
            photon_number_density_cgs_cm3_unyt,
            code_units_obj.number_density_unit,
        ))
    sim.fluid.SetFluidTime(sim.par.simulation.time_proper_code)
    _attach_proper_runtime_states(sim.mesh, sim.fluid)
    return sim.par, sim.mesh, sim.fluid, sim.solver


def write_initial_condition(config):
    """Build the raw IC state, replace any stale snapshot, and write it."""
    par, mesh, fluid, solver = build_static_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    filename = config['par']['simulation']['initial_condition_filename']
    rio.writehdf5(sim, filename)


def _refresh_mesh_geometry(mesh, config):
    par = config['_output_par']
    """Recompute derived mesh geometry from an already ghosted boundary."""
    mesh.width_proper_code = mesh.boundary_proper_code[1:] - mesh.boundary_proper_code[:-1]
    mesh.coordinate_inverse_proper_code = 1.0 / mesh.width_proper_code
    code_units_obj = par.units.CodeUnits
    if par.simulation.coordinate_system == 'cartesian':
        mesh.x_proper_code = 0.5 * (mesh.boundary_proper_code[1:] + mesh.boundary_proper_code[:-1])
        mesh.area_proper_code = np.ones(len(mesh.width_proper_code)) * quantity_to_value(
            par.mesh.area_proper, code_units_obj.area_unit
        )
        mesh.volume_proper_code = mesh.width_proper_code * mesh.area_proper_code
    elif par.simulation.coordinate_system == 'spherical':
        mesh.area_proper_code = (mesh.boundary_proper_code[:-1] ** 2) * 4.0 * np.pi
        mesh.volume_proper_code = np.absolute((mesh.boundary_proper_code[1:] ** 3 - mesh.boundary_proper_code[:-1] ** 3)) * 4.0 * np.pi / 3.0
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


def load_output_state(outputfilename, config):
    snapshot = Rsim(config['par'])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, outputfilename)
    par, mesh, fluid = snapshot.par, snapshot.mesh, snapshot.fluid
    config['_output_par'] = par
    _refresh_mesh_geometry(mesh, config)
    return par, mesh, fluid


def interior_slice(config):
    par = config['_output_par']
    first = par.mesh.ghost_cells
    return slice(first, first + par.mesh.grid_cells)


def ionization_front_position(mesh, fluid, config, neutral_fraction=0.5):
    par = config['_output_par']
    interior = interior_slice(config)
    radius_proper_cgs_kpc_unyt = code_quantity_to_cgs(mesh.x_proper_code[interior], par.units.CodeUnits, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm) * unyt.kpc
    xHI = np.asarray(fluid.xHI[interior])

    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0 * unyt.kpc
    if np.all(ionized):
        return code_quantity_to_cgs(mesh.boundary_proper_code[interior.stop], par.units.CodeUnits, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm) * unyt.kpc

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHI[left]
    x_right = xHI[right]
    if x_right == x_left:
        return radius_proper_cgs_kpc_unyt[left]

    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return radius_proper_cgs_kpc_unyt[left] + weight * (radius_proper_cgs_kpc_unyt[right] - radius_proper_cgs_kpc_unyt[left])


def mean_ionized_temperature(fluid, config):
    par = config['_output_par']
    interior = interior_slice(config)
    xHI = np.asarray(fluid.xHI[interior])
    temperature_proper_cgs_K = code_quantity_to_cgs(fluid.temp_proper_code[interior], par.units.CodeUnits, 'temperature_cgs_K')
    ionized = 1.0 - xHI
    if np.sum(ionized) <= 0.0:
        return 0.0
    return float(np.sum(ionized * temperature_proper_cgs_K) / np.sum(ionized))


def append_history(history, mesh, fluid, config):
    par = config['_output_par']
    history['time_proper_Myr'].append(float(fluid.time_proper_code * par.units.CodeUnits.time_unit.to_value(unyt.Myr)))
    history['front_radius_proper_kpc'].append(
        ionization_front_position(mesh, fluid, config).to_value(unyt.kpc)
    )
    history['mean_ionized_temp_cgs_K'].append(mean_ionized_temperature(fluid, config))


def load_log_reference_profile(filename, radius_unit):
    if filename is None or not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'radius_proper_kpc': data[:, 0] * radius_unit.to_value(unyt.kpc),
        'value': 10.0**data[:, 1],
    }


def save_plot(mesh, fluid, history, config, figure_filename):
    par = config['_output_par']

    radiation = config["par"]['radiation']
    thermo = config["par"]['thermochemistry']
    initial = config['initial_condition']
    example = config.get('example', {})
    interior = interior_slice(config)
    code_units_obj = par.units.CodeUnits
    radius_proper_cgs_kpc = code_quantity_to_cgs(mesh.x_proper_code[interior], code_units_obj, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm)
    radius_proper_cgs_kpc_unyt = radius_proper_cgs_kpc * unyt.kpc
    snapshot = history.get('reference_snapshot', None)
    if snapshot is None:
        xHI = np.asarray(fluid.xHI[interior], dtype=float)
        temperature_cgs_K = code_quantity_to_cgs(fluid.temp_proper_code[interior], code_units_obj, 'temperature_cgs_K')
        profile_time_proper_Myr = float(fluid.time_proper_code * code_units_obj.time_unit.to_value(unyt.Myr))
    else:
        radius_proper_cgs_kpc = snapshot['radius_proper_kpc']
        xHI = snapshot['xHI']
        temperature_cgs_K = snapshot['temperature_cgs_K']
        profile_time_proper_Myr = snapshot['time_proper_Myr']
    xHII = 1.0 - xHI
    plot_radius_max = example.get('plot_radius_max', initial['box_size_proper']).to_value(unyt.kpc)
    reference_radius_unit = example.get('reference_radius_unit', 5.4 * unyt.kpc)
    temperature_reference = load_log_reference_profile(
        example.get('temperature_reference_filename', None),
        reference_radius_unit,
    )
    neutral_fraction_reference = load_log_reference_profile(
        example.get('neutral_fraction_reference_filename', None),
        reference_radius_unit,
    )
    alpha_B = thermo.get('hydrogen_alpha_B')
    if alpha_B is not None:
        xHI_analytic = sa.neutral_fraction_profile(
            radius_proper_cgs_kpc_unyt,
            initial['hydrogen_number_density'],
            thermo['hydrogen_sigma_gamma'],
            alpha_B,
            radiation['source_photon_rate'],
            inner_radius_proper_unyt=example['analytic_inner_radius'],
        )
        xHII_analytic = 1.0 - xHI_analytic
        radius_stromgren = sa.stromgren_radius(
            radiation['source_photon_rate'],
            initial['hydrogen_number_density'],
            alpha_B,
        ).to(unyt.kpc)
        analytic_front = sa.ionization_front_radius(
            np.asarray(history['time_proper_Myr']) * unyt.Myr,
            radiation['source_photon_rate'],
            initial['hydrogen_number_density'],
            alpha_B,
        ).to_value(unyt.kpc)
    else:
        xHI_analytic = xHII_analytic = None
        radius_stromgren = None
        analytic_front = None

    fig, (ax_frac, ax_temp, ax_front) = plt.subplots(
        3,
        1,
        figsize=(7.4, 8.0),
        gridspec_kw={'height_ratios': [1.4, 1.2, 1.2], 'hspace': 0.34},
    )
    ax_frac.plot(radius_proper_cgs_kpc, np.clip(xHI, 1.0e-6, 1.0), label=r'$x_{\rm HI}$')
    ax_frac.plot(radius_proper_cgs_kpc, np.clip(xHII, 1.0e-6, 1.0), label=r'$x_{\rm HII}$')
    if xHI_analytic is not None:
        ax_frac.plot(
            radius_proper_cgs_kpc,
            np.clip(xHI_analytic, 1.0e-6, 1.0),
            color='tab:blue', lw=1.4, ls='--', label=r'$x_{\rm HI}$ analytic',
        )
        ax_frac.plot(
            radius_proper_cgs_kpc,
            np.clip(xHII_analytic, 1.0e-6, 1.0),
            color='tab:orange', lw=1.4, ls='--', label=r'$x_{\rm HII}$ analytic',
        )
    if neutral_fraction_reference is not None:
        ax_frac.scatter(
            neutral_fraction_reference['radius_proper_kpc'],
            np.clip(neutral_fraction_reference['value'], 1.0e-6, 1.0),
            s=18,
            color='black',
            marker='o',
            facecolors='none',
            label=r'$x_{\rm HI}$ 100 Myr ref.',
        )
    if radius_stromgren is not None:
        ax_frac.axvline(radius_stromgren.to_value(unyt.kpc), color='black', lw=1.5, ls=':', label=r'$R_{\rm S}$')
    ax_frac.set_xlim(0.0, plot_radius_max)
    ax_frac.set_ylim(1.0e-6, 1.2)
    ax_frac.set_yscale('log')
    ax_frac.set_ylabel('Hydrogen fraction')
    ax_frac.set_title('Radial profiles at %.0f Myr' % profile_time_proper_Myr)
    ax_frac.grid(True, which='both', alpha=0.25)
    ax_frac.legend(frameon=False, loc='center right')

    ax_temp.plot(radius_proper_cgs_kpc, temperature_cgs_K, color='tab:red', lw=1.8)
    if temperature_reference is not None:
        ax_temp.scatter(
            temperature_reference['radius_proper_kpc'],
            temperature_reference['value'],
            s=18,
            color='black',
            marker='o',
            facecolors='none',
            label='100 Myr ref.',
        )
    if radius_stromgren is not None:
        ax_temp.axvline(radius_stromgren.to_value(unyt.kpc), color='black', lw=1.5, ls=':')
    ax_temp.set_xlim(0.0, plot_radius_max)
    ax_temp.set_yscale('log')
    ax_temp.set_ylabel('Temperature [K]')
    ax_temp.grid(True, which='both', alpha=0.25)
    if temperature_reference is not None:
        ax_temp.legend(frameon=False, loc='upper right')

    ax_front.plot(
        history['time_proper_Myr'],
        history['front_radius_proper_kpc'],
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}=0.5$',
    )
    if analytic_front is not None:
        ax_front.plot(history['time_proper_Myr'], analytic_front, color='black', lw=1.6, ls='--', label=r'$R_I(t)$ fixed-$T$ reference')
    if radius_stromgren is not None:
        ax_front.axhline(radius_stromgren.to_value(unyt.kpc), color='0.25', lw=1.2, ls=':')
    ax_front.set_xlim(0.0, history['time_proper_Myr'][-1])
    ax_front.set_ylim(0.0, plot_radius_max)
    ax_front.set_xlabel('Time [Myr]')
    ax_front.set_ylabel('I-front radius [kpc]')
    ax_front.grid(True, alpha=0.25)
    ax_front.legend(frameon=False, loc='lower right')

    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
