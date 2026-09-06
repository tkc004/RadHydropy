"""Helper utilities for the photoheated static Stromgren sphere example."""

import os
import sys
from types import SimpleNamespace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import example_utils as eu

from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
import radhydropy.io as rio
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
from radhydropy.thermo_networks.hydrogen import collisional_equilibrium_neutral_fraction
from radhydropy.units import (
    CodeUnits,
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
        coordinate=coordinate_proper_code,
        boundary=boundary_proper_code,
        width=width_proper_code,
        area=area_proper_code,
        volume=volume_proper_code,
    )
    fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    fluid.SetPressure()
    fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        density=fluid.rho_proper_code,
        velocity=fluid.vel_proper_code,
        pressure=fluid.pre_proper_code,
        temperature=fluid.temp_proper_code,
        time=fluid.time_proper_code,
        mu=fluid.mu,
        xHI=fluid.xHI if hasattr(fluid, 'xHI') else None,
    )


def build_static_problem(config):
    par_config = config['par']
    simulation = par_config['simulation']
    mesh_config = par_config['mesh']
    hydro = par_config['hydrodynamics']
    boundary = par_config['boundary']
    timestep = par_config['timestep']
    chemistry = par_config.get('chemistry', {})
    thermo = par_config.get('thermochemistry', {})
    radiation = par_config.get('radiation', {})
    output = par_config.get('output', {})
    initial = config['initial_condition']
    code_units_obj = CodeUnits.from_mapping(par_config['units']['CodeUnits'])
    par = SimpleNamespace(
        coordsys=simulation.get('coordinate_system', 'spherical'),
        boundcond=boundary.get('condition', 'OpenSph'),
        nogrid=mesh_config['grid_cells'],
        noghost=mesh_config.get('ghost_cells', 2),
        boxsize=initial['box_size'],
        verbose=par_config.get('diagnostics', {}).get('verbose', 0),
        outdir=output.get('directory', '.'),
        outfileprefix=output.get('filename_prefix', 'Output'),
        savedir=output.get('savedir', output.get('directory', '.')),
        area=mesh_config.get('area', 1.0 * unyt.cm**2),
        EOStype=hydro.get('eos_type', 'polytropic'),
        gamma=hydro.get('gamma', 5.0 / 3.0),
        dtmin=timestep.get('dtmin', 1.0e-6 * unyt.Myr),
        dtmax=timestep.get('dtmax', 1.0 * unyt.Myr),
        hydrogen_chemistry=thermo.get('hydrogen_chemistry', True),
        thermochemistry_network=thermo.get('thermochemistry_network', 'hydrogen'),
        hydrogen_mass_fraction=chemistry.get('hydrogen_mass_fraction', 1.0),
        helium_mass_fraction=chemistry.get('helium_mass_fraction', 0.0),
        hydrogen_helium_coupled_implicit=thermo.get(
            'hydrogen_helium_coupled_implicit', True
        ),
        hydrogen_xHI_initial=chemistry.get('hydrogen_xHI_initial', 1.0),
        hydrogen_xHI_inflow=chemistry.get('hydrogen_xHI_inflow', 1.0),
        hydrogen_xHI_outflow=chemistry.get('hydrogen_xHI_outflow', 1.0),
        hydrogen_source_CFL=thermo.get('hydrogen_source_CFL', 0.1),
        hydrogen_source_dtmin=thermo.get('hydrogen_source_dtmin', 0.0 * unyt.Myr),
        hydrogen_update_mu=thermo.get('hydrogen_update_mu', True),
        hydrogen_thermal_coupling=thermo.get('hydrogen_thermal_coupling', True),
        hydrogen_recombination=thermo.get('hydrogen_recombination', True),
        hydrogen_collisional_ionization=thermo.get('hydrogen_collisional_ionization', False),
        hydrogen_alpha_B=thermo.get('hydrogen_alpha_B'),
        hydrogen_beta=thermo.get('hydrogen_beta', 0.0 * unyt.cm**3 / unyt.s),
        hydrogen_radiation_field=thermo.get('hydrogen_radiation_field', False),
        hydrogen_radiation_evolution=thermo.get('hydrogen_radiation_evolution', False),
        hydrogen_ngamma_initial=thermo.get('hydrogen_ngamma_initial', 0.0 / unyt.cm**3),
        hydrogen_sigma_gamma=thermo.get('hydrogen_sigma_gamma'),
        hydrogen_epsilon_gamma=thermo.get('hydrogen_epsilon_gamma'),
        radiative_transfer=radiation.get('radiative_transfer', True),
        radiative_transfer_method=radiation.get('radiative_transfer_method', 'long_characteristics'),
        radiative_transfer_temporal_scheme=radiation.get(
            'radiative_transfer_temporal_scheme', 'instantaneous'
        ),
        radiative_transfer_c2ray_max_iterations=radiation.get(
            'radiative_transfer_c2ray_max_iterations', 32
        ),
        radiative_transfer_c2ray_tolerance=radiation.get(
            'radiative_transfer_c2ray_tolerance', 1.0e-6
        ),
        radiative_transfer_c2ray_relaxation=radiation.get(
            'radiative_transfer_c2ray_relaxation', 1.0
        ),
        radiative_transfer_c2ray_nonconvergence=radiation.get(
            'radiative_transfer_c2ray_nonconvergence', 'warn'
        ),
        radiative_transfer_boundary_flux=radiation.get(
            'radiative_transfer_boundary_flux',
            0.0 / (unyt.cm**2 * unyt.s),
        ),
        radiative_transfer_source_photon_rate=radiation.get(
            'radiative_transfer_source_photon_rate',
            radiation.get('radiation_spectrum_total_photon_rate'),
        ),
        radiative_transfer_source_photon_rate_groups=radiation.get(
            'radiative_transfer_source_photon_rate_groups',
        ),
        radiation_group_edges_eV=radiation.get('radiation_group_edges_eV'),
        radiation_group_sigma_gamma=radiation.get('radiation_group_sigma_gamma'),
        radiation_group_epsilon_gamma=radiation.get('radiation_group_epsilon_gamma'),
        radiation_group_sigma_gamma_HeI=radiation.get('radiation_group_sigma_gamma_HeI'),
        radiation_group_sigma_gamma_HeII=radiation.get('radiation_group_sigma_gamma_HeII'),
        radiation_group_epsilon_gamma_HeI=radiation.get('radiation_group_epsilon_gamma_HeI'),
        radiation_group_epsilon_gamma_HeII=radiation.get('radiation_group_epsilon_gamma_HeII'),
        radiative_transfer_direction=radiation.get('radiative_transfer_direction', 1),
        CodeUnits=code_units_obj,
        unit_system=code_units_obj.unit_system,
    )
    par.units = SimpleNamespace(CodeUnits=code_units_obj)
    par.simulation = SimpleNamespace(
        coordinate_system=par.coordsys,
        time_code=0.0 * unyt.Myr,
        box_size=par.boxsize,
    )
    par.mesh = SimpleNamespace(grid_cells=par.nogrid, ghost_cells=par.noghost)

    mesh = Mesh()
    mesh.boundary_proper_code = quantity_to_value(np.linspace(
        0.0,
        initial['box_size'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm, code_units_obj.length_unit)

    fluid = Fluid()
    fluid.code_units = code_units_obj
    fluid.eos = EOS(par.EOStype, par.gamma, code_units_obj)
    fluid.rho_proper_code = quantity_to_value((
        np.ones(par.nogrid)
        * initial['hydrogen_number_density']
        * unyt.mp
        / par.hydrogen_mass_fraction
    ).to(unyt.g / unyt.cm**3), code_units_obj.density_unit)
    fluid.vel_proper_code = np.zeros(par.nogrid, dtype=float)
    temperature_proper_cgs_K_unyt = np.ones(par.nogrid) * initial.get(
        'initial_temperature', 1.0e4 * unyt.K
    )
    fluid.temp_proper_code = quantity_to_value(
        temperature_proper_cgs_K_unyt, code_units_obj.temperature_unit
    )
    fluid.mu = np.ones(par.nogrid)
    if initial.get('hydrogen_initial_collisional_equilibrium', False):
        fluid.xHI = np.ones(par.nogrid) * collisional_equilibrium_neutral_fraction(
            initial.get('initial_temperature', 1.0e4 * unyt.K).to_value(unyt.K)
        )
    else:
        fluid.xHI = np.ones(par.nogrid) * chemistry.get(
            'hydrogen_xHI_initial',
            1.0,
        )
    if par.thermochemistry_network == 'hydrogen_helium':
        fluid.xHeI = np.ones(par.nogrid) * chemistry.get(
            'hydrogen_helium_xHeI_initial', 1.0
        )
        fluid.xHeII = np.ones(par.nogrid) * chemistry.get(
            'hydrogen_helium_xHeII_initial', 0.0
        )
        fluid.xHeIII = np.ones(par.nogrid) * chemistry.get(
            'hydrogen_helium_xHeIII_initial', 0.0
        )
        fluid.mu = np.ones(par.nogrid) / (
            par.hydrogen_mass_fraction + par.helium_mass_fraction / 4.0
        )
    group_edges = radiation.get('radiation_group_edges_eV')
    if group_edges is not None:
        ngroup = len(group_edges) - 1
        photon_number_density_cgs_cm3_unyt = np.zeros((ngroup, par.nogrid)) / unyt.cm**3
        fluid.ngamma_code = quantity_to_value(
            photon_number_density_cgs_cm3_unyt,
            code_units_obj.number_density_unit,
        )
    else:
        photon_number_density_cgs_cm3_unyt = np.ones(par.nogrid) * thermo.get(
            'hydrogen_ngamma_initial', 0.0 / unyt.cm**3
        )
        fluid.ngamma_code = quantity_to_value(
            photon_number_density_cgs_cm3_unyt,
            code_units_obj.number_density_unit,
        )
    fluid.SetFluidTime(0.0)
    _attach_proper_runtime_states(mesh, fluid)
    solver = Solver()
    return par, mesh, fluid, solver


def write_initial_condition(config):
    """Build the raw IC state, replace any stale snapshot, and write it."""
    par, mesh, fluid, _ = build_static_problem(config)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    filename = config['par']['simulation']['initial_condition_filename']
    Path(filename).unlink(missing_ok=True)
    rio.writehdf5(sim, filename)


def _refresh_mesh_geometry(mesh, par):
    """Recompute derived mesh geometry from an already ghosted boundary."""
    mesh.width_proper_code = mesh.boundary_proper_code[1:] - mesh.boundary_proper_code[:-1]
    mesh.coordinate_inverse_proper_code = 1.0 / mesh.width_proper_code
    if par.coordsys == 'cartesian':
        mesh.x_proper_code = 0.5 * (mesh.boundary_proper_code[1:] + mesh.boundary_proper_code[:-1])
        if hasattr(par, 'area'):
            mesh.area_proper_code = np.ones(len(mesh.width_proper_code)) * quantity_to_value(par.area, par.CodeUnits.area_unit)
        else:
            mesh.area_proper_code = np.ones(len(mesh.width_proper_code))
        mesh.volume_proper_code = mesh.width_proper_code * mesh.area_proper_code
    elif par.coordsys == 'spherical':
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
        raise ValueError("coordsys unknown: %s" % par.coordsys)
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=mesh.x_proper_code,
        boundary=mesh.boundary_proper_code,
        width=mesh.width_proper_code,
        area=mesh.area_proper_code,
        volume=mesh.volume_proper_code,
    )


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_static_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    _refresh_mesh_geometry(mesh, par)
    return par, mesh, fluid


def interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def ionization_front_position(mesh, fluid, par, neutral_fraction=0.5):
    interior = interior_slice(par)
    radius = code_quantity_to_cgs(mesh.x_proper_code[interior], par.CodeUnits, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm) * unyt.kpc
    xHI = np.asarray(fluid.xHI[interior])

    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0 * unyt.kpc
    if np.all(ionized):
        return code_quantity_to_cgs(mesh.boundary_proper_code[interior.stop], par.CodeUnits, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm) * unyt.kpc

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHI[left]
    x_right = xHI[right]
    if x_right == x_left:
        return radius[left]

    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return radius[left] + weight * (radius[right] - radius[left])


def mean_ionized_temperature(fluid, par):
    interior = interior_slice(par)
    xHI = np.asarray(fluid.xHI[interior])
    temperature = code_quantity_to_cgs(fluid.temp_proper_code[interior], par.CodeUnits, 'temperature_cgs_K')
    ionized = 1.0 - xHI
    if np.sum(ionized) <= 0.0:
        return 0.0
    return float(np.sum(ionized * temperature) / np.sum(ionized))


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(float(fluid.time_proper_code * par.CodeUnits.time_unit.to_value(unyt.Myr)))
    history['front_radius_kpc'].append(
        ionization_front_position(mesh, fluid, par).to_value(unyt.kpc)
    )
    history['mean_ionized_temp_cgs_K'].append(mean_ionized_temperature(fluid, par))


def load_log_reference_profile(filename, radius_unit):
    if filename is None or not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'radius_kpc': data[:, 0] * radius_unit.to_value(unyt.kpc),
        'value': 10.0**data[:, 1],
    }


def save_plot(mesh, fluid, par, history, config, figure_filename):
    par_config = config['par']
    radiation = par_config['radiation']
    thermo = par_config['thermochemistry']
    initial = config['initial_condition']
    example = config.get('example', {})
    interior = interior_slice(par)
    code_units_obj = CodeUnits.from_mapping(par_config['units']['CodeUnits'])
    radius_kpc = code_quantity_to_cgs(mesh.x_proper_code[interior], par.CodeUnits, 'length_cgs_cm') / (1.0 * unyt.kpc).to_value(unyt.cm)
    radius = radius_kpc * unyt.kpc
    snapshot = history.get('reference_snapshot', None)
    if snapshot is None:
        xHI = np.asarray(fluid.xHI[interior], dtype=float)
        temperature_cgs_K = code_quantity_to_cgs(fluid.temp_proper_code[interior], par.CodeUnits, 'temperature_cgs_K')
        profile_time_Myr = float(fluid.time_proper_code * par.CodeUnits.time_unit.to_value(unyt.Myr))
    else:
        radius_kpc = snapshot['radius_kpc']
        xHI = snapshot['xHI']
        temperature_cgs_K = snapshot['temperature_cgs_K']
        profile_time_Myr = snapshot['time_Myr']
    xHII = 1.0 - xHI
    plot_radius_max = example.get('plot_radius_max', initial['box_size']).to_value(unyt.kpc)
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
            radius,
            initial['hydrogen_number_density'],
            thermo['hydrogen_sigma_gamma'],
            alpha_B,
            radiation['radiative_transfer_source_photon_rate'],
            inner_radius=example['analytic_inner_radius'],
        )
        xHII_analytic = 1.0 - xHI_analytic
        radius_stromgren = sa.stromgren_radius(
            radiation['radiative_transfer_source_photon_rate'],
            initial['hydrogen_number_density'],
            alpha_B,
        ).to(unyt.kpc)
        analytic_front = sa.ionization_front_radius(
            np.asarray(history['time_Myr']) * unyt.Myr,
            radiation['radiative_transfer_source_photon_rate'],
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
    ax_frac.plot(radius_kpc, np.clip(xHI, 1.0e-6, 1.0), label=r'$x_{\rm HI}$')
    ax_frac.plot(radius_kpc, np.clip(xHII, 1.0e-6, 1.0), label=r'$x_{\rm HII}$')
    if xHI_analytic is not None:
        ax_frac.plot(
            radius_kpc,
            np.clip(xHI_analytic, 1.0e-6, 1.0),
            color='tab:blue', lw=1.4, ls='--', label=r'$x_{\rm HI}$ analytic',
        )
        ax_frac.plot(
            radius_kpc,
            np.clip(xHII_analytic, 1.0e-6, 1.0),
            color='tab:orange', lw=1.4, ls='--', label=r'$x_{\rm HII}$ analytic',
        )
    if neutral_fraction_reference is not None:
        ax_frac.scatter(
            neutral_fraction_reference['radius_kpc'],
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
    ax_frac.set_title('Radial profiles at %.0f Myr' % profile_time_Myr)
    ax_frac.grid(True, which='both', alpha=0.25)
    ax_frac.legend(frameon=False, loc='center right')

    ax_temp.plot(radius_kpc, temperature_cgs_K, color='tab:red', lw=1.8)
    if temperature_reference is not None:
        ax_temp.scatter(
            temperature_reference['radius_kpc'],
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
        history['time_Myr'],
        history['front_radius_kpc'],
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}=0.5$',
    )
    if analytic_front is not None:
        ax_front.plot(history['time_Myr'], analytic_front, color='black', lw=1.6, ls='--', label=r'$R_I(t)$ fixed-$T$ reference')
    if radius_stromgren is not None:
        ax_front.axhline(radius_stromgren.to_value(unyt.kpc), color='0.25', lw=1.2, ls=':')
    ax_front.set_xlim(0.0, history['time_Myr'][-1])
    ax_front.set_ylim(0.0, plot_radius_max)
    ax_front.set_xlabel('Time [Myr]')
    ax_front.set_ylabel('I-front radius [kpc]')
    ax_front.grid(True, alpha=0.25)
    ax_front.legend(frameon=False, loc='lower right')

    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
