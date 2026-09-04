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

from radhydropy.fluid import Fluid
import radhydropy.io as rio
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
from radhydropy.thermo_networks.hydrogen import collisional_equilibrium_neutral_fraction
from radhydropy.units import (
    CodeUnits,
)

static_stromgren_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'StaticStromgrenSphere1D')
)
if static_stromgren_dir not in sys.path:
    sys.path.append(static_stromgren_dir)

import stromgren_analytic as sa


def build_static_problem(config):
    if 'par' in config:
        config = {**eu.legacy_example_parameters(config), **config.get('initial_condition', {})}
    code_units_obj = CodeUnits.from_mapping(config.get('CodeUnits'))
    par = SimpleNamespace(
        coordsys=config.get('coordsys', 'spherical'),
        boundcond=config.get('boundcond', 'OpenSph'),
        nogrid=config['number_of_cells'],
        noghost=config.get('noghost', 2),
        boxsize=config['boxsize'],
        verbose=config.get('verbose', 0),
        outdir=config.get('outdir', '.'),
        outfileprefix=config.get('outfileprefix', 'Output'),
        savedir=config.get('savedir', config.get('outdir', '.')),
        area=config.get('area', 1.0 * unyt.cm**2),
        EOStype=config.get('EOStype', 'polytropic'),
        gamma=config.get('gamma', 5.0 / 3.0),
        dtmin=config.get('dtmin', 1.0e-6 * unyt.Myr),
        dtmax=config.get('dtmax', 1.0 * unyt.Myr),
        hydrogen_chemistry=config.get('hydrogen_chemistry', True),
        thermochemistry_network=config.get('thermochemistry_network', 'hydrogen'),
        hydrogen_mass_fraction=config.get('hydrogen_mass_fraction', 1.0),
        helium_mass_fraction=config.get('helium_mass_fraction', 0.0),
        hydrogen_helium_coupled_implicit=config.get(
            'hydrogen_helium_coupled_implicit', True
        ),
        hydrogen_xHI_initial=config.get('hydrogen_xHI_initial', 1.0),
        hydrogen_xHI_inflow=config.get('hydrogen_xHI_inflow', 1.0),
        hydrogen_xHI_outflow=config.get('hydrogen_xHI_outflow', 1.0),
        hydrogen_source_CFL=config.get(
            'hydrogen_source_CFL',
            config.get('evolution_timestep_cfl', 0.1),
        ),
        hydrogen_source_dtmin=config.get('hydrogen_source_dtmin', 0.0 * unyt.Myr),
        hydrogen_update_mu=config.get('hydrogen_update_mu', True),
        hydrogen_thermal_coupling=config.get('hydrogen_thermal_coupling', True),
        hydrogen_recombination=config.get('hydrogen_recombination', True),
        hydrogen_collisional_ionization=config.get('hydrogen_collisional_ionization', False),
        hydrogen_alpha_B=config.get('alpha_B_coefficient', config.get('hydrogen_alpha_B')),
        hydrogen_beta=config.get('hydrogen_beta', 0.0 * unyt.cm**3 / unyt.s),
        hydrogen_radiation_field=config.get('hydrogen_radiation_field', False),
        hydrogen_radiation_evolution=config.get('hydrogen_radiation_evolution', False),
        hydrogen_ngamma_initial=config.get('hydrogen_ngamma_initial', 0.0 / unyt.cm**3),
        hydrogen_sigma_gamma=config.get('sigma_gamma', config.get('hydrogen_sigma_gamma')),
        hydrogen_epsilon_gamma=config.get('epsilon_gamma', config.get('hydrogen_epsilon_gamma')),
        radiative_transfer=config.get('radiative_transfer', True),
        radiative_transfer_method=config.get('radiative_transfer_method', 'long_characteristics'),
        radiative_transfer_temporal_scheme=config.get(
            'radiative_transfer_temporal_scheme', 'instantaneous'
        ),
        radiative_transfer_c2ray_max_iterations=config.get(
            'radiative_transfer_c2ray_max_iterations', 32
        ),
        radiative_transfer_c2ray_tolerance=config.get(
            'radiative_transfer_c2ray_tolerance', 1.0e-6
        ),
        radiative_transfer_c2ray_relaxation=config.get(
            'radiative_transfer_c2ray_relaxation', 1.0
        ),
        radiative_transfer_c2ray_nonconvergence=config.get(
            'radiative_transfer_c2ray_nonconvergence', 'warn'
        ),
        radiative_transfer_boundary_flux=config.get(
            'radiative_transfer_boundary_flux',
            0.0 / (unyt.cm**2 * unyt.s),
        ),
        radiative_transfer_source_photon_rate=config.get(
            'source_photon_rate',
            config.get('radiative_transfer_source_photon_rate'),
        ),
        radiative_transfer_source_photon_rate_groups=config.get(
            'radiative_transfer_source_photon_rate_groups',
        ),
        radiation_group_edges_eV=config.get('radiation_group_edges_eV'),
        radiation_group_sigma_gamma=config.get('radiation_group_sigma_gamma'),
        radiation_group_epsilon_gamma=config.get('radiation_group_epsilon_gamma'),
        radiation_group_sigma_gamma_HeI=config.get('radiation_group_sigma_gamma_HeI'),
        radiation_group_sigma_gamma_HeII=config.get('radiation_group_sigma_gamma_HeII'),
        radiation_group_epsilon_gamma_HeI=config.get('radiation_group_epsilon_gamma_HeI'),
        radiation_group_epsilon_gamma_HeII=config.get('radiation_group_epsilon_gamma_HeII'),
        radiative_transfer_direction=config.get('radiative_transfer_direction', 1),
        CodeUnits=code_units_obj,
        unit_system=code_units_obj.unit_system,
    )
    par.units = SimpleNamespace(CodeUnits=code_units_obj)
    par.simulation = SimpleNamespace(
        coordinate_system=par.coordsys,
        current_time=0.0 * unyt.Myr,
        box_size=par.boxsize,
    )
    par.mesh = SimpleNamespace(grid_cells=par.nogrid, ghost_cells=par.noghost)

    mesh = Mesh()
    mesh.boundary = np.linspace(
        0.0,
        config['boxsize'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm

    fluid = Fluid()
    fluid.rho_code = (
        np.ones(par.nogrid)
        * config['hydrogen_number_density']
        * unyt.mp
        / par.hydrogen_mass_fraction
    ).to(unyt.g / unyt.cm**3)
    fluid.vel_code = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp_code = np.ones(par.nogrid) * config.get('initial_temperature', 1.0e4 * unyt.K)
    fluid.mu = np.ones(par.nogrid)
    if config.get('hydrogen_initial_collisional_equilibrium', False):
        fluid.xHI = np.ones(par.nogrid) * collisional_equilibrium_neutral_fraction(
            config.get('initial_temperature', 1.0e4 * unyt.K).to_value(unyt.K)
        )
    else:
        fluid.xHI = np.ones(par.nogrid) * config.get(
            'hydrogen_xHI_initial',
            1.0,
        )
    if par.thermochemistry_network == 'hydrogen_helium':
        fluid.xHeI = np.ones(par.nogrid) * config.get(
            'hydrogen_helium_xHeI_initial', 1.0
        )
        fluid.xHeII = np.ones(par.nogrid) * config.get(
            'hydrogen_helium_xHeII_initial', 0.0
        )
        fluid.xHeIII = np.ones(par.nogrid) * config.get(
            'hydrogen_helium_xHeIII_initial', 0.0
        )
        fluid.mu = np.ones(par.nogrid) / (
            par.hydrogen_mass_fraction + par.helium_mass_fraction / 4.0
        )
    group_edges = config.get('radiation_group_edges_eV')
    if group_edges is not None:
        ngroup = len(group_edges) - 1
        fluid.ngamma_code = np.zeros((ngroup, par.nogrid)) / unyt.cm**3
    else:
        fluid.ngamma_code = np.ones(par.nogrid) * config.get('hydrogen_ngamma_initial', 0.0 / unyt.cm**3)
    fluid.SetFluidTime(0.0 * unyt.Myr)
    solver = Solver()
    return par, mesh, fluid, solver


def write_initial_condition(config, runparams):
    """Build the raw IC state, replace any stale snapshot, and write it."""
    par, mesh, fluid, _ = build_static_problem(config)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    Path(runparams['ICfilename']).unlink(missing_ok=True)
    rio.writehdf5(sim, runparams['ICfilename'])


def _refresh_mesh_geometry(mesh, par):
    """Recompute derived mesh geometry from an already ghosted boundary."""
    mesh.xdelta = mesh.boundary[1:] - mesh.boundary[:-1]
    mesh.oneoverdx = 1.0 / mesh.xdelta
    if par.coordsys == 'cartesian':
        mesh.coordinate = 0.5 * (mesh.boundary[1:] + mesh.boundary[:-1])
        if hasattr(par, 'area'):
            mesh.area = np.ones(len(mesh.xdelta)) * par.area
        else:
            mesh.area = np.ones(len(mesh.xdelta))
        mesh.vol = mesh.xdelta * mesh.area
    elif par.coordsys == 'spherical':
        mesh.area = (mesh.boundary[:-1] ** 2) * 4.0 * np.pi
        mesh.vol = np.absolute((mesh.boundary[1:] ** 3 - mesh.boundary[:-1] ** 3)) * 4.0 * np.pi / 3.0
        vol_denom = mesh.boundary[1:] ** 3 - mesh.boundary[:-1] ** 3
        mesh.coordinate = 0.5 * (mesh.boundary[1:] + mesh.boundary[:-1])
        nonzero_vol_denom = vol_denom != 0.0
        mesh.coordinate[nonzero_vol_denom] = 0.75 * (
            mesh.boundary[1:][nonzero_vol_denom] ** 4 - mesh.boundary[:-1][nonzero_vol_denom] ** 4
        ) / vol_denom[nonzero_vol_denom]
        for ig in range(len(mesh.vol)):
            if (mesh.boundary[ig] < 0.0) and (mesh.boundary[ig + 1] > 0.0):
                mesh.vol[ig] = (mesh.boundary[ig + 1] ** 3) * 4.0 * np.pi / 3.0
                mesh.coordinate[ig] = 0.75 * mesh.boundary[ig + 1]
                mesh.area[ig] = 0.0
    else:
        raise ValueError("coordsys unknown: %s" % par.coordsys)


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
    radius = mesh.coordinate[interior].to(unyt.kpc)
    xHI = np.asarray(fluid.xHI[interior])

    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0 * unyt.kpc
    if np.all(ionized):
        return mesh.boundary[interior.stop].to(unyt.kpc)

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
    temperature = fluid.temp_code[interior].to_value(unyt.K)
    ionized = 1.0 - xHI
    if np.sum(ionized) <= 0.0:
        return 0.0
    return float(np.sum(ionized * temperature) / np.sum(ionized))


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(fluid.time.to_value(unyt.Myr))
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
    interior = interior_slice(par)
    code_units_obj = CodeUnits.from_mapping(config.get('CodeUnits'))
    if hasattr(mesh.coordinate[interior], 'to_value'):
        radius_kpc = mesh.coordinate[interior].to_value(unyt.kpc)
    else:
        radius_kpc = np.asarray(mesh.coordinate[interior], dtype=float)
    radius = radius_kpc * unyt.kpc
    snapshot = history.get('reference_snapshot', None)
    if snapshot is None:
        xHI = np.asarray(fluid.xHI[interior], dtype=float)
        if hasattr(fluid.temp_code[interior], 'to_value'):
            temperature_cgs_K = fluid.temp_code[interior].to_value(unyt.K)
        else:
            temperature_cgs_K = np.asarray(fluid.temp_code[interior], dtype=float)
        if hasattr(fluid.time, 'to_value'):
            profile_time_Myr = float(fluid.time.to_value(unyt.Myr))
        else:
            profile_time_Myr = float(np.asarray(fluid.time, dtype=float))
    else:
        radius_kpc = snapshot['radius_kpc']
        xHI = snapshot['xHI']
        temperature_cgs_K = snapshot['temperature_cgs_K']
        profile_time_Myr = snapshot['time_Myr']
    xHII = 1.0 - xHI
    plot_radius_max = config.get('plot_radius_max', config['boxsize']).to_value(unyt.kpc)
    reference_radius_unit = config.get('reference_radius_unit', 5.4 * unyt.kpc)
    temperature_reference = load_log_reference_profile(
        config.get('temperature_reference_filename', None),
        reference_radius_unit,
    )
    neutral_fraction_reference = load_log_reference_profile(
        config.get('neutral_fraction_reference_filename', None),
        reference_radius_unit,
    )
    alpha_B = config.get('alpha_B_coefficient')
    if alpha_B is not None:
        xHI_analytic = sa.neutral_fraction_profile(
            radius,
            config['hydrogen_number_density'],
            config['sigma_gamma'],
            alpha_B,
            config['source_photon_rate'],
            inner_radius=config['analytic_inner_radius'],
        )
        xHII_analytic = 1.0 - xHI_analytic
        radius_stromgren = sa.stromgren_radius(
            config['source_photon_rate'],
            config['hydrogen_number_density'],
            alpha_B,
        ).to(unyt.kpc)
        analytic_front = sa.ionization_front_radius(
            np.asarray(history['time_Myr']) * unyt.Myr,
            config['source_photon_rate'],
            config['hydrogen_number_density'],
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
