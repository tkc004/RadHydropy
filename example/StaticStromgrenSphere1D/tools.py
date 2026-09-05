"""Helper utilities for the static Stromgren sphere example."""

from types import SimpleNamespace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.thermo_networks.hydrogen as rth
from radhydropy.fluid import Fluid
import radhydropy.io as rio
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits
import stromgren_analytic as sa


def build_static_problem(config):
    par_config = config['par']
    simulation = par_config['simulation']
    mesh_config = par_config['mesh']
    boundary = par_config['boundary']
    units = par_config['units']
    chemistry = par_config.get('chemistry', {})
    thermo = par_config.get('thermochemistry', {})
    radiation = par_config.get('radiation', {})
    output = par_config.get('output', {})
    initial = config['initial_condition']
    code_units_obj = CodeUnits.from_mapping(units['CodeUnits'])
    par = SimpleNamespace(
        coordsys=simulation.get('coordinate_system', 'spherical'),
        boundcond=boundary.get('condition', 'OpenSph'),
        nogrid=mesh_config['grid_cells'],
        noghost=mesh_config.get('ghost_cells', 2),
        boxsize=initial['boxsize'],
        verbose=par_config.get('diagnostics', {}).get('verbose', 0),
        outdir=output.get('directory', '.'),
        outfileprefix=output.get('filename_prefix', 'Output'),
        savedir=output.get('savedir', output.get('directory', '.')),
        area=mesh_config.get('area', 1.0 * unyt.cm**2),
        hydrogen_chemistry=thermo.get('hydrogen_chemistry', True),
        hydrogen_mass_fraction=chemistry.get('hydrogen_mass_fraction', 1.0),
        hydrogen_xHI_initial=chemistry.get('hydrogen_xHI_initial', 1.0),
        hydrogen_xHI_inflow=chemistry.get('hydrogen_xHI_inflow', 1.0),
        hydrogen_xHI_outflow=chemistry.get('hydrogen_xHI_outflow', 1.0),
        hydrogen_source_CFL=thermo.get('hydrogen_source_CFL', 0.1),
        hydrogen_source_dtmin=thermo.get('hydrogen_source_dtmin', 1.0e-3 * unyt.Myr),
        hydrogen_update_mu=thermo.get('hydrogen_update_mu', False),
        hydrogen_thermal_coupling=thermo.get('hydrogen_thermal_coupling', False),
        hydrogen_recombination=thermo.get('hydrogen_recombination', True),
        hydrogen_collisional_ionization=thermo.get('hydrogen_collisional_ionization', False),
        hydrogen_alpha_B=thermo.get('hydrogen_alpha_B'),
        hydrogen_beta=thermo.get('hydrogen_beta', 0.0 * unyt.cm**3 / unyt.s),
        hydrogen_radiation_field=thermo.get('hydrogen_radiation_field', False),
        hydrogen_radiation_evolution=thermo.get('hydrogen_radiation_evolution', False),
        hydrogen_ngamma_initial=thermo.get('hydrogen_ngamma_initial', 0.0 / unyt.cm**3),
        hydrogen_sigma_gamma=thermo.get('hydrogen_sigma_gamma'),
        hydrogen_epsilon_gamma=thermo.get('hydrogen_epsilon_gamma', 0.0 * unyt.erg),
        radiative_transfer=radiation.get('radiative_transfer', True),
        radiative_transfer_method=radiation.get('radiative_transfer_method', 'long_characteristics'),
        radiative_transfer_temporal_scheme=radiation.get(
            'radiative_transfer_temporal_scheme',
            'instantaneous',
        ),
        c2ray_max_iterations=radiation.get('c2ray_max_iterations', 32),
        c2ray_convergence_tolerance=radiation.get(
            'c2ray_convergence_tolerance',
            1.0e-6,
        ),
        c2ray_relaxation=radiation.get('c2ray_relaxation', 1.0),
        c2ray_nonconvergence=radiation.get('c2ray_nonconvergence', 'raise'),
        radiative_transfer_boundary_flux=radiation.get(
            'radiative_transfer_boundary_flux',
            0.0 / (unyt.cm**2 * unyt.s),
        ),
        radiative_transfer_source_photon_rate=radiation.get('radiative_transfer_source_photon_rate'),
        radiative_transfer_direction=radiation.get('radiative_transfer_direction', 1),
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
        initial['boxsize'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm

    fluid = Fluid()
    fluid.rho_code = (
        np.ones(par.nogrid)
        * initial['hydrogen_number_density']
        * unyt.mp
    ).to(unyt.g / unyt.cm**3)
    fluid.vel_code = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp_code = np.ones(par.nogrid) * 1.0e4 * unyt.K
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.ngamma_code = np.ones(par.nogrid) * thermo.get(
        'hydrogen_ngamma_initial',
        0.0 / unyt.cm**3,
    )
    fluid.SetFluidTime(0.0 * unyt.Myr)

    solver = Solver()
    return par, mesh, fluid, solver


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


def write_initial_condition(config):
    """Build the raw IC state and write it to ``ICfilename``."""
    par, mesh, fluid, solver = build_static_problem(config)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    filename = config['par']['simulation']['initial_condition_filename']
    Path(filename).unlink(missing_ok=True)
    rio.writehdf5(sim, filename)


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_static_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    code_units_obj = par.CodeUnits
    par.time = unyt.unyt_array(np.asarray(par.time_code, dtype=float), code_units_obj.time_unit)
    par.boxsize = unyt.unyt_array(np.asarray(par.box_size_code, dtype=float), code_units_obj.length_unit)
    mesh.boundary = unyt.unyt_array(np.asarray(mesh.boundary, dtype=float), code_units_obj.length_unit)
    fluid.rho_code = unyt.unyt_array(np.asarray(fluid.rho_code, dtype=float), code_units_obj.density_unit)
    fluid.vel_code = unyt.unyt_array(np.asarray(fluid.vel_code, dtype=float), code_units_obj.velocity_unit)
    fluid.temp_code = unyt.unyt_array(np.asarray(fluid.temp_code, dtype=float), code_units_obj.temperature_unit)
    fluid.time_code = par.time.copy()
    if hasattr(fluid, 'ngamma_code'):
        fluid.ngamma_code = unyt.unyt_array(
            np.asarray(fluid.ngamma_code, dtype=float),
            code_units_obj.number_density_unit,
        )
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


def ionized_hydrogen_atoms(mesh, fluid, par):
    interior = interior_slice(par)
    nH = rth._cgs_hydrogen_number_density(
        fluid.rho_code[interior].to_value(unyt.g / unyt.cm**3),
        par.hydrogen_mass_fraction,
    )
    ionized_fraction = 1.0 - np.asarray(fluid.xHI[interior])
    ionized_atoms = np.sum(ionized_fraction * nH * mesh.vol[interior])
    return ionized_atoms.to_value('')


def photons_in_volume(mesh, fluid, par):
    interior = interior_slice(par)
    photon_count = np.sum(fluid.ngamma_code[interior] * mesh.vol[interior])
    return photon_count.to_value('')


def total_recombination_rate(mesh, fluid, par):
    interior = interior_slice(par)
    nH = rth._cgs_hydrogen_number_density(
        fluid.rho_code[interior].to_value(unyt.g / unyt.cm**3),
        par.hydrogen_mass_fraction,
    )
    ionized_fraction = 1.0 - np.asarray(fluid.xHI[interior])
    rate = np.sum(
        par.hydrogen_alpha_B
        * ionized_fraction**2
        * nH**2
        * mesh.vol[interior]
    )
    return rate.to(1.0 / unyt.s)


def append_history(history, mesh, fluid, par, config, recombined_photons):
    radiation = config['par']['radiation']
    history['time_Myr'].append(fluid.time_code.to_value(unyt.Myr))
    history['front_radius_kpc'].append(
        ionization_front_position(mesh, fluid, par).to_value(unyt.kpc)
    )
    history['injected_photons'].append(
        (radiation['radiative_transfer_source_photon_rate'] * fluid.time_code).to_value('')
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
    if hasattr(mesh.coordinate[interior], 'to_value'):
        radius_kpc = mesh.coordinate[interior].to_value(unyt.kpc)
    else:
        radius_kpc = np.asarray(mesh.coordinate[interior], dtype=float)
    radius = radius_kpc * unyt.kpc
    plot_radius_max = example.get('plot_radius_max', initial['boxsize']).to_value(unyt.kpc)
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
    plot_radius_max = example.get('plot_radius_max', initial['boxsize']).to_value(unyt.kpc)
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
