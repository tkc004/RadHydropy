"""Utilities for the early isothermal H II region expansion example."""

import glob
import os
from types import SimpleNamespace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import example_utils as eu

import radhydropy.radiative_transfer as rrt
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.thermo_networks.hydrogen as rth
import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits


def build_problem(config):
    code_units = CodeUnits.from_mapping(config.get('CodeUnits'))
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=config['number_of_cells'],
        noghost=2,
        boxsize=config['boxsize'],
        outdir=config.get('outdir', '.'),
        outfileprefix=config.get('outfileprefix', 'Output'),
        savedir=config.get('savedir', config.get('outdir', '.')),
        outputtimefilename=config.get('outputtimefilename', None),
        timesim=config.get('final_time', config.get('timesim', None)),
        area=1.0 * unyt.cm**2,
        EOStype='isothermal',
        gamma=1.0,
        CFL=config['hydro_cfl'],
        order=config['hydro_order'],
        dtmin=1.0e-10 * unyt.Myr,
        dtmax=config['hydro_timestep_max'],
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_xHI_initial=1.0,
        hydrogen_xHI_inflow=1.0,
        hydrogen_xHI_outflow=1.0,
        hydrogen_source_CFL=config['source_cfl'],
        hydrogen_source_dtmin=config['source_timestep_min'],
        hydrogen_update_mu=True,
        hydrogen_thermal_coupling=False,
        hydrogen_recombination=True,
        hydrogen_collisional_ionization=False,
        hydrogen_alpha_B=config['alpha_B_coefficient'],
        hydrogen_beta=0.0 * unyt.cm**3 / unyt.s,
        hydrogen_radiation_field=False,
        hydrogen_radiation_evolution=False,
        hydrogen_ngamma_initial=0.0 / unyt.cm**3,
        hydrogen_sigma_gamma=config['sigma_gamma'],
        hydrogen_epsilon_gamma=0.0 * unyt.erg,
        radiative_transfer=True,
        radiative_transfer_method='long_characteristics',
        radiative_transfer_boundary_flux=0.0 / (unyt.cm**2 * unyt.s),
        radiative_transfer_source_photon_rate=config['source_photon_rate'],
        radiative_transfer_direction=1,
        CodeUnits=code_units,
        code_units=code_units,
        unit_system=code_units.unit_system if code_units is not None else None,
    )

    mesh = Mesh()
    mesh.boundary = np.linspace(
        0.0,
        config['boxsize'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm
    mesh.SetUpMesh(par)

    fluid = Fluid()
    fluid.eos = EOS(par.EOStype, par.gamma)
    fluid.rho = np.ones(par.nogrid) * config['rho_initial']
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * config['neutral_temperature']
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.SetUpFluid(par)
    fluid.SetFluidTime(0.0 * unyt.Myr)

    solver = Solver()
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid)
    result = rrt.trace_long_characteristics(
        mesh,
        fluid.rho,
        fluid.xHI,
        hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
        sigma_gamma=rrt._optional_photon_quantity(
            getattr(par, 'hydrogen_sigma_gamma', None),
            rrt.rh.DEFAULT_SIGMA_GAMMA,
            unyt.cm**2,
        ),
        boundary_flux=rrt._as_photon_flux(
            getattr(par, 'radiative_transfer_boundary_flux', None)
        ),
        source_photon_rate=rrt._as_photon_rate(
            getattr(par, 'radiative_transfer_source_photon_rate', None)
        ),
        direction=getattr(par, 'radiative_transfer_direction', 1),
        coordsys=getattr(mesh, 'coordsys', 'cartesian'),
    )
    fluid.ngamma[:] = result.cell_photon_density.to(fluid.ngamma.units)
    return par, mesh, fluid, solver


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    return par, mesh, fluid


def load_parameters(config_filename, rundir=None):
    from radhydropy.example_config import load_example_parameters

    config_filename = Path(config_filename)
    runparams, icparams = load_example_parameters(config_filename, rundir)
    eu.clean_previous_outputs(runparams)
    return runparams, icparams


def load_labeled_density_snapshots(outputfilenames, config, output_specs):
    snapshots = []
    for index, spec in enumerate(output_specs):
        label = spec.get('label', None)
        if label is None:
            continue
        out_par, out_mesh, out_fluid = load_output_state(outputfilenames[index], config)
        snapshots.append(
            (
                label,
                density_snapshot(out_mesh, out_fluid, out_par),
            )
        )
    return snapshots


def output_files(outdir, outfileprefix):
    pattern = os.path.join(outdir, f'{outfileprefix}_*.hdf5')
    return sorted(glob.glob(pattern))


def interior_slice(par):
    return slice(par.noghost, par.noghost + par.nogrid)


def refresh_state(mesh, fluid, par, solver):
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid)


def apply_piecewise_isothermal_state(mesh, fluid, par, solver, config):
    fluid.eos.apply_piecewise_isothermal_state(
        fluid,
        par,
        config['neutral_temperature'],
        config['ionized_temperature'],
    )
    refresh_state(mesh, fluid, par, solver)


def make_piecewise_isothermal_step_backend(sim, config):
    def step_backend(dt=None, mode='hydro_sources', advect_chemistry=True):
        result = sim.Step(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
        )
        apply_piecewise_isothermal_state(
            sim.mesh,
            sim.fluid,
            sim.par,
            sim.solver,
            config,
        )
        return result

    return step_backend


def ionization_front_position(mesh, fluid, par, ionized_fraction=0.5):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to_value(unyt.pc)
    xHII = 1.0 - np.asarray(fluid.xHI[interior], dtype=float)

    ionized = xHII >= ionized_fraction
    if not np.any(ionized):
        return 0.0
    if np.all(ionized):
        return radius[-1]

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHII[left]
    x_right = xHII[right]
    if x_right == x_left:
        return radius[left]

    weight = (ionized_fraction - x_left) / (x_right - x_left)
    return radius[left] + weight * (radius[right] - radius[left])


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(fluid.time.to_value(unyt.Myr))
    history['front_radius_pc'].append(ionization_front_position(mesh, fluid, par))


def load_history_from_outputs(outputfilenames, config):
    history = {
        'time_Myr': [],
        'front_radius_pc': [],
    }
    for outputfilename in outputfilenames:
        par, mesh, fluid = load_output_state(outputfilename, config)
        append_history(history, mesh, fluid, par)
    return history


def density_snapshot(mesh, fluid, par):
    interior = interior_slice(par)
    return {
        'time_Myr': fluid.time.to_value(unyt.Myr),
        'radius_pc': mesh.coordinate[interior].to_value(unyt.pc).copy(),
        'density_g_cm3': fluid.rho[interior].to_value(unyt.g / unyt.cm**3).copy(),
    }


def front_radius_at_time(history, time):
    time_myr = np.asarray(history['time_Myr'])
    front_radius_pc = np.asarray(history['front_radius_pc'])
    target_time_myr = time.to_value(unyt.Myr)
    if target_time_myr < time_myr[0] or target_time_myr > time_myr[-1]:
        raise ValueError('requested time is outside the recorded history')

    return np.interp(target_time_myr, time_myr, front_radius_pc) * unyt.pc


def stromgren_radius(config):
    nH = rth._cgs_hydrogen_number_density(
        config['rho_initial'].to_value(unyt.g / unyt.cm**3),
        hydrogen_mass_fraction=1.0,
    )
    radius = (
        3.0
        * config['source_photon_rate']
        / (4.0 * np.pi * config['alpha_B_coefficient'] * nH**2)
    ) ** (1.0 / 3.0)
    return radius.to(unyt.pc)


def neutral_sound_speed(config):
    return np.sqrt(
        unyt.kb * config['neutral_temperature'] / unyt.mp
    ).to(unyt.cm / unyt.s)


def stagnation_radius(config):
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = config['ionized_sound_speed'].to(unyt.cm / unyt.s)
    return (
        (ionized_sound_speed / neutral_sound_speed(config)) ** (4.0 / 3.0)
        * radius_stromgren
    ).to(unyt.pc)


def spitzer_radius(time, config):
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = config['ionized_sound_speed'].to(unyt.cm / unyt.s)
    factor = (
        1.0
        + 7.0
        * ionized_sound_speed
        * time.to(unyt.s)
        / (4.0 * radius_stromgren.to(unyt.cm))
    )
    return (radius_stromgren * factor**(4.0 / 7.0)).to(unyt.pc)


def hosokawa_inutsuka_radius(time, config):
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = config['ionized_sound_speed'].to(unyt.cm / unyt.s)
    factor = (
        1.0
        + 7.0
        * np.sqrt(4.0 / 3.0)
        * ionized_sound_speed
        * time.to(unyt.s)
        / (4.0 * radius_stromgren.to(unyt.cm))
    )
    return (radius_stromgren * factor**(4.0 / 7.0)).to(unyt.pc)


def save_front_plot(history, config, figure_filename):
    time = np.asarray(history['time_Myr']) * unyt.Myr
    time_myr = time.to_value(unyt.Myr)
    front_radius_pc = np.asarray(history['front_radius_pc'])
    stromgren_radius_pc = stromgren_radius(config).to_value(unyt.pc)
    radius_spitzer_pc = spitzer_radius(time, config).to_value(unyt.pc)
    radius_hosokawa_inutsuka_pc = hosokawa_inutsuka_radius(
        time,
        config,
    ).to_value(unyt.pc)
    show_stagnation_radius = config.get('show_stagnation_radius', False)
    if show_stagnation_radius:
        radius_stagnation_pc = stagnation_radius(config).to_value(unyt.pc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time_myr,
        front_radius_pc,
        color='tab:blue',
        lw=2.0,
        label=r'RadHydropy $x_{\rm HII}=0.5$',
    )
    ax.plot(
        time_myr,
        radius_spitzer_pc,
        color='tab:orange',
        lw=1.8,
        ls='--',
        label=(
            r'Spitzer, $c_i=%.2f$ km s$^{-1}$'
            % config['ionized_sound_speed'].to_value(unyt.km / unyt.s)
        ),
    )
    ax.plot(
        time_myr,
        radius_hosokawa_inutsuka_pc,
        color='tab:green',
        lw=1.8,
        ls=':',
        label='Hosokawa-Inutsuka',
    )
    ax.axhline(
        stromgren_radius_pc,
        color='black',
        lw=1.4,
        ls='--',
        label=r'$R_{\rm S}$',
    )
    if show_stagnation_radius:
        ax.axhline(
            radius_stagnation_pc,
            color='tab:red',
            lw=1.6,
            ls='-.',
            label=r'$R_{\rm stag}$',
        )
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Ionization-front radius [pc]')
    ax.set_xlim(0.0, config['final_time'].to_value(unyt.Myr))
    radius_limits = (
        1.05 * np.max(front_radius_pc),
        1.05 * np.max(radius_spitzer_pc),
        1.05 * np.max(radius_hosokawa_inutsuka_pc),
        1.1 * stromgren_radius_pc,
    )
    if show_stagnation_radius:
        radius_limits += (1.1 * radius_stagnation_pc,)
    ax.set_ylim(0.0, max(radius_limits))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_density_profile_plot(snapshot, config, figure_filename):
    time = snapshot['time_Myr'] * unyt.Myr
    radius_pc = np.asarray(snapshot['radius_pc'])
    density_g_cm3 = np.asarray(snapshot['density_g_cm3'])
    spitzer_radius_pc = spitzer_radius(time, config).to_value(unyt.pc)
    hosokawa_inutsuka_radius_pc = hosokawa_inutsuka_radius(
        time,
        config,
    ).to_value(unyt.pc)
    show_stagnation_radius = config.get('show_stagnation_radius', False)
    if show_stagnation_radius:
        radius_stagnation_pc = stagnation_radius(config).to_value(unyt.pc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        radius_pc,
        density_g_cm3,
        color='tab:blue',
        lw=2.0,
        label='RadHydropy',
    )
    ax.axvline(
        spitzer_radius_pc,
        color='tab:orange',
        lw=1.8,
        ls='--',
        label='Spitzer',
    )
    ax.axvline(
        hosokawa_inutsuka_radius_pc,
        color='tab:green',
        lw=1.8,
        ls=':',
        label='Hosokawa-Inutsuka',
    )
    if show_stagnation_radius:
        ax.axvline(
            radius_stagnation_pc,
            color='tab:red',
            lw=1.6,
            ls='-.',
            label=r'$R_{\rm stag}$',
        )
    ax.set_yscale('log')
    ax.set_xlabel('Radius [pc]')
    ax.set_ylabel(r'Density [g cm$^{-3}$]')
    ax.set_title('Density profile at %.3f Myr' % snapshot['time_Myr'])
    ax.set_xlim(0.0, config['boxsize'].to_value(unyt.pc))
    positive_density = density_g_cm3[density_g_cm3 > 0.0]
    if positive_density.size:
        ymin = 10.0 ** np.floor(np.log10(0.8 * np.min(positive_density)))
        ymax = 10.0 ** np.ceil(np.log10(1.2 * np.max(positive_density)))
        ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_density_profile_plots(snapshots, config, figure_filenames):
    if len(snapshots) != len(figure_filenames):
        raise ValueError('density snapshots and figure filenames differ in length')
    for snapshot, figure_filename in zip(snapshots, figure_filenames):
        save_density_profile_plot(snapshot, config, figure_filename)
