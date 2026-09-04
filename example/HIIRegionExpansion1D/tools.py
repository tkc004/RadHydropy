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
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value


def build_problem(config):
    """Build the H II initial state from direct nested configuration groups."""
    par_config = config['par']
    simulation = par_config['simulation']
    mesh_config = par_config['mesh']
    hydro = par_config['hydrodynamics']
    boundary = par_config['boundary']
    timestep = par_config['timestep']
    output = par_config['output']
    chemistry = par_config['chemistry']
    thermochemistry = par_config['thermochemistry']
    radiation = par_config['radiation']
    initial = config['initial_condition']
    code_units = CodeUnits.from_mapping(par_config['units']['CodeUnits'])
    grid_cells = initial['grid_cells']
    box_size = initial['boxsize']
    par = SimpleNamespace(
        coordsys=simulation['coordinate_system'],
        boundcond=boundary['condition'],
        nogrid=grid_cells,
        noghost=mesh_config.get('ghost_cells', 2),
        boxsize=box_size,
        outdir=output['directory'],
        outfileprefix=output['filename_prefix'],
        savedir=output.get('savedir', output['directory']),
        outputtimefilename=output.get('time_list_filename'),
        verbose=par_config.get('diagnostics', {}).get('verbose', 0),
        timesim=simulation['final_time'],
        area=mesh_config.get('area', 1.0 * unyt.cm**2),
        EOStype=hydro['eos_type'],
        gamma=hydro['gamma'],
        CFL=hydro['CFL'],
        order=hydro['order'],
        dtmin=timestep['dtmin'],
        dtmax=timestep['dtmax'],
        hydrogen_chemistry=thermochemistry.get('hydrogen_chemistry', True),
        hydrogen_mass_fraction=chemistry.get('hydrogen_mass_fraction', 1.0),
        hydrogen_xHI_initial=chemistry.get('hydrogen_xHI_initial', 1.0),
        hydrogen_xHI_inflow=chemistry.get('hydrogen_xHI_inflow', 1.0),
        hydrogen_xHI_outflow=chemistry.get('hydrogen_xHI_outflow', 1.0),
        hydrogen_source_CFL=timestep.get('hydrogen_source_CFL'),
        hydrogen_source_dtmin=timestep.get('hydrogen_source_dtmin'),
        hydrogen_update_mu=chemistry.get('hydrogen_update_mu', True),
        hydrogen_thermal_coupling=thermochemistry.get('hydrogen_thermal_coupling', False),
        hydrogen_recombination=thermochemistry.get('hydrogen_recombination', True),
        hydrogen_collisional_ionization=thermochemistry.get('hydrogen_collisional_ionization', False),
        hydrogen_alpha_B=chemistry['hydrogen_alpha_B'],
        hydrogen_beta=chemistry['hydrogen_beta'],
        hydrogen_radiation_field=radiation.get('hydrogen_radiation_field', False),
        hydrogen_radiation_evolution=radiation.get('hydrogen_radiation_evolution', False),
        hydrogen_ngamma_initial=radiation.get('hydrogen_ngamma_initial', 0.0 / unyt.cm**3),
        hydrogen_sigma_gamma=radiation['hydrogen_sigma_gamma'],
        hydrogen_epsilon_gamma=radiation['hydrogen_epsilon_gamma'],
        radiative_transfer=radiation.get('radiative_transfer', True),
        radiative_transfer_method=radiation.get('method', 'long_characteristics'),
        radiative_transfer_temporal_scheme=radiation.get('temporal_scheme', 'instantaneous'),
        radiative_transfer_c2ray_max_iterations=radiation.get('c2ray_max_iterations', 32),
        radiative_transfer_c2ray_tolerance=radiation.get('c2ray_tolerance', 1.0e-6),
        radiative_transfer_c2ray_relaxation=radiation.get('c2ray_relaxation', 1.0),
        radiative_transfer_c2ray_nonconvergence=radiation.get('c2ray_nonconvergence', 'warn'),
        radiative_transfer_boundary_flux=radiation['boundary_flux'],
        radiative_transfer_source_photon_rate=radiation['source_photon_rate'],
        radiative_transfer_direction=radiation.get('direction', 1),
        CodeUnits=code_units,
        unit_system=code_units.unit_system,
    )
    par.mesh = SimpleNamespace(ghost_cells=par.noghost, grid_cells=grid_cells, area=par.area)
    par.simulation = SimpleNamespace(
        coordinate_system=par.coordsys,
        final_time=par.timesim,
        initial_condition_filename=simulation['initial_condition_filename'],
        current_time=initial.get('current_time', 0.0 * unyt.Myr),
        box_size=box_size,
    )
    par.hydrodynamics = SimpleNamespace(
        eos_type=par.EOStype, gamma=par.gamma, CFL=par.CFL, order=par.order,
        riemann_solver=hydro.get('riemann_solver', 'Rusanov'),
    )
    par.units = SimpleNamespace(CodeUnits=code_units)
    par.boundary = SimpleNamespace(condition=par.boundcond)
    par.timestep = SimpleNamespace(dtmin=par.dtmin, dtmax=par.dtmax)
    par.output = SimpleNamespace(
        directory=par.outdir, filename_prefix=par.outfileprefix,
        cadence=output.get('cadence'), time_list_filename=par.outputtimefilename,
    )
    par.radiation = SimpleNamespace(
        radiative_transfer=par.radiative_transfer,
        method=par.radiative_transfer_method,
        temporal_scheme=par.radiative_transfer_temporal_scheme,
        direction=par.radiative_transfer_direction,
        boundary_flux=par.radiative_transfer_boundary_flux,
        source_photon_rate=par.radiative_transfer_source_photon_rate,
    )
    mesh = Mesh()
    mesh.boundary = np.linspace(0.0, box_size.to_value(unyt.cm), grid_cells + 1) * unyt.cm
    fluid = Fluid()
    fluid.eos = EOS(par.EOStype, par.gamma, code_units)
    fluid.rho_code = np.ones(grid_cells) * initial['rho_initial']
    fluid.vel_code = np.zeros(grid_cells) * unyt.cm / unyt.s
    fluid.temp_code = np.ones(grid_cells) * initial['neutral_temperature']
    fluid.mu = np.ones(grid_cells)
    fluid.xHI = np.ones(grid_cells)
    fluid.SetFluidTime(initial.get('current_time', 0.0 * unyt.Myr))
    return par, mesh, fluid, Solver()


def write_initial_condition(config):
    """Build the raw IC state and write it to ``ICfilename``."""
    par, mesh, fluid, _ = build_problem(config)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    icfilename = config['par']['simulation']['initial_condition_filename']
    Path(icfilename).unlink(missing_ok=True)
    rio.writehdf5(sim, icfilename)


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    code_units_obj = par.CodeUnits
    par.Time = np.asarray(par.Time, dtype=float) * code_units_obj.time_unit
    par.BoxSize = np.asarray(par.BoxSize, dtype=float) * code_units_obj.length_unit
    fluid.time = np.asarray(fluid.time, dtype=float) * code_units_obj.time_unit
    mesh.boundary = np.asarray(mesh.boundary, dtype=float) * code_units_obj.length_unit
    fluid.rho_code = np.asarray(fluid.rho_code, dtype=float) * code_units_obj.density_unit
    fluid.vel_code = np.asarray(fluid.vel_code, dtype=float) * code_units_obj.velocity_unit
    fluid.temp_code = np.asarray(fluid.temp_code, dtype=float) * code_units_obj.temperature_unit
    if hasattr(fluid, 'ngamma_code'):
        if not hasattr(fluid.ngamma_code, 'units'):
            fluid.ngamma_code = (
                np.asarray(fluid.ngamma_code, dtype=float)
                * code_units_obj.number_density_unit
            )
    # ``readhdf5`` restores the saved boundary and fluid state, but it does not
    # recompute the derived mesh geometry. Rebuild those cached geometric
    # fields from the loaded boundary so post-processing uses the snapshot's
    # actual coordinates instead of the constructor-time placeholders.
    boundary = mesh.boundary
    if par.coordsys == 'cartesian':
        mesh.xdelta = boundary[1:] - boundary[:-1]
        mesh.oneoverdx = 1.0 / mesh.xdelta
        mesh.coordinate = 0.5 * (boundary[1:] + boundary[:-1])
        if hasattr(par, 'area'):
            mesh.area = np.ones(len(mesh.xdelta)) * par.area
        else:
            mesh.area = np.ones(len(mesh.xdelta))
        mesh.vol = mesh.xdelta * mesh.area
    elif par.coordsys == 'spherical':
        mesh.xdelta = boundary[1:] - boundary[:-1]
        mesh.oneoverdx = 1.0 / mesh.xdelta
        mesh.area = (boundary[:-1] ** 2) * 4.0 * np.pi
        mesh.vol = np.absolute((boundary[1:] ** 3 - boundary[:-1] ** 3)) * 4.0 * np.pi / 3.0
        vol_denom = boundary[1:] ** 3 - boundary[:-1] ** 3
        mesh.coordinate = 0.5 * (boundary[1:] + boundary[:-1])
        nonzero_vol_denom = vol_denom != 0.0
        mesh.coordinate[nonzero_vol_denom] = 0.75 * (
            boundary[1:][nonzero_vol_denom] ** 4 - boundary[:-1][nonzero_vol_denom] ** 4
        ) / vol_denom[nonzero_vol_denom]
        if np.any((boundary[:-1] < 0.0) & (boundary[1:] > 0.0)):
            crossing = np.where((boundary[:-1] < 0.0) & (boundary[1:] > 0.0))[0]
            for ig in crossing:
                mesh.vol[ig] = (boundary[ig + 1] ** 3) * 4.0 * np.pi / 3.0
                mesh.coordinate[ig] = 0.75 * boundary[ig + 1]
                mesh.area[ig] = 0.0
    return par, mesh, fluid


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
    mesh = getattr(par, 'mesh', None)
    ghost_cells = getattr(mesh, 'ghost_cells', getattr(par, 'noghost', 0))
    grid_cells = getattr(mesh, 'grid_cells', getattr(par, 'nogrid', None))
    return slice(ghost_cells, ghost_cells + grid_cells)


def refresh_state(mesh, fluid, par, solver):
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid, verbose=getattr(par, 'verbose', 0))


def apply_piecewise_isothermal_state(mesh, fluid, par, solver, config):
    initial_condition = config['initial_condition']
    fluid.eos.apply_piecewise_isothermal_state(
        fluid,
        par,
        initial_condition['neutral_temperature'],
        initial_condition['ionized_temperature'],
    )
    refresh_state(mesh, fluid, par, solver)


def time_myr(value, code_units):
    myr_in_s = (1.0 * unyt.Myr).to_value(unyt.s)
    return float(code_quantity_to_cgs(value, code_units, 'time_s') / myr_in_s)


def print_startup_diagnostics(sim, config, initial_condition):
    """Print the main physical scales before the long run starts."""
    initial_condition = config['initial_condition']
    interior = interior_slice(sim.par)
    rho_code = np.asarray(sim.fluid.rho_code[interior], dtype=float)
    vel_code = np.asarray(sim.fluid.vel_code[interior], dtype=float)
    temp_code = np.asarray(sim.fluid.temp_code[interior], dtype=float)
    xHI = np.asarray(sim.fluid.xHI[interior], dtype=float)
    ngamma_code = np.asarray(sim.fluid.ngamma_code[interior], dtype=float) if hasattr(sim.fluid, 'ngamma_code') else None
    code_units_obj = sim.par.units.CodeUnits
    rho_cgs = code_quantity_to_cgs(
        rho_code,
        code_units_obj,
        'density_cgs_g_cm3',
    )
    ngamma_cgs = None
    if ngamma_code is not None:
        ngamma_cgs = code_quantity_to_cgs(ngamma_code, code_units_obj, 'number_density_cgs_cm3')

    print('--- Startup diagnostics ---')
    print('cells = %d' % sim.par.mesh.grid_cells)
    print('time = %.6e Myr' % time_myr(sim.fluid.time, code_units_obj))
    print('rho range = [%.3e, %.3e] g/cm^3' % (np.min(rho_cgs), np.max(rho_cgs)))
    print('vel max abs = %.3e km/s' % (np.max(np.abs(vel_code)) / 1.0e5))
    print('temperature range = [%.3e, %.3e] K' % (np.min(temp_code), np.max(temp_code)))
    print('neutral fraction range = [%.3e, %.3e]' % (np.min(xHI), np.max(xHI)))
    if ngamma_code is not None:
        print('ngamma_cgs_cm3 range = [%.3e, %.3e] code units' % (np.min(ngamma_code), np.max(ngamma_code)))
        if ngamma_cgs is not None:
            print('ngamma_cgs_cm3 range = [%.3e, %.3e] cm^-3' % (np.min(ngamma_cgs), np.max(ngamma_cgs)))
            boundary_cgs_cm = code_quantity_to_cgs(
                sim.mesh.boundary[interior.start : interior.start + 2],
                code_units_obj,
                'length_cgs_cm',
            )
            inner_radius_cgs_cm = 0.5 * (boundary_cgs_cm[0] + boundary_cgs_cm[1])
            thin_estimate = initial_condition['source_photon_rate'].to_value(1 / unyt.s) / (
                4.0 * np.pi * inner_radius_cgs_cm**2 * unyt.c.to_value(unyt.cm / unyt.s)
            )
            print('optically thin inner-cell ngamma_cgs_cm3 estimate = %.3e cm^-3' % thin_estimate)
    print('neutral sound speed = %.3e km/s' % neutral_sound_speed(config).to_value(unyt.km / unyt.s))
    print(
        'ionized sound speed (config) = %.3e km/s'
        % initial_condition['ionized_sound_speed'].to_value(unyt.km / unyt.s)
    )
    print('stromgren radius = %.3e pc' % stromgren_radius(config).to_value(unyt.pc))
    print('stagnation radius = %.3e pc' % stagnation_radius(config).to_value(unyt.pc))
    print(
        'Spitzer radius at final time = %.3e pc'
        % spitzer_radius(initial_condition['final_time'], config).to_value(unyt.pc)
    )
    print(
        'Hosokawa-Inutsuka radius at final time = %.3e pc'
        % hosokawa_inutsuka_radius(initial_condition['final_time'], config).to_value(unyt.pc)
    )
    try:
        hydro_dt = sim.solver.GetTimeStep(sim.mesh, sim.fluid, sim.par)
        hydro_dt_s = hydro_dt.to_value(unyt.s) if hasattr(hydro_dt, 'to_value') else float(hydro_dt)
        print('hydro timestep estimate = %.3e s' % hydro_dt_s)
    except Exception as exc:
        print('hydro timestep estimate failed: %s' % exc)
        hydro_dt_s = None
    try:
        source_dt, thermal_rate = sim.solver.GetSourceTimestepFast(
            sim.mesh,
            sim.fluid,
            sim.par,
            sim.par.timestep.dtmax,
        )
        source_dt_s = source_dt.to_value(unyt.s) if hasattr(source_dt, 'to_value') else float(source_dt)
        print('source timestep estimate = %.3e s' % source_dt_s)
        if hydro_dt_s is not None and source_dt_s > 0.0:
            print('estimated source substeps per hydro step = %.1f' % (hydro_dt_s / source_dt_s))
        if thermal_rate is not None:
            print(
                'thermal rate range = [%.3e, %.3e]'
                % (
                    np.min(np.asarray(thermal_rate, dtype=float)),
                    np.max(np.asarray(thermal_rate, dtype=float)),
                )
            )
    except Exception as exc:
        print('source timestep estimate failed: %s' % exc)


def make_logging_step_backend(sim, config, max_logged_steps=5):
    """Wrap the isothermal step backend with a short startup trace."""
    base_step_backend = make_piecewise_isothermal_step_backend(sim, config)
    code_units_obj = sim.par.units.CodeUnits
    state = {'count': 0}
    interior = interior_slice(sim.par)

    def step_backend(dt=None, mode='hydro_sources', advect_chemistry=True):
        step_index = state['count']
        should_log = step_index < max_logged_steps
        if should_log:
            print(
                '--- step %d begin: time=%.6e Myr dt=%s mode=%s ---'
                % (step_index + 1, time_myr(sim.fluid.time, code_units_obj), dt, mode)
            )
        result = base_step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
        )
        if should_log:
            vel_code = np.asarray(sim.fluid.vel_code[interior], dtype=float)
            rho_code = np.asarray(sim.fluid.rho_code[interior], dtype=float)
            xHI = np.asarray(sim.fluid.xHI[interior], dtype=float)
            vmax = np.max(np.abs(vel_code)) / 1.0e5
            front_radius = ionization_front_position(sim.mesh, sim.fluid, sim.par)
            print(
                '--- step %d end: time=%.6e Myr hydro_steps=%d source_steps=%d front=%.3e pc vmax=%.3e km/s rho=[%.3e, %.3e] xHI=[%.3e, %.3e] ---'
                % (
                    step_index + 1,
                    time_myr(sim.fluid.time, code_units_obj),
                    result['hydro_steps'],
                    result['source_steps'],
                    front_radius,
                    vmax,
                    np.min(rho_code),
                    np.max(rho_code),
                    np.min(xHI),
                    np.max(xHI),
                )
            )
            if step_index + 1 == max_logged_steps:
                print('--- step logging disabled after %d steps ---' % max_logged_steps)
        state['count'] += 1
        return result

    return step_backend


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


def _value_in_unit(value, unit):
    return np.asarray(quantity_to_value(value, unit), dtype=float)


def _scalar_in_unit(value, unit):
    values = _value_in_unit(value, unit)
    return float(np.reshape(values, -1)[0])


def ionization_front_position(mesh, fluid, par, ionized_fraction=0.5):
    interior = interior_slice(par)
    radius = _value_in_unit(mesh.coordinate[interior], unyt.pc)
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
    history['time_Myr'].append(_scalar_in_unit(fluid.time, unyt.Myr))
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
    ngamma_code = np.asarray(fluid.ngamma_code)
    if ngamma_code.ndim > 1:
        ngamma_code = np.sum(ngamma_code, axis=0)
    return {
        'time_Myr': _scalar_in_unit(fluid.time, unyt.Myr),
        'radius_pc': _value_in_unit(mesh.coordinate[interior], unyt.pc).copy(),
        'density_cgs_g_cm3': _value_in_unit(fluid.rho_code[interior], unyt.g / unyt.cm**3).copy(),
        'radiation_density_cgs_cm3': _value_in_unit(
            ngamma_code[interior], 1.0 / unyt.cm**3
        ).copy(),
    }


def front_radius_at_time(history, time):
    time_myr = np.asarray(history['time_Myr'])
    front_radius_pc = np.asarray(history['front_radius_pc'])
    target_time_myr = time.to_value(unyt.Myr)
    if time_myr.size == 0:
        raise ValueError('history is empty')
    tol = max(1.0e-12 * max(1.0, np.max(np.abs(time_myr)), abs(target_time_myr)), 1.0e-30)
    if target_time_myr < time_myr[0] - tol or target_time_myr > time_myr[-1] + tol:
        raise ValueError('requested time is outside the recorded history')
    target_time_myr = float(np.clip(target_time_myr, time_myr[0], time_myr[-1]))
    return np.interp(target_time_myr, time_myr, front_radius_pc) * unyt.pc


def stromgren_radius(config):
    config = config['initial_condition']
    nH = rth._cgs_hydrogen_number_density(
        config['rho_initial'].to_value(unyt.g / unyt.cm**3),
        hydrogen_mass_fraction=1.0,
    ) * (1.0 / unyt.cm**3)
    radius = (
        3.0
        * config['source_photon_rate']
        / (4.0 * np.pi * config['alpha_B_coefficient'] * nH**2)
    ) ** (1.0 / 3.0)
    return radius.to(unyt.pc)


def neutral_sound_speed(config):
    config = config['initial_condition']
    return np.sqrt(
        unyt.kb * config['neutral_temperature'] / unyt.mp
    ).to(unyt.cm / unyt.s)


def stagnation_radius(config):
    initial_condition = config['initial_condition']
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = initial_condition['ionized_sound_speed'].to(unyt.cm / unyt.s)
    return (
        (ionized_sound_speed / neutral_sound_speed(config)) ** (4.0 / 3.0)
        * radius_stromgren
    ).to(unyt.pc)


def spitzer_radius(time, config):
    initial_condition = config['initial_condition']
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = initial_condition['ionized_sound_speed'].to(unyt.cm / unyt.s)
    factor = (
        1.0
        + 7.0
        * ionized_sound_speed
        * time.to(unyt.s)
        / (4.0 * radius_stromgren.to(unyt.cm))
    )
    return (radius_stromgren * factor**(4.0 / 7.0)).to(unyt.pc)


def hosokawa_inutsuka_radius(time, config):
    initial_condition = config['initial_condition']
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = initial_condition['ionized_sound_speed'].to(unyt.cm / unyt.s)
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
    initial_condition = config['initial_condition']
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
            % initial_condition['ionized_sound_speed'].to_value(unyt.km / unyt.s)
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
    ax.set_xlim(0.0, config['initial_condition']['final_time'].to_value(unyt.Myr))
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
    density_cgs_g_cm3 = np.asarray(snapshot['density_cgs_g_cm3'])
    radiation_density_cgs_cm3 = np.asarray(snapshot['radiation_density_cgs_cm3'])
    spitzer_radius_pc = spitzer_radius(time, config).to_value(unyt.pc)
    hosokawa_inutsuka_radius_pc = hosokawa_inutsuka_radius(
        time,
        config,
    ).to_value(unyt.pc)
    show_stagnation_radius = config.get('show_stagnation_radius', False)
    if show_stagnation_radius:
        radius_stagnation_pc = stagnation_radius(config).to_value(unyt.pc)

    fig, (ax, radiation_ax) = plt.subplots(
        2, 1, figsize=(7.2, 7.2), sharex=True
    )
    ax.plot(
        radius_pc,
        density_cgs_g_cm3,
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
    ax.set_xlim(0.0, config['initial_condition']['boxsize'].to_value(unyt.pc))
    positive_density = density_cgs_g_cm3[density_cgs_g_cm3 > 0.0]
    if positive_density.size:
        ymin = 10.0 ** np.floor(np.log10(0.8 * np.min(positive_density)))
        ymax = 10.0 ** np.ceil(np.log10(1.2 * np.max(positive_density)))
        ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    radiation_ax.plot(
        radius_pc,
        np.where(radiation_density_cgs_cm3 > 0.0, radiation_density_cgs_cm3, np.nan),
        color='tab:purple',
        lw=2.0,
        label='RadHydropy',
    )
    radiation_ax.axvline(spitzer_radius_pc, color='tab:orange', lw=1.8, ls='--')
    radiation_ax.axvline(
        hosokawa_inutsuka_radius_pc, color='tab:green', lw=1.8, ls=':'
    )
    if show_stagnation_radius:
        radiation_ax.axvline(radius_stagnation_pc, color='tab:red', lw=1.6, ls='-.')
    radiation_ax.set_yscale('log')
    radiation_ax.set_xlabel('Radius [pc]')
    radiation_ax.set_ylabel(r'Photon density [cm$^{-3}$]')
    positive_radiation = radiation_density_cgs_cm3[radiation_density_cgs_cm3 > 0.0]
    if positive_radiation.size:
        ymin = 10.0 ** np.floor(np.log10(0.8 * np.min(positive_radiation)))
        ymax = 10.0 ** np.ceil(np.log10(1.2 * np.max(positive_radiation)))
        radiation_ax.set_ylim(ymin, ymax)
    radiation_ax.grid(True, alpha=0.25)
    radiation_ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_density_profile_plots(snapshots, config, figure_filenames):
    if len(snapshots) != len(figure_filenames):
        raise ValueError('density snapshots and figure filenames differ in length')
    for snapshot, figure_filename in zip(snapshots, figure_filenames):
        save_density_profile_plot(snapshot, config, figure_filename)
