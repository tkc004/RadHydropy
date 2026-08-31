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


def _config_value(config, *keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default


def _runtime_values(runparams):
    """Expose nested runtime values to the IC fixture builder."""
    if 'simulation' not in runparams:
        return runparams
    simulation = runparams['simulation']
    mesh = runparams['mesh']
    hydro = runparams['hydrodynamics']
    boundary = runparams['boundary']
    timestep = runparams['timestep']
    output = runparams['output']
    diagnostics = runparams.get('diagnostics', {})
    radiation = runparams.get('radiation', {})
    chemistry = runparams.get('chemistry', {})
    thermochemistry = runparams.get('thermochemistry', {})
    values = {
        'CodeUnits': runparams['units']['CodeUnits'],
        'coordsys': simulation['coordinate_system'],
        'boundcond': boundary['condition'],
        'noghost': mesh['ghost_cells'],
        'outdir': output['directory'],
        'outfileprefix': output['filename_prefix'],
        'savedir': output.get('savedir', output['directory']),
        'outputtimefilename': output.get('time_list_filename'),
        'verbose': diagnostics.get('verbose', 0),
        'timesim': simulation['final_time'],
        'area': mesh.get('area', 1.0 * unyt.cm**2),
        'EOStype': hydro['eos_type'],
        'gamma': hydro['gamma'],
        'CFL': hydro['CFL'],
        'order': hydro['order'],
        'dtmin': timestep['dtmin'],
        'dtmax': timestep['dtmax'],
        'hydrogen_source_CFL': timestep.get('hydrogen_source_CFL'),
        'hydrogen_source_dtmin': timestep.get('hydrogen_source_dtmin'),
        'radiative_transfer_direction': radiation.get('direction', 1),
        'radiative_transfer_boundary_flux': radiation.get('boundary_flux'),
        'radiative_transfer_source_photon_rate': radiation.get('source_photon_rate'),
        'radiative_transfer_method': radiation.get('method', 'long_characteristics'),
        'radiative_transfer_temporal_scheme': radiation.get('temporal_scheme', 'instantaneous'),
    }
    for nested, legacy in (
        ('c2ray_max_iterations', 'radiative_transfer_c2ray_max_iterations'),
        ('c2ray_tolerance', 'radiative_transfer_c2ray_tolerance'),
        ('c2ray_relaxation', 'radiative_transfer_c2ray_relaxation'),
        ('c2ray_nonconvergence', 'radiative_transfer_c2ray_nonconvergence'),
    ):
        if nested in radiation:
            values[legacy] = radiation[nested]
    for key in (
        'hydrogen_source_CFL', 'hydrogen_source_dtmin',
        'radiative_transfer',
    ):
        if key in runparams:
            values[key] = runparams[key]
    for key in (
        'hydrogen_chemistry', 'hydrogen_mass_fraction', 'hydrogen_xHI_initial',
        'hydrogen_xHI_inflow', 'hydrogen_xHI_outflow', 'hydrogen_update_mu',
        'hydrogen_alpha_B', 'hydrogen_beta',
    ):
        if key in chemistry:
            values[key] = chemistry[key]
    for key in (
        'hydrogen_chemistry', 'hydrogen_thermal_coupling',
        'hydrogen_recombination', 'hydrogen_collisional_ionization',
    ):
        if key in thermochemistry:
            values[key] = thermochemistry[key]
    for key in (
        'hydrogen_radiation_field', 'hydrogen_radiation_evolution',
        'hydrogen_ngamma_initial', 'hydrogen_sigma_gamma',
        'hydrogen_epsilon_gamma',
    ):
        if key in radiation:
            values[key] = radiation[key]
    return values


def build_problem(config, runparams=None):
    if runparams is None:
        runparams = config.get('par', config)
    config = {**config, **_runtime_values(runparams)}
    code_units_obj = CodeUnits.from_mapping(config.get('CodeUnits'))
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=_config_value(config, 'number_of_cells', 'grid_cells'),
        noghost=config.get('noghost', 2),
        boxsize=config['boxsize'],
        outdir=config.get('outdir', '.'),
        outfileprefix=config.get('outfileprefix', 'Output'),
        savedir=config.get('savedir', config.get('outdir', '.')),
        outputtimefilename=config.get('outputtimefilename', None),
        verbose=config.get('verbose', 0),
        timesim=_config_value(config, 'timesim', 'final_time'),
        area=config['area'],
        EOStype='isothermal',
        gamma=1.0,
        CFL=_config_value(config, 'CFL', 'hydro_cfl'),
        order=_config_value(config, 'order', 'hydro_order'),
        dtmin=_config_value(config, 'dtmin', 'hydro_timestep_min'),
        dtmax=_config_value(config, 'dtmax', 'hydro_timestep_max'),
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=_config_value(config, 'hydrogen_mass_fraction', default=1.0),
        hydrogen_xHI_initial=_config_value(config, 'hydrogen_xHI_initial', default=1.0),
        hydrogen_xHI_inflow=_config_value(config, 'hydrogen_xHI_inflow', default=1.0),
        hydrogen_xHI_outflow=_config_value(config, 'hydrogen_xHI_outflow', default=1.0),
        hydrogen_source_CFL=_config_value(config, 'hydrogen_source_CFL', 'source_cfl'),
        hydrogen_source_dtmin=_config_value(config, 'hydrogen_source_dtmin', 'source_timestep_min'),
        hydrogen_update_mu=_config_value(config, 'hydrogen_update_mu', default=True),
        hydrogen_thermal_coupling=_config_value(config, 'hydrogen_thermal_coupling', default=False),
        hydrogen_recombination=_config_value(config, 'hydrogen_recombination', default=True),
        hydrogen_collisional_ionization=_config_value(
            config,
            'hydrogen_collisional_ionization',
            default=False,
        ),
        hydrogen_alpha_B=_config_value(config, 'hydrogen_alpha_B', 'alpha_B_coefficient'),
        hydrogen_beta=_config_value(config, 'hydrogen_beta'),
        hydrogen_radiation_field=_config_value(config, 'hydrogen_radiation_field', default=False),
        hydrogen_radiation_evolution=_config_value(config, 'hydrogen_radiation_evolution', default=False),
        hydrogen_ngamma_initial=_config_value(config, 'hydrogen_ngamma_initial', default=0.0 / unyt.cm**3),
        hydrogen_sigma_gamma=_config_value(config, 'hydrogen_sigma_gamma', 'sigma_gamma'),
        hydrogen_epsilon_gamma=_config_value(config, 'hydrogen_epsilon_gamma', default=0.0 * unyt.erg),
        radiative_transfer=_config_value(config, 'radiative_transfer', default=True),
        radiative_transfer_method=_config_value(
            config,
            'radiative_transfer_method',
            default='long_characteristics',
        ),
        radiative_transfer_temporal_scheme=_config_value(
            config,
            'radiative_transfer_temporal_scheme',
            default='instantaneous',
        ),
        radiative_transfer_c2ray_max_iterations=_config_value(
            config,
            'radiative_transfer_c2ray_max_iterations',
            default=32,
        ),
        radiative_transfer_c2ray_tolerance=_config_value(
            config,
            'radiative_transfer_c2ray_tolerance',
            default=1.0e-6,
        ),
        radiative_transfer_c2ray_relaxation=_config_value(
            config,
            'radiative_transfer_c2ray_relaxation',
            default=1.0,
        ),
        radiative_transfer_c2ray_nonconvergence=_config_value(
            config,
            'radiative_transfer_c2ray_nonconvergence',
            default='warn',
        ),
        radiative_transfer_boundary_flux=_config_value(
            config,
            'radiative_transfer_boundary_flux',
        ),
        radiative_transfer_source_photon_rate=_config_value(
            config,
            'radiative_transfer_source_photon_rate',
            'source_photon_rate',
        ),
        radiative_transfer_direction=_config_value(config, 'radiative_transfer_direction', default=1),
        CodeUnits=code_units_obj,
        unit_system=code_units_obj.unit_system,
    )
    # Runtime code consumes explicit nested parameter groups.  Keep the flat
    # fields above as configuration metadata, but make the runtime groups
    # authoritative for the returned fixture.
    par.mesh = SimpleNamespace(ghost_cells=par.noghost, grid_cells=par.nogrid, area=par.area)
    par.simulation = SimpleNamespace(
        coordinate_system=par.coordsys,
        final_time=par.timesim,
        initial_condition_filename=_config_value(config, 'ICfilename'),
        current_time=0.0 * unyt.Myr,
        box_size=par.boxsize,
    )
    par.hydrodynamics = SimpleNamespace(
        eos_type=par.EOStype,
        gamma=par.gamma,
        CFL=par.CFL,
        order=par.order,
        riemann_solver=_config_value(config, 'riemann_solver', default='Rusanov'),
    )
    par.units = SimpleNamespace(CodeUnits=code_units_obj)
    par.boundary = SimpleNamespace(condition=par.boundcond)
    par.timestep = SimpleNamespace(dtmin=par.dtmin, dtmax=par.dtmax)
    par.output = SimpleNamespace(
        directory=par.outdir,
        filename_prefix=par.outfileprefix,
        cadence=None,
        time_list_filename=par.outputtimefilename,
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
    mesh.boundary = np.linspace(
        0.0,
        config['boxsize'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm

    fluid = Fluid()
    fluid.eos = EOS(par.EOStype, par.gamma, code_units_obj)
    fluid.rho = np.ones(par.nogrid) * config['rho_initial']
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * config['neutral_temperature']
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.SetFluidTime(0.0 * unyt.Myr)

    solver = Solver()
    return par, mesh, fluid, solver


def write_initial_condition(config, runparams):
    """Build the raw IC state and write it to ``ICfilename``."""
    par, mesh, fluid, _ = build_problem(config, runparams)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    icfilename = runparams.get(
        'ICfilename',
        runparams['simulation']['initial_condition_filename'],
    )
    Path(icfilename).unlink(missing_ok=True)
    rio.writehdf5(sim, icfilename)


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_problem(config, config.get('par', config))
    rio.readhdf5(par, mesh, fluid, outputfilename)
    code_units_obj = par.CodeUnits
    par.Time = np.asarray(par.Time, dtype=float) * code_units_obj.time_unit
    par.BoxSize = np.asarray(par.BoxSize, dtype=float) * code_units_obj.length_unit
    fluid.time = np.asarray(fluid.time, dtype=float) * code_units_obj.time_unit
    mesh.boundary = np.asarray(mesh.boundary, dtype=float) * code_units_obj.length_unit
    fluid.rho = np.asarray(fluid.rho, dtype=float) * code_units_obj.density_unit
    fluid.vel = np.asarray(fluid.vel, dtype=float) * code_units_obj.velocity_unit
    fluid.temp = np.asarray(fluid.temp, dtype=float) * code_units_obj.temperature_unit
    if hasattr(fluid, 'ngamma'):
        fluid.ngamma = np.asarray(fluid.ngamma, dtype=float) * code_units_obj.number_density_unit
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
    mesh = getattr(par, 'mesh', None)
    ghost_cells = getattr(mesh, 'ghost_cells', getattr(par, 'noghost', 0))
    grid_cells = getattr(mesh, 'grid_cells', getattr(par, 'nogrid', None))
    return slice(ghost_cells, ghost_cells + grid_cells)


def refresh_state(mesh, fluid, par, solver):
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid, verbose=getattr(par, 'verbose', 0))


def apply_piecewise_isothermal_state(mesh, fluid, par, solver, config):
    fluid.eos.apply_piecewise_isothermal_state(
        fluid,
        par,
        config['neutral_temperature'],
        config['ionized_temperature'],
    )
    refresh_state(mesh, fluid, par, solver)


def time_myr(value, code_units):
    myr_in_s = (1.0 * unyt.Myr).to_value(unyt.s)
    return float(code_quantity_to_cgs(value, code_units, 'time_s') / myr_in_s)


def print_startup_diagnostics(sim, config, icparams):
    """Print the main physical scales before the long run starts."""
    interior = interior_slice(sim.par)
    rho = np.asarray(sim.fluid.rho[interior], dtype=float)
    vel = np.asarray(sim.fluid.vel[interior], dtype=float)
    temp = np.asarray(sim.fluid.temp[interior], dtype=float)
    xHI = np.asarray(sim.fluid.xHI[interior], dtype=float)
    ngamma = np.asarray(sim.fluid.ngamma[interior], dtype=float) if hasattr(sim.fluid, 'ngamma') else None
    code_units_obj = sim.par.units.CodeUnits
    ngamma_cgs = None
    if ngamma is not None:
        ngamma_cgs = code_quantity_to_cgs(ngamma, code_units_obj, 'number_density_cm3')

    print('--- Startup diagnostics ---')
    print('cells = %d' % sim.par.mesh.grid_cells)
    print('time = %.6e Myr' % time_myr(sim.fluid.time, code_units_obj))
    print('rho range = [%.3e, %.3e] g/cm^3' % (np.min(rho), np.max(rho)))
    print('vel max abs = %.3e km/s' % (np.max(np.abs(vel)) / 1.0e5))
    print('temperature range = [%.3e, %.3e] K' % (np.min(temp), np.max(temp)))
    print('neutral fraction range = [%.3e, %.3e]' % (np.min(xHI), np.max(xHI)))
    if ngamma is not None:
        print('ngamma range = [%.3e, %.3e] code units' % (np.min(ngamma), np.max(ngamma)))
        if ngamma_cgs is not None:
            print('ngamma range = [%.3e, %.3e] cm^-3' % (np.min(ngamma_cgs), np.max(ngamma_cgs)))
            boundary_cm = code_quantity_to_cgs(
                sim.mesh.boundary[interior.start : interior.start + 2],
                code_units_obj,
                'length_cm',
            )
            inner_radius_cm = 0.5 * (boundary_cm[0] + boundary_cm[1])
            thin_estimate = config['source_photon_rate'].to_value(1 / unyt.s) / (
                4.0 * np.pi * inner_radius_cm**2 * unyt.c.to_value(unyt.cm / unyt.s)
            )
            print('optically thin inner-cell ngamma estimate = %.3e cm^-3' % thin_estimate)
    print('neutral sound speed = %.3e km/s' % neutral_sound_speed(config).to_value(unyt.km / unyt.s))
    print(
        'ionized sound speed (config) = %.3e km/s'
        % config['ionized_sound_speed'].to_value(unyt.km / unyt.s)
    )
    print('stromgren radius = %.3e pc' % stromgren_radius(config).to_value(unyt.pc))
    print('stagnation radius = %.3e pc' % stagnation_radius(config).to_value(unyt.pc))
    print(
        'Spitzer radius at final time = %.3e pc'
        % spitzer_radius(icparams['final_time'], config).to_value(unyt.pc)
    )
    print(
        'Hosokawa-Inutsuka radius at final time = %.3e pc'
        % hosokawa_inutsuka_radius(icparams['final_time'], config).to_value(unyt.pc)
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
            vel = np.asarray(sim.fluid.vel[interior], dtype=float)
            rho = np.asarray(sim.fluid.rho[interior], dtype=float)
            xHI = np.asarray(sim.fluid.xHI[interior], dtype=float)
            vmax = np.max(np.abs(vel)) / 1.0e5
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
                    np.min(rho),
                    np.max(rho),
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
    return {
        'time_Myr': _scalar_in_unit(fluid.time, unyt.Myr),
        'radius_pc': _value_in_unit(mesh.coordinate[interior], unyt.pc).copy(),
        'density_g_cm3': _value_in_unit(fluid.rho[interior], unyt.g / unyt.cm**3).copy(),
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
