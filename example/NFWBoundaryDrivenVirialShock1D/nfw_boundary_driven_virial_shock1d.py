"""Boundary-fed virial shock in a fixed NFW halo, with an HM12 PIE restart."""

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = EXAMPLE_DIR.parent
PROJECT_ROOT = EXAMPLE_ROOT.parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'radhydropy-cache'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'radhydropy-matplotlib'))

import unyt
import numpy as np

import example_utils as eu
import radhydropy.io as rio
from radhydropy.gravity import Gravity, nfw_potential
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits, code_unit_scales
from radhydropy.thermo_networks.pie import MetalPIETable
from tools import (
    build_initial_condition, boundary_inflow_state, nfw_halo_parameters, pie_stability_diagnostics,
    plot_comparison, plot_stability_diagnostics, shock_history,
    virial_temperature, write_report, write_stability_report,
)


DEFAULT_CONFIG = EXAMPLE_DIR / 'nfw_boundary_driven_virial_shock1d.yaml'


class BoundaryAccretionSolver(Solver):
    """Use a non-injecting inner diode and maintained outer accretion."""

    def SetBoundary(self, mesh, fluid, par):
        first = par.mesh.ghost_cells
        right_start = first + par.mesh.grid_cells
        scales = code_unit_scales(par.units.CodeUnits)

        left = self._boundary_state(fluid, first)
        # Negative velocity points through the inner boundary and out of the
        # domain. Suppress only a positive velocity that would inject gas.
        left['vel_proper_code'] = min(float(fluid.vel_proper_code[first]), 0.0)
        right = {
            'rho_proper_code': par.boundary.rho_inflow_proper,
            'vel_proper_code': par.boundary.vel_inflow_proper,
            'pre_proper_code': fluid.eos.pressure(
                par.boundary.rho_inflow_proper,
                par.boundary.temperature_inflow_proper,
                par.boundary.inflow_mu,
            ),
        }
        if hasattr(fluid, 'xHI'):
            left['xHI'] = float(fluid.xHI[first])
            right['xHI'] = getattr(par.chemistry, 'hydrogen_xHI_inflow', 1.0)
        if hasattr(fluid, 'ngamma_code'):
            left['ngamma_code'] = fluid.ngamma_code[..., first]
            right['ngamma_code'] = self._to_code_number_density(
                getattr(par.radiation, 'hydrogen_ngamma_inflow', 0.0), scales
            )
        self._copy_boundary_state(fluid, slice(0, first), left)
        self._copy_boundary_state(
            fluid, slice(right_start, right_start + par.mesh.ghost_cells), right
        )


def _strip_snapshot_ghosts(sim):
    """Convert an evolved snapshot back to the ghost-free IC layout."""
    first = int(sim.par.mesh.ghost_cells)
    count = int(sim.par.mesh.grid_cells)
    total = count + 2 * first
    if len(sim.mesh.boundary_proper_code) == count + 1:
        return
    if len(sim.mesh.boundary_proper_code) != total + 1:
        raise ValueError('restart snapshot has an unexpected mesh size')
    sim.mesh.boundary_proper_code = sim.mesh.boundary_proper_code[first:first + count + 1].copy()
    for name, value in vars(sim.fluid).items():
        if name in {'eos', 'time'}:
            continue
        array = np.asarray(value)
        if array.ndim and array.shape[-1] == total:
            setattr(sim.fluid, name, array[..., first:first + count].copy())


def _run_stage(config, halo, mode, restart=False):
    stage_config = copy.deepcopy(config)
    outdir = Path(stage_config['par']['output']['directory'])
    outdir.mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(stage_config)
    sim = Rsim(stage_config['par'])
    sim.solver = BoundaryAccretionSolver()
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    if restart:
        # A restart snapshot carries the previous stage's output settings.
        # Restore the current stage's destinations and schedule after reading
        # the snapshot so PIE outputs are selected and written for this stage.
        sim.par.output.directory = stage_config['par']['output']['directory']
        sim.par.output.filename_prefix = stage_config['par']['output']['filename_prefix']
        sim.par.output.time_list_filename = stage_config['par']['output']['time_list_filename']
    if restart:
        _strip_snapshot_ghosts(sim)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.geometry_state.x_proper_code, halo['rho_scale_cgs_g_cm3_unyt'], halo['radius_scale_proper_kpc_unyt'],
            code_units=sim.par.units.CodeUnits,
        ),
        coordinate=sim.mesh.geometry_state.x_proper_code.copy(),
        code_units=sim.par.units.CodeUnits,
    )
    sim.Run(mode=mode)
    return sorted(outdir.glob(
        f"{config['par']['output']['filename_prefix']}_*.hdf5"
    ))


def _write_adiabatic_energy_audit(files, config, filename):
    """Write the open-boundary total-energy budget for an adiabatic stage."""
    if len(files) < 2:
        raise RuntimeError('energy audit requires at least two snapshots')

    def snapshot_energy(path):
        snapshot = Rsim(config['par'])
        rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(path))
        first = int(snapshot.par.mesh.ghost_cells)
        last = first + int(snapshot.par.mesh.grid_cells)
        code_unit_system = snapshot.par.units.CodeUnits
        energy_cgs_erg = (
            np.asarray(snapshot.fluid.Energy_code[first:last], dtype=float)
            * float(code_unit_system.energy_unit.to_value(unyt.erg))
        )
        time_proper_cgs_s = (
            float(np.asarray(snapshot.fluid.time_proper_code).flat[0])
            * float(code_unit_system.time_unit.to_value(unyt.s))
        )
        boundary = float(getattr(snapshot.par, 'CumulativeHydroBoundaryEnergyCode', 0.0))
        gravity = float(getattr(snapshot.par, 'CumulativeGravityWorkCode', 0.0))
        return time_proper_cgs_s / float((1.0 * unyt.Myr).to_value(unyt.s)), float(np.sum(energy_cgs_erg)), boundary, gravity

    initial = snapshot_energy(files[0])
    final = snapshot_energy(files[-1])
    code_unit_system = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    energy_scale = code_unit_system.energy_unit.to_value(unyt.erg)
    delta_energy = final[1] - initial[1]
    boundary_work = (final[2] - initial[2]) * energy_scale
    gravity_work = (final[3] - initial[3]) * energy_scale
    residual = delta_energy - boundary_work - gravity_work
    with Path(filename).open('w', encoding='utf-8') as stream:
        stream.write('quantity value_cgs_erg\n')
        stream.write(f'initial_gas_energy {initial[1]:.12e}\n')
        stream.write(f'final_gas_energy {final[1]:.12e}\n')
        stream.write(f'delta_gas_energy {delta_energy:.12e}\n')
        stream.write(f'boundary_energy {boundary_work:.12e}\n')
        stream.write(f'gravity_work {gravity_work:.12e}\n')
        stream.write(f'budget_residual {residual:.12e}\n')
        stream.write(f'residual_fraction_of_delta {residual / max(abs(delta_energy), 1.0e-99):.12e}\n')
    print('adiabatic energy audit = %s' % filename)
    print('adiabatic energy residual = %.6e erg (%.6e of delta)' % (
        residual, residual / max(abs(delta_energy), 1.0e-99)
    ))


def _scheduled_times_myr(filename, expected_count, offset_myr=0.0):
    times = rio.load_output_time_list(filename).to_value(unyt.Myr)
    if offset_myr:
        # Restart schedules are absolute times.  The restart itself is also
        # retained as the first diagnostic snapshot.
        times = times[times >= float(offset_myr)]
        if not len(times) or times[0] > float(offset_myr):
            times = np.insert(times, 0, float(offset_myr))
    if len(times) != expected_count:
        raise ValueError(
            f'{filename} contains {len(times)} times for {expected_count} snapshots'
        )
    return times


def main(config_filename=DEFAULT_CONFIG, adiabatic_only=False):
    config_filename = Path(config_filename).resolve()
    config = eu.load_nested_example_config(config_filename)

    initial_condition = config['initial_condition']
    exampleparams = config['example']
    config["par"]['simulation']['initial_condition_filename'] = str(
        (config_filename.parent / config["par"]['simulation']['initial_condition_filename']).resolve()
    )
    config["par"]['output']['directory'] = str(
        (config_filename.parent / config["par"]['output']['directory']).resolve()
    )
    config["par"]['output']['directory'] = str(
        (config_filename.parent / config["par"]['output']['directory']).resolve()
    )
    config["par"]['output']['time_list_filename'] = str(
        (config_filename.parent / config["par"]['output']['time_list_filename']).resolve()
    )
    config["par"]['thermochemistry']['metal_pie_table_filename'] = str(
        (config_filename.parent / config["par"]['thermochemistry']['metal_pie_table_filename']).resolve()
    )
    code_units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    pie_table = MetalPIETable(
        config["par"]['thermochemistry']['metal_pie_table_filename']
    )
    halo = nfw_halo_parameters(
        initial_condition['halo_mass'], initial_condition['concentration'], initial_condition['redshift'],
        initial_condition['overdensity'], initial_condition['h0'],
    )
    nested_config = dict(config)
    nested_config['_code_units'] = code_units
    nested_config['_pie_table'] = pie_table
    initial = build_initial_condition(nested_config)
    inflow = boundary_inflow_state(nested_config, halo, pie_table)
    config["par"]['boundary'].update(inflow)
    initial_filename = config["par"]['simulation']['initial_condition_filename']
    Path(initial_filename).parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, initial_filename)

    adiabatic_config = copy.deepcopy(config)
    adiabatic_config['par']['simulation']['final_time'] = exampleparams['adiabatic_final_time']
    adiabatic_config['par']['thermochemistry']['network'] = 'hydrogen'
    adiabatic_config['par']['thermochemistry']['metal_pie_enabled'] = False
    adiabatic_files = _run_stage(adiabatic_config, halo, 'hydro')
    if not adiabatic_files:
        raise RuntimeError('adiabatic stage produced no snapshots')
    adiabatic_audit = Path(adiabatic_config['par']['output']['directory']) / 'NFWBoundaryDrivenVirialShock1D_AdiabaticEnergyAudit.txt'
    _write_adiabatic_energy_audit(adiabatic_files, adiabatic_config, adiabatic_audit)
    if adiabatic_only:
        return

    pie_config = copy.deepcopy(config)
    pie_config['par']['simulation']['name'] = config["par"]['simulation']['name'] + '_PIE'
    pie_config['par']['simulation']['initial_condition_filename'] = str(adiabatic_files[-1])
    pie_config['par']['simulation']['final_time'] = exampleparams['pie_final_time']
    pie_config['par']['output']['directory'] = str(
        (config_filename.parent / exampleparams['pie_outdir']).resolve()
    )
    pie_config['par']['output']['time_list_filename'] = str(
        (config_filename.parent / exampleparams['pie_outputtimefilename']).resolve()
    )
    pie_config['par']['thermochemistry']['network'] = 'pie_uvbg_cooling'
    pie_config['par']['thermochemistry']['metal_pie_enabled'] = True
    pie_files = _run_stage(pie_config, halo, 'hydro_sources', restart=True)
    if not pie_files:
        raise RuntimeError('PIE stage produced no snapshots')

    directory = Path(config["par"]['output']['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    ad_report = directory / 'NFWBoundaryDrivenVirialShock1D_AdiabaticShockHistory.txt'
    pie_report = directory / 'NFWBoundaryDrivenVirialShock1D_PIEShockHistory.txt'
    figure = directory / 'NFWBoundaryDrivenVirialShock1D.jpg'
    stability_report = directory / 'NFWBoundaryDrivenVirialShock1D_PIEStability.txt'
    stability_figure = directory / 'NFWBoundaryDrivenVirialShock1D_PIEStability.jpg'
    adiabatic_times = _scheduled_times_myr(
        adiabatic['output']['time_list_filename'], len(adiabatic_files)
    )
    pie_times = _scheduled_times_myr(
        pie['output']['time_list_filename'], len(pie_files),
        offset_myr=exampleparams['adiabatic_final_time'].to_value(unyt.Myr),
    )
    write_report(
        shock_history(
            adiabatic_files, halo, adiabatic_config, times_myr=adiabatic_times
        ), ad_report
    )
    write_report(
        shock_history(pie_files, halo, pie_config, times_myr=pie_times), pie_report
    )
    plot_comparison(
        adiabatic_files, pie_files, halo, figure,
        pie_config,
        adiabatic_times_myr=adiabatic_times, pie_times_myr=pie_times,
    )
    stability = pie_stability_diagnostics(
        pie_files, pie_times, halo, pie_table, pie_config, initial_condition['mu']
    )
    write_stability_report(stability, stability_report)
    plot_stability_diagnostics(stability, stability_figure)

    print('halo mass = %.6g Msun' % halo['mass_halo_proper_g_unyt'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % halo['radius_virial_proper_kpc_unyt'].to_value(unyt.kpc))
    print('Tvir = %.6g K' % virial_temperature(halo, initial_condition['mu']).to_value(unyt.K))
    print('outer PIE temperature = %.6g K' % inflow['temperature_inflow_proper'].to_value(unyt.K))
    print('adiabatic snapshots = %d; PIE snapshots = %d' % (
        len(adiabatic_files), len(pie_files)))
    print('figure = %s' % figure)
    print('PIE diagnostics = %s' % stability_figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--adiabatic-only', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config, adiabatic_only=args.adiabatic_only)
