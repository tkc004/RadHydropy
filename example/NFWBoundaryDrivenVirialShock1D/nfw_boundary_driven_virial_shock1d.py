"""Boundary-fed virial shock in a fixed NFW halo, with an HM12 PIE restart."""

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import h5py
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
    Simwrap, boundary_inflow_state, nfw_halo_parameters, pie_stability_diagnostics,
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
        left['vel_code'] = min(float(fluid.vel_code[first]), 0.0)
        right = {
            'rho_code': par.boundary.inflow_density,
            'vel_code': par.boundary.inflow_velocity,
            'pre_code': fluid.eos.pressure(
                par.boundary.inflow_density,
                par.boundary.inflow_temperature,
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
    if len(sim.mesh.boundary) == count + 1:
        return
    if len(sim.mesh.boundary) != total + 1:
        raise ValueError('restart snapshot has an unexpected mesh size')
    sim.mesh.boundary = sim.mesh.boundary[first:first + count + 1].copy()
    for name, value in vars(sim.fluid).items():
        if name in {'eos', 'time'}:
            continue
        array = np.asarray(value)
        if array.ndim and array.shape[-1] == total:
            setattr(sim.fluid, name, array[..., first:first + count].copy())


def _run_stage(par_config, halo, mode, restart=False):
    par_config = copy.deepcopy(par_config)
    outdir = Path(par_config['output']['directory'])
    outdir.mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(par_config['output'])
    sim = Rsim(par_config)
    sim.solver = BoundaryAccretionSolver()
    sim.Callreadhdf5()
    if restart:
        _strip_snapshot_ghosts(sim)
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        externalgravity=True,
        potential=nfw_potential(
            sim.mesh.coordinate, halo['scale_density'], halo['scale_radius'],
            code_units=sim.par.units.CodeUnits,
        ),
        coordinate=sim.mesh.coordinate.copy(),
        code_units=sim.par.units.CodeUnits,
    )
    sim.Run(mode=mode)
    return sorted(outdir.glob(
        f"{par_config['output']['filename_prefix']}_*.hdf5"
    ))


def _write_adiabatic_energy_audit(files, code_units, filename):
    """Write the open-boundary total-energy budget for an adiabatic stage."""
    if len(files) < 2:
        raise RuntimeError('energy audit requires at least two snapshots')

    def snapshot_energy(path):
        with h5py.File(path, 'r') as handle:
            header = handle['Header']
            data = handle['Data']
            first = int(header.attrs.get('GhostCells', 2))
            count = int(header.attrs['GridCells'])
            energy = np.asarray(data['Energy'][first:first + count], dtype=float)
            energy_unit = unyt.Unit(data['Energy'].attrs['units'])
            energy_scale = (1.0 * energy_unit).to_value(unyt.erg)
            total_energy = float(np.sum(energy) * energy_scale)
            time = (
                float(np.asarray(header['Time'][()]))
                * unyt.Unit(header['Time'].attrs['units'])
            ).to_value(unyt.Myr)
            boundary = float(header.attrs.get('CumulativeHydroBoundaryEnergyCode', 0.0))
            gravity = float(header.attrs.get('CumulativeGravityWorkCode', 0.0))
            return time, total_energy, boundary, gravity

    initial = snapshot_energy(files[0])
    final = snapshot_energy(files[-1])
    energy_scale = code_units.energy_unit.to_value(unyt.erg)
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
    if len(times) != expected_count:
        raise ValueError(
            f'{filename} contains {len(times)} times for {expected_count} snapshots'
        )
    return times + float(offset_myr)


def main(config_filename=DEFAULT_CONFIG, adiabatic_only=False):
    config_filename = Path(config_filename).resolve()
    config = eu.load_nested_example_config(config_filename)
    par_config = config['par']
    icparams = config['initial_condition']
    exampleparams = config['example']
    par_config['simulation']['initial_condition_filename'] = str(
        (config_filename.parent / par_config['simulation']['initial_condition_filename']).resolve()
    )
    par_config['output']['directory'] = str(
        (config_filename.parent / par_config['output']['directory']).resolve()
    )
    par_config['output']['savedir'] = str(
        (config_filename.parent / par_config['output']['savedir']).resolve()
    )
    par_config['output']['time_list_filename'] = str(
        (config_filename.parent / par_config['output']['time_list_filename']).resolve()
    )
    par_config['thermochemistry']['metal_pie_table_filename'] = str(
        (config_filename.parent / par_config['thermochemistry']['metal_pie_table_filename']).resolve()
    )
    code_units = CodeUnits.from_mapping(par_config['units']['CodeUnits'])
    pie_table = MetalPIETable(
        par_config['thermochemistry']['metal_pie_table_filename']
    )
    halo = nfw_halo_parameters(
        icparams['halo_mass'], icparams['concentration'], icparams['redshift'],
        icparams['overdensity'], icparams['h0'],
    )
    initial = Simwrap(icparams, par_config, code_units, pie_table)
    inflow = boundary_inflow_state(icparams, halo, pie_table, par_config)
    par_config['boundary'].update(inflow)
    initial_filename = par_config['simulation']['initial_condition_filename']
    Path(initial_filename).parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, initial_filename)

    adiabatic = copy.deepcopy(par_config)
    adiabatic['simulation']['final_time'] = exampleparams['adiabatic_final_time']
    adiabatic['thermochemistry']['network'] = 'hydrogen'
    adiabatic['thermochemistry']['metal_pie_enabled'] = False
    adiabatic_files = _run_stage(adiabatic, halo, 'hydro')
    if not adiabatic_files:
        raise RuntimeError('adiabatic stage produced no snapshots')
    adiabatic_audit = Path(adiabatic['output']['savedir']) / 'NFWBoundaryDrivenVirialShock1D_AdiabaticEnergyAudit.txt'
    _write_adiabatic_energy_audit(adiabatic_files, code_units, adiabatic_audit)
    if adiabatic_only:
        return

    pie = copy.deepcopy(par_config)
    pie['simulation']['name'] = par_config['simulation']['name'] + '_PIE'
    pie['simulation']['initial_condition_filename'] = str(adiabatic_files[-1])
    pie['simulation']['final_time'] = exampleparams['pie_final_time']
    pie['output']['directory'] = str(
        (config_filename.parent / exampleparams['pie_outdir']).resolve()
    )
    pie['output']['savedir'] = pie['output']['directory']
    pie['output']['time_list_filename'] = str(
        (config_filename.parent / exampleparams['pie_outputtimefilename']).resolve()
    )
    pie['thermochemistry']['network'] = 'pie_uvbg_cooling'
    pie['thermochemistry']['metal_pie_enabled'] = True
    pie_files = _run_stage(pie, halo, 'hydro_sources', restart=True)
    if not pie_files:
        raise RuntimeError('PIE stage produced no snapshots')

    savedir = Path(par_config['output']['savedir'])
    savedir.mkdir(parents=True, exist_ok=True)
    ad_report = savedir / 'NFWBoundaryDrivenVirialShock1D_AdiabaticShockHistory.txt'
    pie_report = savedir / 'NFWBoundaryDrivenVirialShock1D_PIEShockHistory.txt'
    figure = savedir / 'NFWBoundaryDrivenVirialShock1D.jpg'
    stability_report = savedir / 'NFWBoundaryDrivenVirialShock1D_PIEStability.txt'
    stability_figure = savedir / 'NFWBoundaryDrivenVirialShock1D_PIEStability.jpg'
    adiabatic_times = _scheduled_times_myr(
        adiabatic['output']['time_list_filename'], len(adiabatic_files)
    )
    pie_times = _scheduled_times_myr(
        pie['output']['time_list_filename'], len(pie_files),
        offset_myr=exampleparams['adiabatic_final_time'].to_value(unyt.Myr),
    )
    write_report(
        shock_history(adiabatic_files, halo, times_myr=adiabatic_times), ad_report
    )
    write_report(shock_history(pie_files, halo, times_myr=pie_times), pie_report)
    plot_comparison(
        adiabatic_files, pie_files, halo, figure,
        adiabatic_times_myr=adiabatic_times, pie_times_myr=pie_times,
    )
    stability = pie_stability_diagnostics(
        pie_files, pie_times, halo, pie_table, pie, icparams['mu']
    )
    write_stability_report(stability, stability_report)
    plot_stability_diagnostics(stability, stability_figure)

    print('halo mass = %.6g Msun' % halo['mass'].to_value(unyt.Msun))
    print('R200 = %.6g kpc' % halo['virial_radius'].to_value(unyt.kpc))
    print('Tvir = %.6g K' % virial_temperature(halo, icparams['mu']).to_value(unyt.K))
    print('outer PIE temperature = %.6g K' % inflow['inflow_temperature'].to_value(unyt.K))
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
