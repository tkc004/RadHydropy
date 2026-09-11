"""Hydro plus centrifugal-source expansion benchmark."""

import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/radhydropy-matplotlib')
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'example'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
from radhydropy.initial_condition_writer import InitialConditionWriter
import example_utils as eu
from shell_remap import centrifugal_shell_reference


CONFIG = ROOT / 'gas_centrifugal_hydro_expansion1d.yaml'

def spherical_centers(boundary_proper_code):
    return 0.75 * (
        boundary_proper_code[1:]**4 - boundary_proper_code[:-1]**4
    ) / (boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3)


def build_initial_condition(config):
    """Build the proper-coordinate IC through the shared writer boundary."""
    initial_condition = config['initial_condition']
    units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    count = int(config['par']['mesh']['grid_cells'])
    boundary_proper_unyt = np.linspace(0.0, 1.0, count + 1)
    boundary_proper_unyt = (
        initial_condition['radius_inner_proper']
        + boundary_proper_unyt * (
            initial_condition['radius_outer_proper']
            - initial_condition['radius_inner_proper']
        )
    )
    boundary_proper_code = quantity_to_value(boundary_proper_unyt, units.length_unit)
    radius_proper_code = spherical_centers(boundary_proper_code)
    writer = InitialConditionWriter(par_config=config['par'], code_units=units)
    writer.box_size = writer.radquantity(initial_condition['radius_outer_proper'])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(radius_proper_code * units.length_unit)
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(count) * initial_condition['rho_proper']
    )
    writer.fluid.vel_radarray = writer.radarray(np.zeros(count) * units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(count) * initial_condition['temperature_proper']
    )
    writer.simulation.fluid.mu = np.ones(count)
    writer.simulation.fluid.specific_angular_momentum_radarray = writer.radarray(
        float(initial_condition['rotation_factor'])
        * np.sqrt(
            quantity_to_value(initial_condition['central_mass_proper'], units.mass_unit)
            * radius_proper_code
        ) * units.length_unit.units * units.velocity_unit.units,
        field_name='specific_angular_momentum_code',
    )
    return writer


class FixedCentralGravity:
    cosmological = False
    dark_matter = None

    def __init__(self, central_mass):
        self.central_mass = central_mass

    def acceleration_on_mesh(self, mesh, par=None, **kwargs):
        radius_proper_code = np.abs(np.asarray(mesh.x_proper_code, dtype=float))
        acceleration = np.zeros_like(radius_proper_code)
        valid = radius_proper_code > 0.0
        acceleration[valid] = -self.central_mass / radius_proper_code[valid]**2
        return acceleration

    def potential_on(self, x_proper_code):
        radius_proper_code = np.abs(np.asarray(x_proper_code, dtype=float))
        potential_proper_code = np.zeros_like(radius_proper_code)
        valid = radius_proper_code > 0.0
        potential_proper_code[valid] = -self.central_mass / radius_proper_code[valid]
        return potential_proper_code

    def potential_on_mesh(self, mesh):
        return self.potential_on(mesh.x_proper_code)


def run_simulation(config):
    par = config['par']
    initial_condition = config['initial_condition']
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    count = int(par['mesh']['grid_cells'])
    initial = build_initial_condition(config)
    specific_angular_momentum_proper_code = np.asarray(
        initial.simulation.fluid.specific_angular_momentum_radarray, dtype=float
    ).copy()
    filename = ROOT / par['simulation']['initial_condition_filename']
    filename.parent.mkdir(parents=True, exist_ok=True)
    initial.write(filename)
    sim = rio.loadhdf5(config, str(filename))
    if hasattr(sim.fluid, 'specific_angular_momentum_code'):
        del sim.fluid.specific_angular_momentum_code
    if hasattr(sim.fluid, 'AngularMomentum_code'):
        del sim.fluid.AngularMomentum_code
    central_mass = quantity_to_value(initial_condition['central_mass_proper'], units.mass_unit)
    sim.par.gravity = FixedCentralGravity(central_mass)
    sim.SetMesh()
    sim.SetFluid()
    ghost_cells = int(sim.par.mesh.ghost_cells)
    sim.fluid.specific_angular_momentum_code = as_named_array(np.concatenate((
        np.zeros(ghost_cells), specific_angular_momentum_proper_code,
        np.zeros(ghost_cells),
    )))
    if hasattr(sim.fluid, 'AngularMomentum_code'):
        del sim.fluid.AngularMomentum_code
    sim.SetInitFluid()
    sim.par.gravity = FixedCentralGravity(central_mass)
    active = slice(int(sim.par.mesh.ghost_cells), int(sim.par.mesh.ghost_cells) + int(sim.par.mesh.grid_cells))
    initial_mass = np.asarray(sim.fluid.Mass_code[active], dtype=float).copy()
    initial_energy = np.asarray(sim.fluid.Energy_code[active], dtype=float).copy()
    initial_radius = np.asarray(sim.mesh.x_proper_code[active], dtype=float).copy()
    sim.Run(outputtime=0, mode='hydro')
    final_filename = sorted(
        (ROOT / par['output']['directory']).glob('Output_[0-9][0-9][0-9].hdf5')
    )[-1]
    final_sim = rio.loadhdf5(config, final_filename)
    return (
        sim, final_sim.mesh, final_sim.fluid, initial_mass, initial_energy,
        initial_radius, float(sim.cumulative_gravity_work),
        float(sim.cumulative_gravity_potential_change),
        float(sim.cumulative_gravity_potential_flux),
    )


def main(config_filename=CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    initial_condition = config['initial_condition']
    units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    (sim, saved_mesh, saved, initial_mass, initial_energy,
     initial_radius, cumulative_gravity_work, cumulative_potential_change,
     cumulative_potential_flux) = run_simulation(config)
    active = slice(int(sim.par.mesh.ghost_cells), int(sim.par.mesh.ghost_cells) + int(sim.par.mesh.grid_cells))
    central_mass = quantity_to_value(initial_condition['central_mass_proper'], units.mass_unit)
    rotation_factor = float(initial_condition['rotation_factor'])
    final_time = float(sim.fluid.time_proper_code)
    saved_boundary = np.asarray(saved_mesh.boundary_proper_code, dtype=float)
    first = int(sim.par.mesh.ghost_cells)
    source_boundary = saved_boundary[first:first + int(sim.par.mesh.grid_cells) + 1]
    saved_radius = spherical_centers(saved_boundary)[active]
    reference = centrifugal_shell_reference(
        source_boundary,
        source_boundary,
        final_time,
        quantity_to_value(initial_condition['rho_proper'], units.density_unit),
        central_mass,
        rotation_factor,
        samples_per_cell=int(initial_condition.get('reference_samples_per_cell', 32)),
    )
    vel_proper_code_reference = reference['vel_proper_code']
    ode_j = reference['specific_angular_momentum_proper_code']
    saved_velocity = np.asarray(saved.vel_proper_code[active], dtype=float)
    saved_j = np.asarray(saved.specific_angular_momentum_code[active], dtype=float)
    saved_mass = np.asarray(saved.Mass_code[active], dtype=float)
    saved_energy = np.asarray(saved.Energy_code[active], dtype=float)
    velocity_error = float(np.max(np.abs(saved_velocity - vel_proper_code_reference)))
    j_error = float(np.max(np.abs(saved_j - ode_j)))
    mass_error = float(
        abs(np.sum(saved_mass) - np.sum(initial_mass))
        / max(abs(np.sum(initial_mass)), 1.0e-300)
    )
    if velocity_error > 0.08:
        raise RuntimeError('hydro expansion velocity disagrees with shell ODE: %.6g' % velocity_error)
    if not np.all(np.isfinite(saved_j)):
        raise RuntimeError('hydro expansion produced invalid specific angular momentum')
    if mass_error > 1.0e-10:
        raise RuntimeError('closed hydro expansion lost mass: relative error %.6g' % mass_error)

    # Cell-centered potential energy uses the extensive cell mass, which
    # already contains the spherical cell volume.  This is the discrete
    # Eulerian counterpart of the shell potential energy.
    potential_initial = -central_mass * np.sum(initial_mass / initial_radius)
    potential_final = -central_mass * np.sum(saved_mass / saved_radius)
    potential_change = potential_final - potential_initial
    potential_work_residual = cumulative_gravity_work + potential_change

    # The central-gravity potential closes the gas-energy audit.  The shell
    # ODE conserves this quantity even though gas kinetic and rotational energy
    # separately exchange during the expansion.
    saved_total_energy = (
        np.sum(saved_energy) - np.sum(central_mass * saved_mass / saved_radius)
    )
    ode_total_energy = np.sum(reference['energy_proper_code'])
    energy_error = float(abs(saved_total_energy - ode_total_energy))
    energy_scale = max(abs(float(ode_total_energy)), 1.0e-12)
    if energy_error / energy_scale > 2.0e-3:
        raise RuntimeError(
            'hydro expansion total-energy audit failed: relative error %.6g'
            % (energy_error / energy_scale)
        )

    rho_proper_code = np.asarray(saved.rho_proper_code[active], dtype=float)
    temp_proper_code = np.asarray(saved.temp_proper_code[active], dtype=float)
    mu = np.asarray(saved.mu[active], dtype=float)
    pre_proper_code = np.asarray(sim.fluid.eos.pressure(rho_proper_code, temp_proper_code, mu), dtype=float)
    pressure_ratio = np.divide(
        pre_proper_code / np.maximum(rho_proper_code, np.finfo(float).tiny),
        central_mass / np.maximum(saved_radius, np.finfo(float).tiny),
    )
    figure = ROOT / par['output']['directory'] / 'GasCentrifugalHydroExpansion1D.jpg'
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flat
    axes[0].plot(saved_radius, saved_velocity, ':o', markersize=4, label='Rsim')
    axes[0].plot(saved_radius, vel_proper_code_reference, '--', label='pressureless shell ODE')
    axes[0].set_ylabel('radial velocity')
    axes[1].plot(saved_radius, rho_proper_code, ':o', markersize=4, label='Rsim density')
    axes[1].set_ylabel('density')
    axes[2].plot(saved_radius, saved_j, ':o', markersize=4, label='Rsim $J/M$')
    axes[2].plot(saved_radius, ode_j, '--', label='advected shell $j$')
    axes[2].set_ylabel('specific angular momentum')
    axes[3].semilogy(
        saved_radius, np.maximum(pressure_ratio, np.finfo(float).tiny),
        ':o', markersize=4, label='Rsim pressure support',
    )
    axes[3].axhline(1.0, color='k', linestyle=':')
    axes[3].set_ylabel('thermal / dynamical scale')
    axes[4].bar(
        ['ODE', 'Rsim'], [ode_total_energy, saved_total_energy],
        color=['C0', 'C1'],
    )
    axes[4].set_ylabel('gas + gravitational energy')
    axes[4].set_title('energy audit')
    axes[5].axis('off')
    for axis in axes[:4]:
        axis.set_xlabel('radius')
        axis.grid(alpha=0.25)
        axis.legend()
    axes[4].grid(axis='y', alpha=0.25)
    fig.suptitle('Centrifugal hydro expansion benchmark')
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print('centrifugal hydro expansion check passed')
    print('maximum velocity error = %.6g' % velocity_error)
    print('maximum mapped J/M error = %.6g' % j_error)
    print('relative global mass error = %.6g' % mass_error)
    print('cumulative gravity work = %.6g' % cumulative_gravity_work)
    print('cell-centered potential change = %.6g' % potential_change)
    print('reported potential change = %.6g' % cumulative_potential_change)
    print('face potential-energy boundary_proper_code flux = %.6g' % cumulative_potential_flux)
    print('gravity/potential closure residual = %.6g' % potential_work_residual)
    print('total energy audit error = %.6g' % energy_error)
    print('maximum thermal/dynamical scale = %.6g' % np.max(pressure_ratio))
    print('figure = %s' % figure)


if __name__ == '__main__':
    main()
