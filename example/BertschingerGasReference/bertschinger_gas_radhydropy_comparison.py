"""Run RadHydropy and compare it with the standalone Bertschinger solution."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unyt

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.gravity import Gravity
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    SUPERCOMOVING_RUNTIME_FIELDS,
)
import example_utils as eu
from bertschinger_gas import solve_bertschinger_gas


DEFAULT_CONFIG = Path(__file__).with_name('bertschinger_gas_radhydropy.yaml')


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class BertschingerBoundarySolver(Solver):
    """Use an inner outflow and the exact cold growing mode outside.

    The inner edge is an excision boundary (the first active cell is not the
    origin), so its primitive state is copied into the ghosts.  At the outer
    edge the ghost state is prescribed in supercomoving variables.  In
    particular, the velocity is the peculiar velocity, not the Hubble-flow
    velocity, and the pressure is recomputed from the boundary density and
    temperature rather than taken from the YAML outflow values.
    """

    def SetBoundary(self, mesh, fluid, par):
        super().SetBoundary(mesh, fluid, par)
        first = int(par.mesh.ghost_cells)
        last = first + int(par.mesh.grid_cells)
        left = slice(0, first)
        right = slice(last, last + int(par.mesh.ghost_cells))
        for name in ('rho_comoving_code', 'vel_supercomoving_code', 'pre_supercomoving_code', 'temp_supercomoving_code', 'mu'):
            if hasattr(fluid, name):
                getattr(fluid, name)[left] = getattr(fluid, name)[first]

        tau = float(np.asarray(fluid.tau_supercomoving_code, dtype=float).reshape(-1)[0])
        cosmology = par.cosmology
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
        scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
        hubble = float(cosmology.hubble_from_supercomoving(tau))
        radius = np.asarray(mesh.x_comoving_code[right], dtype=float)
        amplitude = float(par.perturbation_amplitude)
        delta = amplitude / np.maximum(radius, 1.0e-30)**3
        fluid.rho_comoving_code[right] = (
            float(cosmology.background_density(cosmic_time)) * scale_factor**3
        )
        fluid.vel_supercomoving_code[right] = -scale_factor**2 * hubble * delta * radius / 3.0
        # The Bertschinger exterior is pressureless.  Do not use the finite
        # cold-temperature floor from the active IC in the outer ghosts.
        fluid.temp_supercomoving_code[right] = 0.0
        fluid.mu[right] = float(par.mu_outflow)
        fluid.pre_supercomoving_code[right] = 0.0


def _spherical_centers(boundary_comoving_code):
    return 0.75 * (boundary_comoving_code[1:]**4 - boundary_comoving_code[:-1]**4) / (
        boundary_comoving_code[1:]**3 - boundary_comoving_code[:-1]**3
    )


def _interpolate_profile(solution, radius):
    """Interpolate the piecewise standalone profile at dimensionless radius."""
    combined_lambda = np.concatenate((solution.lambda_in, solution.lambda_out))
    order = np.argsort(combined_lambda)
    combined_lambda = combined_lambda[order]
    combined_density = np.concatenate((solution.density_in, solution.density_out))[order]
    combined_velocity = np.concatenate((solution.velocity_in, solution.velocity_out))[order]
    density = np.interp(
        radius,
        combined_lambda,
        combined_density,
    )
    velocity = np.interp(
        radius,
        combined_lambda,
        combined_velocity,
    )
    pressure = np.zeros_like(radius)
    interior = radius <= solution.shock_lambda
    pressure[interior] = np.interp(
        radius[interior], solution.lambda_in, solution.pressure_in
    )
    # The cold exterior has P=0.  Mask it instead of drawing an arbitrary
    # positive floor on a logarithmic comparison plot.
    pressure[~interior] = np.nan
    return density, velocity, pressure


def build_initial_condition(config):
    sim = SimpleNamespace()
    initial_condition = config['initial_condition']

    solution = config['_reference_solution']
    simulation = config["par"]['simulation']
    mesh = config["par"]['mesh']
    code_units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    sim.par.hydrodynamics = SimpleNamespace(gamma=5.0 / 3.0)
    grid_cells = int(mesh['grid_cells'])
    initial_time = float(initial_condition['initial_cosmic_time'])
    sim.par.cosmological_expansion = True
    sim.par.supercomoving_coordinates = True
    sim.par.cosmological_gravity = True
    sim.par.cosmology_type = 'einstein_de_sitter'
    sim.par.cosmology_t_ref = initial_time
    sim.par.cosmology_a_ref = 1.0
    sim.par.cosmology = EinsteinDeSitter.from_code_units(
        code_units, t_ref=initial_time, a_ref=1.0
    )
    sim.par.tau_supercomoving_code = np.ones(1) * sim.par.cosmology.supercomoving_time(initial_time)
    sim.par.coordinate_frame = 'comoving'
    sim.par.time_coordinate = 'supercomoving'
    sim.par.velocity_representation = 'supercomoving_peculiar'
    sim.par.density_representation = 'comoving'
    sim.par.pressure_representation = 'supercomoving'
    sim.par.temperature_representation = 'supercomoving'
    sim.par.perturbation_amplitude = float(initial_condition['perturbation_amplitude'])
    sim.par.simulation = SimpleNamespace(
        # The config["par"] and HDF5 state use supercomoving time.  The IC profile
        # itself is evaluated at this cosmic time, but the serialized state
        # must start at the corresponding tau (zero at t_ref here).
        tau_supercomoving_code=sim.par.tau_supercomoving_code.copy(),
        box_size=initial_condition['box_size'],
        coordinate_system=simulation['coordinate_system'],
    )
    sim.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
    inner_radius_code = float(
        initial_condition['inner_radius'].to_value(code_units.length_unit)
    )
    outer_radius_code = float(
        initial_condition['outer_radius'].to_value(code_units.length_unit)
    )
    sim.mesh.boundary_comoving_code = np.linspace(
        inner_radius_code, outer_radius_code, grid_cells + 1
    )
    sim.mesh.x_comoving_code = _spherical_centers(sim.mesh.boundary_comoving_code)
    sim.mesh.area_comoving_code = 4.0 * np.pi * sim.mesh.boundary_comoving_code[:-1]**2
    sim.mesh.volume_comoving_code = 4.0 * np.pi / 3.0 * (
        sim.mesh.boundary_comoving_code[1:]**3 - sim.mesh.boundary_comoving_code[:-1]**3
    )

    radius = np.asarray(sim.mesh.x_comoving_code, dtype=float)
    cosmology = sim.par.cosmology
    scale_factor = float(cosmology.scale_factor(initial_time))
    rho_background = float(cosmology.background_density(initial_time)) * scale_factor**3
    amplitude = float(initial_condition['perturbation_amplitude'])
    rta = scale_factor * (
        amplitude / solution.mass_out[np.argmin(
            np.abs(solution.lambda_out - 1.0)
        )]
    ) ** (1.0 / 3.0)
    similarity_radius = scale_factor * radius / rta
    combined_lambda = np.concatenate((solution.lambda_in, solution.lambda_out))
    order = np.argsort(combined_lambda)
    combined_lambda = combined_lambda[order]
    density_profile = np.concatenate(
        (solution.density_in, solution.density_out)
    )[order]
    velocity_profile = np.concatenate(
        (solution.velocity_in, solution.velocity_out)
    )[order]
    density = rho_background * np.interp(
        similarity_radius, combined_lambda, density_profile
    )
    velocity_physical = np.interp(
        similarity_radius, combined_lambda, velocity_profile
    ) * rta / initial_time
    hubble_initial = float(cosmology.hubble(initial_time))
    velocity = scale_factor * (
        velocity_physical - hubble_initial * scale_factor * radius
    )
    pressure_profile = np.zeros_like(similarity_radius)
    interior = similarity_radius <= solution.shock_lambda
    pressure_profile[interior] = np.interp(
        similarity_radius[interior], solution.lambda_in, solution.pressure_in
    )
    pressure = pressure_profile * rho_background * (rta / initial_time) ** 2
    mean_molecular_weight = float(initial_condition['mean_molecular_weight'])
    temperature_code = pressure * mean_molecular_weight / (
        np.maximum(density, 1.0e-300)
        * code_units.boltzmann_code / code_units.proton_mass_code
    )
    temperature_code = np.maximum(temperature_code, 0.0)
    sim.par.initial_temperature_code = float(np.max(temperature_code))
    sim.par.mu_outflow = float(initial_condition['mean_molecular_weight'])
    sim.fluid.rho_comoving_code = density
    sim.fluid.vel_supercomoving_code = velocity
    sim.fluid.temp_supercomoving_code = temperature_code
    sim.fluid.mu = np.ones(grid_cells) * float(initial_condition['mean_molecular_weight'])
    sim.mesh.geometry_state = MeshGeometryState(
        x_comoving_code=sim.mesh.x_comoving_code,
        boundary_comoving_code=sim.mesh.boundary_comoving_code,
        width_comoving_code=np.diff(sim.mesh.boundary_comoving_code),
        area_comoving_code=sim.mesh.area_comoving_code,
        volume_comoving_code=sim.mesh.volume_comoving_code,
    )
    sim.fluid.tau_supercomoving_code = float(
        np.asarray(sim.par.tau_supercomoving_code, dtype=float).reshape(-1)[0]
    )
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        rho_comoving_code=sim.fluid.rho_comoving_code,
        vel_supercomoving_code=sim.fluid.vel_supercomoving_code,
        pre_supercomoving_code=np.zeros_like(sim.fluid.rho_comoving_code),
        temp_supercomoving_code=sim.fluid.temp_supercomoving_code,
        tau_supercomoving_code=sim.fluid.tau_supercomoving_code,
        mu_dimensionless=sim.fluid.mu,
    )

    sim.dark_matter = DarkMatterShells(
        radius=np.array([float(initial_condition['outer_radius'].to_value(unyt.kpc)) * 2.0]),
        velocity=np.zeros(1),
        mass=np.full(1, 1.0e-30) * code_units.mass_unit,
        code_units=code_units,
    )


    return sim
def _similarity_profiles(sim, solution):
    interior = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    tau = float(np.asarray(sim.fluid.tau_supercomoving_code, dtype=float).reshape(-1)[0])
    cosmology = sim.par.cosmology
    scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
    cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
    radius = scale_factor * np.asarray(sim.mesh.x_comoving_code[interior], dtype=float)
    density = cosmology.physical_density(
        np.asarray(sim.fluid.rho_comoving_code[interior], dtype=float), tau
    )
    velocity = cosmology.physical_velocity(
        np.asarray(sim.mesh.x_comoving_code[interior], dtype=float),
        np.asarray(sim.fluid.vel_supercomoving_code[interior], dtype=float), tau,
    )
    pressure = cosmology.physical_pressure(
        np.asarray(sim.fluid.pre_supercomoving_code[interior], dtype=float), tau, sim.par.hydrodynamics.gamma
    )
    boundaries = np.asarray(
        sim.mesh.boundary_comoving_code[sim.par.mesh.ghost_cells:sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells + 1],
        dtype=float,
    ) * scale_factor
    time = cosmic_time
    # The standalone normalization uses M_excess = M_ta at lambda=1.
    # For the scale-free IC, M_excess=(4*pi/3) rho_b(t_i) A, hence
    # r_ta(t_i)=(A/TURNAROUND_MASS)^(1/3), followed by r_ta~t^(8/9).
    initial_time = float(sim.par.cosmology.t_ref)
    initial_scale = float(sim.par.cosmology.a_ref)
    amplitude = float(sim.par.perturbation_amplitude)
    rta_initial = initial_scale * (
        amplitude / solution.mass_out[np.argmin(np.abs(solution.lambda_out - 1.0))]
    ) ** (1.0 / 3.0)
    rta = rta_initial * (time / initial_time) ** (8.0 / 9.0)
    rho_background = cosmology.background_density(time)
    velocity_scale = rta / time
    pressure_scale = rho_background * velocity_scale**2
    shell_mass = density * (4.0 * np.pi / 3.0) * (
        boundaries[1:]**3 - boundaries[:-1]**3
    )
    mass = np.cumsum(shell_mass)
    mass_scale = 4.0 * np.pi / 3.0 * rho_background * rta**3
    fixed_mass = getattr(getattr(sim, 'par', None), 'dark_matter', None)
    if fixed_mass is not None:
        fixed_mass = float(np.asarray(fixed_mass.fixed_enclosed_mass, dtype=float))
        mass += fixed_mass
    return {
        'lambda': radius / rta,
        'density': density / rho_background,
        'velocity': velocity / velocity_scale,
        'pressure': pressure / pressure_scale,
        'mass': mass / mass_scale,
        'time': time,
    }


def _plot_comparison(numerical, reference, output, numerical_label):
    """Plot one RadHydro state and the standalone solution in similarity units."""
    lam = numerical['lambda']
    analytic = _interpolate_profile(reference, lam)
    analytic_profiles = {
        'density': analytic[0],
        'velocity': analytic[1],
        'pressure': analytic[2],
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), squeeze=False)
    axes = axes.ravel()
    for axis, name, scale in zip(
        axes[:3], ('density', 'velocity', 'pressure'),
        ('loglog', 'semilogx', 'loglog'),
    ):
        getattr(axis, scale)(lam, numerical[name], label=numerical_label)
        getattr(axis, scale)(lam, analytic_profiles[name], '--', label='standalone')
        axis.set(xlabel=r'$\lambda$', ylabel=name)
    axes[3].loglog(lam, numerical['mass'], label=numerical_label)
    axes[3].loglog(
        reference.lambda_in, reference.mass_in, '--', label='standalone interior'
    )
    axes[3].loglog(
        reference.lambda_out, reference.mass_out, '--', label='standalone exterior'
    )
    axes[3].set(xlabel=r'$\lambda$', ylabel='mass')
    for axis in axes:
        axis.axvline(reference.shock_lambda, color='k', linestyle=':', alpha=0.5)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[3].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return analytic_profiles


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)

    initial_condition = config['initial_condition']
    output = config["par"]['output']
    eu.clean_previous_outputs(config)
    reference = solve_bertschinger_gas()
    config['_reference_solution'] = reference
    initial = build_initial_condition(config)
    rio.writehdf5(initial, config["par"]['simulation']['initial_condition_filename'])

    sim = Rsim(config["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
    sim.SetMesh()
    sim.SetFluid()
    sim.fluid.SetFluidTime(sim.par.tau_supercomoving_code)
    sim.SetInitFluid()
    initial_tau = np.asarray(sim.par.tau_supercomoving_code, dtype=float)
    sim.par.tau_supercomoving_code = initial_tau.copy()
    sim.par.simulation.tau_supercomoving_code = initial_tau.copy()
    sim.fluid.SetFluidTime(initial_tau)
    if not (
        np.allclose(sim.par.tau_supercomoving_code, initial_tau)
        and np.allclose(sim.par.simulation.tau_supercomoving_code, initial_tau)
        and np.isclose(float(np.asarray(sim.fluid.tau_supercomoving_code)), float(initial_tau.flat[0]))
    ):
        raise RuntimeError("supercomoving startup clocks disagree after SetInitFluid")
    sim.par.cosmology = initial.par.cosmology
    sim.par.gravity = Gravity(
        selfgravity=True, cosmological=True, cosmology=sim.par.cosmology,
        dark_matter=initial.dark_matter, code_units=sim.par.units.CodeUnits,
    )
    sim.par.dark_matter = initial.dark_matter
    sim.solver = BertschingerBoundarySolver()
    initial_numerical = _similarity_profiles(sim, reference)
    sim.Run(mode='hydro')

    numerical = _similarity_profiles(sim, reference)
    initial_output = Path(output['savedir']) / 'BertschingerGasReference_RadHydroInitialCondition.jpg'
    _plot_comparison(initial_numerical, reference, initial_output, 'RadHydro IC')
    comparison_output = Path(output['savedir']) / 'BertschingerGasReference_RadHydroComparison.jpg'
    analytic_profiles = _plot_comparison(numerical, reference, comparison_output, 'RadHydro final')

    report = Path(output['savedir']) / 'BertschingerGasReference_RadHydroComparison.txt'
    lam = numerical['lambda']
    with report.open('w', encoding='utf-8') as stream:
        stream.write('final_similarity_time %.12g\n' % numerical['time'])
        stream.write('standalone_shock_lambda %.12g\n' % reference.shock_lambda)
        for name in ('density', 'velocity', 'pressure'):
            valid = np.isfinite(analytic_profiles[name])
            error = np.sqrt(np.mean(
                (numerical[name][valid] - analytic_profiles[name][valid])**2
            ))
            stream.write('%s_rms_error %.12g\n' % (name, error))
        outer = lam > 1.0
        stream.write('outer_density_mean %.12g\n' % np.mean(numerical['density'][outer]))
        stream.write('outer_density_min %.12g\n' % np.min(numerical['density'][outer]))
        stream.write('outer_density_max %.12g\n' % np.max(numerical['density'][outer]))
    print('shock lambda = %.8f' % reference.shock_lambda)
    print('initial-condition figure = %s' % initial_output)
    print('comparison figure = %s' % comparison_output)
    print('comparison report = %s' % report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
