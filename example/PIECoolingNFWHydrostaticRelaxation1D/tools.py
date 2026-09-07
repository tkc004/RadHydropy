"""HM12 PIE diagnostics for an NFW hydrostatic atmosphere."""

import importlib.util
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value
from basic_hydro_utils import finalize_initial_condition

BASE_PATH = Path(__file__).resolve().parents[1] / 'NFWHydrostaticEquilibrium1D' / 'tools.py'
SPEC = importlib.util.spec_from_file_location('nfw_hydrostatic_tools_for_pie', BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS

nfw_halo_parameters = BASE.nfw_halo_parameters
virial_temperature = BASE.virial_temperature
spherical_cell_centers = BASE.spherical_cell_centers
hydrostatic_density_profile = BASE.hydrostatic_density_profile
nfw_enclosed_mass = BASE.nfw_enclosed_mass


def build_initial_condition(config):
    """Build the hydrostatic NFW atmosphere using canonical runtime fields."""
    initial = config['initial_condition']
    sim = Rsim(config['par'])
    code_units = sim.par.units.CodeUnits
    grid_cells = int(config['par']['mesh']['grid_cells'])
    rmin = initial.get('rmin', initial.get('inner_radius'))
    rmax = initial.get('rmax', initial.get('outer_radius'))
    boundary = as_named_array(quantity_to_value(
        np.linspace(rmin, rmax, grid_cells + 1), code_units.length_unit
    ))
    volume = 4.0 * np.pi / 3.0 * (boundary[1:]**3 - boundary[:-1]**3)
    coordinate = 0.75 * (boundary[1:]**4 - boundary[:-1]**4) / (boundary[1:]**3 - boundary[:-1]**3)
    sim.mesh.boundary_proper_code = boundary
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=coordinate, boundary=boundary,
        width=np.diff(boundary), area=4.0 * np.pi * boundary[:-1]**2,
        volume=volume,
    )
    halo = nfw_halo_parameters(
        initial['halo_mass'], initial['concentration'], initial['redshift'],
        initial['overdensity'], initial['h0'],
    )
    temperature = virial_temperature(halo, initial['mu'])
    radius = spherical_cell_centers(np.asarray(boundary) * code_units.length_unit)
    density = hydrostatic_density_profile(
        radius, np.asarray(boundary) * code_units.length_unit, halo, temperature,
        initial['mu'], initial['gas_fraction'],
    )
    sim.fluid.rho_proper_code = as_named_array(quantity_to_value(density, code_units.density_unit))
    sim.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells))
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(np.ones(grid_cells) * temperature, code_units.temperature_unit))
    sim.fluid.mu = as_named_array(np.ones(grid_cells) * initial['mu'])
    sim.fluid.time_proper_code = 0.0
    sim.SetMesh()
    sim.fluid.SetUpFluid(sim.par, sim.mesh)
    sim.fluid.SetFluidTime(0.0)
    sim.fluid.SetEnergyDensity()
    sim.mesh._par = sim.par
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    finalize_initial_condition(sim, grid_cells)
    sim.ConvertParametersToCodeUnits()
    return sim


def load_snapshot(filename, config):
    with h5py.File(filename, 'r') as handle:
        data = handle['Data']
        header = handle['Header']
        noghost = int(config['par']['mesh']['ghost_cells'])
        nogrid = int(header.attrs['GridCells'])
        boundary = np.asarray(data['boundary_proper_code'][()])
        boundary = boundary[noghost:noghost + nogrid + 1]
        # Raw output datasets are written in their physical units (cm, g cm^-3,
        # K, and cm s^-1).  The CodeUnits metadata describes the runtime state,
        # but must not be applied a second time to these HDF5 values.
        radius = spherical_cell_centers(boundary * unyt.cm).to_value(unyt.kpc)
        density = np.asarray(data['rho_proper_code'][()])[noghost:noghost + nogrid]
        temperature = np.asarray(data['temp_proper_code'][()])[noghost:noghost + nogrid]
        velocity = (np.asarray(data['vel_proper_code'][()])[noghost:noghost + nogrid]
                    / 1.0e5)
        time_proper_code = float(header.attrs.get('time_proper_code', 0.0))
        # Fixed output-time files store the physical time in the fluid state;
        # the header time is retained as a fallback for older snapshots.
        return time_proper_code, radius, density, temperature, velocity


def analyze_snapshot(filename, config, halo, temperature):
    time, radius, density, temp, velocity = load_snapshot(filename, config)
    radius_cgs_cm = radius * (1.0 * unyt.kpc).to_value(unyt.cm)
    mu = float(config['initial_condition']['mu'])
    pressure = density * BOLTZMANN_CONSTANT_CGS * temp / (mu * PROTON_MASS_CGS)
    mass = nfw_enclosed_mass(radius_cgs_cm * unyt.cm, halo).to_value(unyt.g)
    gravity = unyt.physical_constants.gravitational_constant.to_value(
        unyt.cm**3 / unyt.g / unyt.s**2
    ) * mass / np.maximum(radius_cgs_cm, 1.0) ** 2
    dpdr = np.gradient(pressure, radius_cgs_cm)
    force_residual = (dpdr + density * gravity) / np.maximum(density * gravity, 1.0e-99)
    r200 = halo['virial_radius'].to_value(unyt.kpc)
    inside = radius <= r200
    shell_edges = np.gradient(radius_cgs_cm)
    atmosphere_mass = float(np.sum(4.0 * np.pi * radius_cgs_cm[inside]**2
                                   * shell_edges[inside] * density[inside]))
    central = radius < 0.1 * r200
    return {
        'time_Myr': time,
        'radius_kpc': radius,
        'density_cgs_g_cm3': density,
        'temperature_cgs_K': temp,
        'velocity_km_s': velocity,
        'pressure_cgs_erg_cm3': pressure,
        'force_residual': force_residual,
        'atmosphere_mass_Msun': atmosphere_mass / unyt.Msun.to_value(unyt.g),
        'central_density_cgs_g_cm3': float(np.median(density[central])),
        'central_temperature_cgs_K': float(np.median(temp[central])),
        'minimum_temperature_cgs_K': float(np.min(temp)),
        'temperature': temperature.to_value(unyt.K),
    }


def write_report(results, filename, temperature_floor):
    with open(filename, 'w', encoding='utf-8') as report:
        report.write(
            'time_Myr central_density_cgs_g_cm3 central_temperature_cgs_K '
            'minimum_temperature_cgs_K atmosphere_mass_Msun max_abs_force_residual '
            'temperature_floor_cgs_K floor_reached\n'
        )
        for row in results:
            report.write(
                '%.8g %.8g %.8g %.8g %.8g %.8g %.8g %s\n' % (
                    row['time_Myr'], row['central_density_cgs_g_cm3'],
                    row['central_temperature_cgs_K'], row['minimum_temperature_cgs_K'],
                    row['atmosphere_mass_Msun'],
                    np.nanmax(np.abs(row['force_residual'])),
                    temperature_floor, row['minimum_temperature_cgs_K'] <= temperature_floor * 1.01,
                )
            )


def plot_results(results, halo, filename):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(results)))
    r200 = halo['virial_radius'].to_value(unyt.kpc)
    for color, row in zip(colors, results):
        label = f"{row['time_Myr']:.0f} Myr"
        axes[0, 0].plot(row['radius_kpc'], row['density_cgs_g_cm3'], color=color, label=label)
        axes[0, 1].plot(row['radius_kpc'], row['temperature_cgs_K'], color=color, label=label)
        axes[1, 0].plot(row['radius_kpc'], row['velocity_km_s'], color=color, label=label)
        axes[1, 1].plot(row['radius_kpc'], row['force_residual'], color=color, label=label)
    axes[0, 0].set_ylabel(r'$\rho$ [g cm$^{-3}$]'); axes[0, 0].set_yscale('log')
    axes[0, 1].set_ylabel('$T$ [K]'); axes[0, 1].set_yscale('log')
    axes[1, 0].set_ylabel('$v_r$ [km s$^{-1}$]')
    axes[1, 1].set_ylabel(r'$(dP/dr+\rho g)/(\rho g)$')
    axes[1, 1].axhline(0.0, color='black', ls=':')
    for axis in axes.flat:
        axis.set_xlabel('r [kpc]')
        axis.axvline(r200, color='black', ls='--', alpha=0.5)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    fig.suptitle('HM12 PIE relaxation of an NFW hydrostatic atmosphere')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)
