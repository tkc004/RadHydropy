"""HM12 PIE diagnostics for an NFW hydrostatic atmosphere."""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from radhydropy.arrays import as_named_array
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, quantity_to_value
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
    radius_inner_proper_unyt = initial['radius_inner_proper']
    radius_outer_proper_unyt = initial['radius_outer_proper']
    boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(
            radius_inner_proper_unyt,
            radius_outer_proper_unyt,
            grid_cells + 1,
        ), code_units.length_unit
    ))
    volume_proper_code = 4.0 * np.pi / 3.0 * (boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3)
    x_proper_code = 0.75 * (boundary_proper_code[1:]**4 - boundary_proper_code[:-1]**4) / (boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3)
    sim.mesh.boundary_proper_code = boundary_proper_code
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=x_proper_code, boundary_proper_code=boundary_proper_code,
        width_proper_code=np.diff(boundary_proper_code), area_proper_code=4.0 * np.pi * boundary_proper_code[:-1]**2,
        volume_proper_code=volume_proper_code,
    )
    halo = nfw_halo_parameters(
        initial['halo_mass'], initial['concentration'], initial['redshift'],
        initial['overdensity'], initial['h0'],
    )
    temperature_virial_unyt = virial_temperature(halo, initial['mu'])
    radius_proper_unyt = spherical_cell_centers(np.asarray(boundary_proper_code) * code_units.length_unit)
    rho_proper_cgs_g_cm3_unyt = hydrostatic_density_profile(
        radius_proper_unyt, np.asarray(boundary_proper_code) * code_units.length_unit, halo, temperature_virial_unyt,
        initial['mu'], initial['gas_fraction'],
    )
    sim.fluid.rho_proper_code = as_named_array(quantity_to_value(rho_proper_cgs_g_cm3_unyt, code_units.density_unit))
    sim.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells))
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(np.ones(grid_cells) * temperature_virial_unyt, code_units.temperature_unit))
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


def load_output_state(filename, config):
    code_units = config.get('_code_units')
    if code_units is None:
        code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    snapshot = Rsim(config['par'])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(filename))
    first = int(snapshot.par.mesh.ghost_cells)
    count = int(snapshot.par.mesh.grid_cells)
    physical = slice(first, first + count)
    boundary_proper_code = np.asarray(snapshot.mesh.boundary_proper_code)[
        first:first + count + 1
    ]
    radius_proper_kpc = spherical_cell_centers(
        boundary_proper_code * code_units.length_unit
    ).to_value(unyt.kpc)
    rho_proper_cgs_g_cm3 = (
        np.asarray(snapshot.fluid.rho_proper_code)[physical]
        * code_units.density_unit
    ).to_value(unyt.g / unyt.cm**3)
    temperature_proper_cgs_K = (
        np.asarray(snapshot.fluid.temp_proper_code)[physical]
        * code_units.temperature_unit
    ).to_value(unyt.K)
    vel_peculiar_proper_km_s = (
        np.asarray(snapshot.fluid.vel_proper_code)[physical]
        * code_units.velocity_unit
    ).to_value(unyt.km / unyt.s)
    time_proper_code = float(np.asarray(snapshot.fluid.time_proper_code).reshape(-1)[0])
    return {
        'time_proper_code': time_proper_code,
        'radius_proper_kpc': radius_proper_kpc,
        'rho_proper_cgs_g_cm3': rho_proper_cgs_g_cm3,
        'temperature_proper_cgs_K': temperature_proper_cgs_K,
        'vel_peculiar_proper_km_s': vel_peculiar_proper_km_s,
    }


def analyze_snapshot(filename, config, halo, temperature_virial_unyt):
    snapshot = load_output_state(filename, config)
    time_proper_code = snapshot['time_proper_code']
    radius_proper_kpc = snapshot['radius_proper_kpc']
    rho_proper_cgs_g_cm3 = snapshot['rho_proper_cgs_g_cm3']
    temperature_proper_cgs_K = snapshot['temperature_proper_cgs_K']
    vel_peculiar_proper_km_s = snapshot['vel_peculiar_proper_km_s']
    radius_proper_cgs_cm = radius_proper_kpc * (1.0 * unyt.kpc).to_value(unyt.cm)
    mu = float(config['initial_condition']['mu'])
    pre_proper_cgs_erg_cm3 = rho_proper_cgs_g_cm3 * BOLTZMANN_CONSTANT_CGS * temperature_proper_cgs_K / (mu * PROTON_MASS_CGS)
    mass_proper_cgs_g = nfw_enclosed_mass(radius_proper_cgs_cm * unyt.cm, halo).to_value(unyt.g)
    gravity = unyt.physical_constants.gravitational_constant.to_value(
        unyt.cm**3 / unyt.g / unyt.s**2
    ) * mass_proper_cgs_g / np.maximum(radius_proper_cgs_cm, 1.0) ** 2
    dpdr_proper_cgs = np.gradient(pre_proper_cgs_erg_cm3, radius_proper_cgs_cm)
    force_residual = (dpdr_proper_cgs + rho_proper_cgs_g_cm3 * gravity) / np.maximum(rho_proper_cgs_g_cm3 * gravity, 1.0e-99)
    r200 = halo['radius_virial_proper_kpc_unyt'].to_value(unyt.kpc)
    inside = radius_proper_kpc <= r200
    shell_width_proper_cgs_cm = np.gradient(radius_proper_cgs_cm)
    atmosphere_mass = float(np.sum(4.0 * np.pi * radius_proper_cgs_cm[inside]**2
                                   * shell_width_proper_cgs_cm[inside] * rho_proper_cgs_g_cm3[inside]))
    central = radius_proper_kpc < 0.1 * r200
    return {
        'time_proper_Myr': time_proper_code,
        'radius_proper_kpc': radius_proper_kpc,
        'rho_proper_cgs_g_cm3': rho_proper_cgs_g_cm3,
        'temperature_proper_cgs_K': temperature_proper_cgs_K,
        'vel_peculiar_proper_km_s': vel_peculiar_proper_km_s,
        'pressure_cgs_erg_cm3': pre_proper_cgs_erg_cm3,
        'force_residual': force_residual,
        'atmosphere_mass_Msun': atmosphere_mass / unyt.Msun.to_value(unyt.g),
        'central_rho_proper_cgs_g_cm3': float(np.median(rho_proper_cgs_g_cm3[central])),
        'central_temperature_proper_cgs_K': float(np.median(temperature_proper_cgs_K[central])),
        'minimum_temperature_proper_cgs_K': float(np.min(temperature_proper_cgs_K)),
        'temperature_virial_K': temperature_virial_unyt.to_value(unyt.K),
    }


def write_report(results, filename, temperature_floor):
    with open(filename, 'w', encoding='utf-8') as report:
        report.write(
            'time_proper_Myr central_rho_proper_cgs_g_cm3 central_temperature_proper_cgs_K '
            'minimum_temperature_proper_cgs_K atmosphere_mass_Msun max_abs_force_residual '
            'temperature_floor_cgs_K floor_reached\n'
        )
        for row in results:
            report.write(
                '%.8g %.8g %.8g %.8g %.8g %.8g %.8g %s\n' % (
                    row['time_proper_Myr'], row['central_rho_proper_cgs_g_cm3'],
                    row['central_temperature_proper_cgs_K'], row['minimum_temperature_proper_cgs_K'],
                    row['atmosphere_mass_Msun'],
                    np.nanmax(np.abs(row['force_residual'])),
                    temperature_floor, row['minimum_temperature_proper_cgs_K'] <= temperature_floor * 1.01,
                )
            )


def plot_results(results, halo, filename):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(results)))
    r200 = halo['radius_virial_proper_kpc_unyt'].to_value(unyt.kpc)
    for color, row in zip(colors, results):
        label = f"{row['time_proper_Myr']:.0f} Myr"
        axes[0, 0].plot(row['radius_proper_kpc'], row['rho_proper_cgs_g_cm3'], color=color, label=label)
        axes[0, 1].plot(row['radius_proper_kpc'], row['temperature_proper_cgs_K'], color=color, label=label)
        axes[1, 0].plot(row['radius_proper_kpc'], row['vel_peculiar_proper_km_s'], color=color, label=label)
        axes[1, 1].plot(row['radius_proper_kpc'], row['force_residual'], color=color, label=label)
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
