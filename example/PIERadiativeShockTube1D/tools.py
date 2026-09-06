"""Initial conditions and diagnostics for the PIE radiative shock tube."""

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
BOLTZMANN_ERG_cgs_K = unyt.kb.to_value(unyt.erg / unyt.K)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    code_units = config['_code_units']
    result = Rsim(par)
    result.par.simulation.coordinate_system = initial['coordinate_system']
    result.par.simulation.time_code = quantity_to_value(initial['current_time'], code_units.time_unit)
    result.par.simulation.box_size = quantity_to_value(initial['box_size'], code_units.length_unit)
    grid_cells = int(par['mesh']['grid_cells'])
    result.par.mesh.grid_cells = grid_cells
    result.par.mesh.ghost_cells = int(par['mesh']['ghost_cells'])
    boxsize = initial['box_size']
    boundary = as_named_array(quantity_to_value(
        np.linspace(0.0 * boxsize, boxsize, grid_cells + 1), code_units.length_unit
    ))
    width = np.diff(boundary)
    coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    midpoint_proper_code = quantity_to_value(
        0.5 * boxsize, code_units.length_unit
    )
    result.mesh.boundary_proper_code = boundary
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=coordinate, boundary=boundary,
        width=width, area=np.ones(grid_cells), volume=width,
    )
    collision_velocity_proper_code = quantity_to_value(
        initial['collision_velocity'], code_units.velocity_unit
    )
    result.fluid.vel_proper_code = as_named_array(np.where(
        coordinate < midpoint_proper_code,
        collision_velocity_proper_code,
        -collision_velocity_proper_code,
    ))
    hydrogen_mass_fraction = float(par['thermochemistry']['hydrogen_mass_fraction'])
    rho = initial['hydrogen_density'] * unyt.mp / hydrogen_mass_fraction
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * rho, code_units.density_unit
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['initial_temperature'], code_units.temperature_unit
    ))
    result.fluid.mu = np.ones(result.par.mesh.grid_cells) * initial['mean_molecular_weight']
    result.fluid.time_proper_code = 0.0
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.fluid.SetFluidTime(0.0)
    result.fluid.SetEnergyDensity()
    result.fluid._refresh_runtime_state()
    result.mesh._par = result.par
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    result.ConvertParametersToCodeUnits()
    return result


def _physical_cells(data, header):
    noghost = int(header.attrs.get('GhostCells', 0))
    nogrid = int(header.attrs['GridCells'])
    return slice(noghost, noghost + nogrid)


def load_snapshot(filename):
    import h5py

    with h5py.File(filename, 'r') as handle:
        data = handle['Data']
        header = handle['Header']
        physical = _physical_cells(data, header)
        noghost = int(header.attrs.get('GhostCells', 0))
        nogrid = int(header.attrs['GridCells'])
        boundary = np.asarray(data['boundary_proper_code'][()])[noghost:noghost + nogrid + 1]
        return {
            'time_Myr': float(header['time_proper_code'][()]) / SECONDS_PER_MYR,
            'boundary_cgs_cm': boundary,
            'density_cgs_g_cm3': np.asarray(data['rho_proper_code'][()])[physical],
            'velocity_cgs_cm_s': np.asarray(data['vel_proper_code'][()])[physical],
            'temperature_cgs_K': np.asarray(data['temp_proper_code'][()])[physical],
        }


def strong_shock_expectation(gamma, upstream_velocity_cgs_cm_s, mu):
    """Return strong-shock compression, postshock speed and temperature."""
    compression = (gamma + 1.0) / (gamma - 1.0)
    post_velocity = upstream_velocity_cgs_cm_s / compression
    post_temperature = (
        2.0 * (gamma - 1.0) / (gamma + 1.0) ** 2
        * mu * PROTON_MASS_G / BOLTZMANN_ERG_cgs_K
        * upstream_velocity_cgs_cm_s ** 2
    )
    return compression, post_velocity, post_temperature


def cooling_length_estimate(
    table,
    temperature,
    density_cgs_g_cm3,
    hydrogen_mass_fraction,
    mu,
    gamma,
    metallicity,
    redshift,
    post_velocity_cgs_cm_s,
):
    """Estimate l_cool = u_post * t_cool using the net PIE rate."""
    hydrogen_density = density_cgs_g_cm3 * hydrogen_mass_fraction / PROTON_MASS_G
    heating, cooling = table.rates(
        temperature, hydrogen_density, metallicity=metallicity, redshift=redshift
    )
    net_cooling = max(float(np.asarray(cooling) - np.asarray(heating)), 1.0e-99)
    thermal_energy = 1.5 * density_cgs_g_cm3 * BOLTZMANN_ERG_cgs_K * temperature / (
        mu * PROTON_MASS_G
    )
    cooling_time_s = thermal_energy / net_cooling
    return post_velocity_cgs_cm_s * cooling_time_s
