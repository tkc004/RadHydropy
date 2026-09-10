"""Initial conditions and diagnostics for the PIE radiative shock tube."""

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, quantity_to_value
from basic_hydro_utils import finalize_initial_condition


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
BOLTZMANN_ERG_cgs_K = unyt.kb.to_value(unyt.erg / unyt.K)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    code_units = config['_code_units']
    result = Rsim(config["par"])
    result.par.simulation.coordinate_system = initial['coordinate_system']
    result.par.simulation.time_proper_code = quantity_to_value(initial['time_proper'], code_units.time_unit)
    result.par.simulation.box_size_proper_code = quantity_to_value(initial['box_size_proper'], code_units.length_unit)
    grid_cells = int(par['mesh']['grid_cells'])
    result.par.mesh.grid_cells = grid_cells
    result.par.mesh.ghost_cells = int(par['mesh']['ghost_cells'])
    box_size_proper_unyt = initial['box_size_proper']
    boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(0.0 * box_size_proper_unyt, box_size_proper_unyt, grid_cells + 1), code_units.length_unit
    ))
    width = np.diff(boundary_proper_code)
    x_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    midpoint_proper_code = quantity_to_value(
        0.5 * box_size_proper_unyt, code_units.length_unit
    )
    result.mesh.boundary_proper_code = boundary_proper_code
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=x_proper_code, boundary_proper_code=boundary_proper_code,
        width_proper_code=width, area_proper_code=np.ones(grid_cells), volume_proper_code=width,
    )
    collision_velocity_proper_code = quantity_to_value(
        initial['vel_collision_proper'], code_units.velocity_unit
    )
    result.fluid.vel_proper_code = as_named_array(np.where(
        x_proper_code < midpoint_proper_code,
        collision_velocity_proper_code,
        -collision_velocity_proper_code,
    ))
    hydrogen_mass_fraction = float(par['thermochemistry']['hydrogen_mass_fraction'])
    rho_proper_unyt = initial['hydrogen_number_density'] * unyt.mp / hydrogen_mass_fraction
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * rho_proper_unyt, code_units.density_unit
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['temperature_proper'], code_units.temperature_unit
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
    finalize_initial_condition(result, grid_cells)
    result.ConvertParametersToCodeUnits()
    return result


def load_output_state(filename, config):
    """Load one snapshot through the configured canonical runtime state."""
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    snapshot = Rsim(config['par'])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(filename))
    first = int(snapshot.par.mesh.ghost_cells)
    count = int(snapshot.par.mesh.grid_cells)
    physical = slice(first, first + count)
    boundary_proper_code = np.asarray(snapshot.mesh.boundary_proper_code)[
        first:first + count + 1
    ]
    return {
        'time_Myr': (
            float(np.asarray(snapshot.fluid.time_proper_code).reshape(-1)[0])
            * float(code_units.time_unit.to_value('s')) / SECONDS_PER_MYR
        ),
        'boundary_cgs_cm': boundary_proper_code * float(
            code_units.length_unit.to_value('cm')
        ),
        'density_cgs_g_cm3': np.asarray(snapshot.fluid.rho_proper_code)[physical] * float(
            code_units.density_unit.to_value('g/cm**3')
        ),
        'velocity_cgs_cm_s': np.asarray(snapshot.fluid.vel_proper_code)[physical] * float(
            code_units.velocity_unit.to_value('cm/s')
        ),
        'temperature_cgs_K': np.asarray(snapshot.fluid.temp_proper_code)[physical] * float(
            code_units.temperature_unit.to_value('K')
        ),
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
    temperature_proper_cgs_K,
    density_cgs_g_cm3,
    hydrogen_mass_fraction,
    mu,
    gamma,
    metallicity,
    redshift,
    post_velocity_cgs_cm_s,
):
    """Estimate l_cool = u_post * t_cool using the net PIE rate."""
    hydrogen_number_density_cgs_cm3 = density_cgs_g_cm3 * hydrogen_mass_fraction / PROTON_MASS_G
    heating, cooling = table.rates(
        temperature_proper_cgs_K,
        hydrogen_number_density_cgs_cm3,
        metallicity=metallicity,
        redshift=redshift,
    )
    net_cooling = max(float(np.asarray(cooling) - np.asarray(heating)), 1.0e-99)
    thermal_energy = 1.5 * density_cgs_g_cm3 * BOLTZMANN_ERG_cgs_K * temperature_proper_cgs_K / (
        mu * PROTON_MASS_G
    )
    cooling_time_s = thermal_energy / net_cooling
    return post_velocity_cgs_cm_s * cooling_time_s
