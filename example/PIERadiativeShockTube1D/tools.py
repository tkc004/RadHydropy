"""Initial conditions and diagnostics for the PIE radiative shock tube."""

import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
BOLTZMANN_ERG_cgs_K = unyt.kb.to_value(unyt.erg / unyt.K)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    box_size_proper_unyt = initial['box_size_proper']
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * box_size_proper_unyt
    coordinate_proper_unyt = 0.5 * (boundary_proper_unyt[:-1] + boundary_proper_unyt[1:])
    collision_velocity_proper_unyt = initial['vel_collision_proper']
    midpoint_proper_unyt = 0.5 * box_size_proper_unyt
    velocity_proper_unyt = np.where(
        coordinate_proper_unyt < midpoint_proper_unyt,
        collision_velocity_proper_unyt,
        -collision_velocity_proper_unyt,
    )
    hydrogen_mass_fraction = float(config['par']['thermochemistry']['hydrogen_mass_fraction'])
    rho_proper_unyt = initial['hydrogen_number_density'] * unyt.mp / hydrogen_mass_fraction
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(box_size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(np.ones(grid_cells) * rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(velocity_proper_unyt)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial['temperature_proper'])
    writer.fluid.mu = np.full(grid_cells, initial['mean_molecular_weight'])
    return writer


def load_output_state(filename, config):
    """Load one snapshot through the configured canonical runtime state."""
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    snapshot = rio.loadhdf5(config, str(filename))
    first = int(snapshot.par.mesh.ghost_cells)
    count = int(snapshot.par.mesh.grid_cells)
    physical = slice(first, first + count)
    boundary_proper_code = snapshot.mesh.boundary_radarray.to_value(code_units.length_unit)[
        first:first + count + 1
    ]
    return {
        'time_proper_Myr': (
            float(np.asarray(snapshot.fluid.time_proper_code).reshape(-1)[0])
            * float(code_units.time_unit.to_value('s')) / SECONDS_PER_MYR
        ),
        'boundary_proper_cgs_cm': snapshot.mesh.boundary_radarray.to(unyt.cm).value[first:first + count + 1],
        'rho_proper_cgs_g_cm3': snapshot.fluid.rho_radarray.to(unyt.g / unyt.cm**3).value[physical],
        'vel_peculiar_proper_cgs_cm_s': snapshot.fluid.vel_radarray.to(unyt.cm / unyt.s).value[physical],
        'temperature_proper_cgs_K': snapshot.fluid.temp_radarray.to(unyt.K).value[physical],
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
    rho_proper_cgs_g_cm3,
    hydrogen_mass_fraction,
    mu,
    gamma,
    metallicity,
    redshift,
    post_velocity_cgs_cm_s,
):
    """Estimate l_cool = u_post * t_cool using the net PIE rate."""
    hydrogen_number_density_cgs_cm3 = rho_proper_cgs_g_cm3 * hydrogen_mass_fraction / PROTON_MASS_G
    heating, cooling = table.rates(
        temperature_proper_cgs_K,
        hydrogen_number_density_cgs_cm3,
        metallicity=metallicity,
        redshift=redshift,
    )
    net_cooling = max(float(np.asarray(cooling) - np.asarray(heating)), 1.0e-99)
    thermal_energy = 1.5 * rho_proper_cgs_g_cm3 * BOLTZMANN_ERG_cgs_K * temperature_proper_cgs_K / (
        mu * PROTON_MASS_G
    )
    cooling_time_s = thermal_energy / net_cooling
    return post_velocity_cgs_cm_s * cooling_time_s
