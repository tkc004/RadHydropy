"""Initial condition and plotting helpers for outflow into vacuum."""

import numpy as np
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value


def analytic_density_profile(radius_proper_code, time_proper_code, config, cell_faces=None):
    """Cold spherical outflow profile, sampled as cell averages when given."""
    radius_proper_code = np.asarray(radius_proper_code, dtype=float)
    initial = config['initial_condition']
    boundary = config['par']['boundary']
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    injection_radius_proper_code = quantity_to_value(
        initial['radius_injection_proper'], code_units.length_unit
    )
    density_outflow_proper_code = quantity_to_value(
        boundary['rho_outflow_proper'], code_units.density_unit
    )
    velocity_outflow_proper_code = quantity_to_value(
        boundary['vel_outflow_proper'], code_units.velocity_unit
    )
    front_radius_proper_code = injection_radius_proper_code + velocity_outflow_proper_code * float(time_proper_code)
    profile = np.full_like(radius_proper_code, np.nan, dtype=float)
    if cell_faces is None:
        inside = (radius_proper_code >= injection_radius_proper_code) & (radius_proper_code <= front_radius_proper_code)
        profile[inside] = density_outflow_proper_code * (injection_radius_proper_code / radius_proper_code[inside])**2
        return profile, front_radius_proper_code
    faces = np.asarray(cell_faces, dtype=float)
    left = np.maximum(faces[:-1], injection_radius_proper_code)
    right = np.minimum(faces[1:], front_radius_proper_code)
    inside = right > left
    volume_factor = faces[1:]**3 - faces[:-1]**3
    profile[inside] = (
        3.0 * density_outflow_proper_code * injection_radius_proper_code**2
        * (right[inside] - left[inside]) / volume_factor[inside]
    )
    return profile, front_radius_proper_code


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    grid_cells = int(initial['grid_cells'])
    boundary_proper_unyt = np.linspace(
        0.0, 1.0, grid_cells + 1
    ) * initial['box_size_proper'] + initial['radius_injection_proper']
    inner_proper_unyt = boundary_proper_unyt[:-1]
    outer_proper_unyt = boundary_proper_unyt[1:]
    coordinate_proper_unyt = 0.75 * (
        outer_proper_unyt**4 - inner_proper_unyt**4
    ) / (outer_proper_unyt**3 - inner_proper_unyt**3)
    writer = InitialConditionWriter(
        ic_config=config["initial_condition"],
        par_config=config['par'], code_units=code_units
    )
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(coordinate_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(grid_cells) * code_units.velocity_unit
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.zeros(grid_cells) * code_units.temperature_unit
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.zeros(grid_cells) * code_units.density_unit
    )
    writer.simulation.fluid.mu = np.full(
        grid_cells, float(initial['mean_molecular_weight'])
    )
    return writer
