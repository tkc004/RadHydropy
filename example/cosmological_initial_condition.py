"""Shared writer-backed initial-condition builder for cosmological tests."""

import copy

import numpy as np

from radhydropy.cosmology_context import CosmologyContext
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.runtime_fields import MeshGeometryState, SUPERCOMOVING_RUNTIME_FIELDS


def build_initial_condition(config):
    """Build a cosmological IC through :class:`InitialConditionWriter`."""
    par = config["par"]
    initial_condition = config["initial_condition"]
    code_cosmology = config["_code_cosmology"]

    writer = InitialConditionWriter(
        par_config=copy.deepcopy(par),
        cosmology_context=CosmologyContext(
            gamma=float(par["hydrodynamics"]["gamma"]),
            cosmology=code_cosmology.type_name,
        ),
    )
    result = writer.simulation
    code_units = writer.code_units
    grid_cells = int(par["mesh"]["grid_cells"])
    initial_time_code = float(
        initial_condition["time_cosmic"].to_value(code_units.time_unit)
    )
    scale_factor = float(code_cosmology.scale_factor(initial_time_code))
    hubble_unit_km_s_Mpc = (
        code_units.velocity_unit.to_value("km/s")
        / code_units.length_unit.to_value("Mpc")
    )
    result.par.cosmology_context = CosmologyContext(
        gamma=float(par["hydrodynamics"]["gamma"]),
        cosmology=code_cosmology.type_name,
        scale_factor=scale_factor,
        hubble_parameter_km_s_Mpc=(
            float(code_cosmology.hubble(initial_time_code))
            * hubble_unit_km_s_Mpc
        ),
    )
    result.par.cosmological_expansion = True
    result.par.supercomoving_coordinates = True
    result.par.cosmological_gravity = bool(
        par.get("cosmology", {}).get("cosmological", False)
    )
    result.par.cosmology = code_cosmology
    result.par.cosmology_type = code_cosmology.type_name
    result.par.cosmology_t_ref = code_cosmology.t_ref
    result.par.cosmology_a_ref = code_cosmology.a_ref
    result.par.coordinate_frame = "comoving"
    result.par.time_coordinate = "supercomoving"
    result.par.velocity_representation = "supercomoving_peculiar"
    result.par.density_representation = "comoving"
    result.par.pressure_representation = "supercomoving"
    result.par.temperature_representation = "supercomoving"

    initial_tau = config.get("_initial_tau_supercomoving_code")
    if initial_tau is None:
        initial_tau = float(code_cosmology.supercomoving_time(initial_time_code))
    result.par.tau_supercomoving_code = np.asarray([float(initial_tau)])
    result.par.simulation.tau_supercomoving_code = result.par.tau_supercomoving_code

    box_size_comoving_code = float(
        initial_condition["box_size_comoving"].to_value(code_units.length_unit)
    )
    result.par.simulation.box_size_comoving_code = box_size_comoving_code
    result.par.simulation.coordinate_system = par["simulation"].get(
        "coordinate_system", "cartesian"
    )
    boundary_override = config.get("_boundary_comoving_code")
    if boundary_override is None:
        boundary_start = float(config.get("_boundary_start_code", 0.0))
        boundary = np.linspace(
            boundary_start, boundary_start + box_size_comoving_code, grid_cells + 1
        )
    else:
        boundary = np.asarray(boundary_override, dtype=float)
    x = np.asarray(
        config.get("_x_comoving_code", 0.5 * (boundary[:-1] + boundary[1:])),
        dtype=float,
    )
    width = np.diff(boundary)
    if result.par.simulation.coordinate_system == "spherical":
        area = np.asarray(
            config.get("_area_comoving_code", 4.0 * np.pi * boundary[:-1] ** 2),
            dtype=float,
        )
        volume = np.asarray(
            config.get(
                "_volume_comoving_code",
                4.0 * np.pi / 3.0 * (boundary[1:] ** 3 - boundary[:-1] ** 3),
            ),
            dtype=float,
        )
    else:
        area = np.asarray(config.get("_area_comoving_code", np.ones(grid_cells)), dtype=float)
        volume = np.asarray(config.get("_volume_comoving_code", width), dtype=float)

    writer.box_size = writer.radquantity(initial_condition["box_size_comoving"])
    writer.mesh.boundary_radarray = writer.radarray(
        boundary * code_units.length_unit, representation="comoving"
    )
    writer.set_field(
        "x_comoving_code",
        x,
    )
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        x_comoving_code=x,
        boundary_comoving_code=boundary,
        width_comoving_code=width,
        area_comoving_code=area,
        volume_comoving_code=volume,
    )

    if "rho_left_proper" in initial_condition:
        left = x < 0.5 * box_size_comoving_code
        rho_proper_code = np.where(
            left,
            float(initial_condition["rho_left_proper"].to_value(code_units.density_unit)),
            float(initial_condition["rho_right_proper"].to_value(code_units.density_unit)),
        )
        temp_proper_code = np.where(
            left,
            float(initial_condition["temperature_left_proper"].to_value(code_units.temperature_unit)),
            float(initial_condition["temperature_right_proper"].to_value(code_units.temperature_unit)),
        )
        rho_field, temp_field = "rho_proper_code", "temp_proper_code"
    else:
        rho_comoving_code = np.asarray(config["_rho_comoving_code"], dtype=float)
        temp_supercomoving_code = np.asarray(config["_temp_supercomoving_code"], dtype=float)
        rho_field, temp_field = "rho_comoving_code", "temp_supercomoving_code"
    vel_supercomoving_code = np.asarray(
        config.get("_vel_supercomoving_code", np.zeros(grid_cells)), dtype=float
    )
    mu = np.asarray(config.get("_mu_dimensionless", np.ones(grid_cells)), dtype=float)
    writer.fluid.rho_radarray = writer.radarray(
        (rho_proper_code if rho_field == "rho_proper_code" else rho_comoving_code)
        * code_units.density_unit,
        representation="proper" if rho_field == "rho_proper_code" else "comoving",
    )
    writer.fluid.temp_radarray = writer.radarray(
        (temp_proper_code if temp_field == "temp_proper_code" else temp_supercomoving_code)
        * code_units.temperature_unit,
        representation="proper" if temp_field == "temp_proper_code" else "supercomoving",
    )
    writer.fluid.vel_radarray = writer.radarray(
        vel_supercomoving_code * code_units.velocity_unit,
        representation="supercomoving",
    )
    result.fluid.mu = mu
    if "_specific_angular_momentum_code" in config:
        result.par.gas_angular_momentum = True
        result.par.gas_rotational_energy = True
        writer.fluid.specific_angular_momentum_radarray = writer.radarray(
            np.asarray(config["_specific_angular_momentum_code"], dtype=float)
            * (code_units.length_unit**2 / code_units.time_unit),
            field_name="specific_angular_momentum_code",
        )
    result.fluid.tau_supercomoving_code = float(result.par.tau_supercomoving_code[0])
    return writer
