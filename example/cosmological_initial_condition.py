"""Shared canonical initial-condition builder for Cartesian cosmological tests."""

import numpy as np

from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    SUPERCOMOVING_RUNTIME_FIELDS,
)


def build_initial_condition(config):
    """Build a typed Cartesian supercomoving state from a nested config.

    The caller may attach the runtime-only code cosmology as
    ``config["_code_cosmology"]``.  Density-evolution and Hubble-flow cases
    attach their computed uniform profile through the explicitly named
    ``*_comoving_code`` and ``*_supercomoving_code`` private inputs; Sod cases
    use the physical left/right values in ``initial_condition`` directly.
    """
    par = config["par"]
    initial_condition = config["initial_condition"]
    code_cosmology = config["_code_cosmology"]
    result = Rsim(config["par"])
    code_units = result.par.units.CodeUnits
    grid_cells = int(par["mesh"]["grid_cells"])
    box_size_comoving_code = float(
        initial_condition["box_size_comoving"].to_value(code_units.length_unit)
    )
    initial_time_code = float(
        initial_condition["time_cosmic"].to_value(code_units.time_unit)
    )
    initial_tau_supercomoving_code = config.get(
        "_initial_tau_supercomoving_code"
    )
    if initial_tau_supercomoving_code is None:
        initial_tau_supercomoving_code = float(
            code_cosmology.supercomoving_time(initial_time_code)
        )
    result.par.tau_supercomoving_code = np.asarray(
        [float(initial_tau_supercomoving_code)]
    )
    result.par.simulation.tau_supercomoving_code = result.par.tau_supercomoving_code
    result.par.simulation.box_size_comoving_code = box_size_comoving_code
    result.par.simulation.coordinate_system = par["simulation"].get(
        "coordinate_system", "cartesian"
    )
    result.par.cosmological_expansion = True
    result.par.supercomoving_coordinates = True
    result.par.cosmological_gravity = bool(
        par.get("gravity", {}).get("cosmological_gravity", False)
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

    boundary_override = config.get("_boundary_comoving_code")
    if boundary_override is None:
        boundary_start_code = float(config.get("_boundary_start_code", 0.0))
        boundary_comoving_code = np.linspace(
            boundary_start_code, boundary_start_code + box_size_comoving_code,
            grid_cells + 1,
        )
    else:
        boundary_comoving_code = np.asarray(boundary_override, dtype=float)
    result.mesh.boundary_comoving_code = boundary_comoving_code
    result.mesh.x_comoving_code = np.asarray(
        config.get("_x_comoving_code", 0.5 * (boundary_comoving_code[1:] + boundary_comoving_code[:-1])),
        dtype=float,
    )
    result.mesh.width_comoving_code = np.diff(boundary_comoving_code)
    if par["simulation"].get("coordinate_system") == "spherical":
        result.mesh.area_comoving_code = np.asarray(
            config.get("_area_comoving_code", 4.0 * np.pi * boundary_comoving_code[:-1] ** 2),
            dtype=float,
        )
        result.mesh.volume_comoving_code = np.asarray(
            config.get(
                "_volume_comoving_code",
                4.0 * np.pi / 3.0 * (boundary_comoving_code[1:] ** 3 - boundary_comoving_code[:-1] ** 3),
            ),
            dtype=float,
        )
    else:
        result.mesh.area_comoving_code = np.asarray(
            config.get("_area_comoving_code", np.ones(grid_cells)), dtype=float
        )
        result.mesh.volume_comoving_code = np.asarray(
            config.get("_volume_comoving_code", np.diff(boundary_comoving_code)), dtype=float
        )

    if "rho_left_proper" in initial_condition:
        left = result.mesh.x_comoving_code < 0.5 * box_size_comoving_code
        rho_left_proper_code = float(
            initial_condition["rho_left_proper"].to_value(code_units.density_unit)
        )
        rho_right_proper_code = float(
            initial_condition["rho_right_proper"].to_value(code_units.density_unit)
        )
        rho_comoving_code = np.where(
            left, rho_left_proper_code, rho_right_proper_code,
        )
        temp_left_proper_code = initial_condition["temperature_left_proper"].to_value(
            code_units.temperature_unit
        )
        temp_right_proper_code = initial_condition["temperature_right_proper"].to_value(
            code_units.temperature_unit
        )
        temp_supercomoving_code = np.where(
            left, temp_left_proper_code, temp_right_proper_code
        )
        mu = np.full(grid_cells, float(initial_condition["mu"]))
    else:
        rho_comoving_code = np.asarray(
            config["_rho_comoving_code"], dtype=float
        )
        temp_supercomoving_code = np.asarray(
            config["_temp_supercomoving_code"], dtype=float
        )
        mu = np.asarray(config.get("_mu_dimensionless", np.ones(grid_cells)), dtype=float)
    vel_supercomoving_code = np.asarray(
        config.get("_vel_supercomoving_code", np.zeros(grid_cells)), dtype=float
    )
    result.fluid.rho_comoving_code = rho_comoving_code
    result.fluid.vel_supercomoving_code = vel_supercomoving_code
    result.fluid.temp_supercomoving_code = temp_supercomoving_code
    result.fluid.mu = mu
    if "_specific_angular_momentum_code" in config:
        result.par.gas_angular_momentum = True
        result.par.gas_rotational_energy = True
        result.fluid.specific_angular_momentum_code = np.asarray(
            config["_specific_angular_momentum_code"], dtype=float
        )
    result.fluid.tau_supercomoving_code = float(
        result.par.tau_supercomoving_code[0]
    )
    result.fluid.eos = EOS(
        result.par.hydrodynamics.eos_type,
        result.par.hydrodynamics.gamma,
        code_units,
    )
    pre_supercomoving_code = result.fluid.eos.pressure(
        rho_comoving_code, temp_supercomoving_code, mu
    )
    result.fluid.pre_supercomoving_code = pre_supercomoving_code
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        x_comoving_code=result.mesh.x_comoving_code,
        boundary_comoving_code=result.mesh.boundary_comoving_code,
        width_comoving_code=result.mesh.width_comoving_code,
        area_comoving_code=result.mesh.area_comoving_code,
        volume_comoving_code=result.mesh.volume_comoving_code,
    )
    result.fluid.runtime_state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        rho_comoving_code=rho_comoving_code,
        vel_supercomoving_code=vel_supercomoving_code,
        pre_supercomoving_code=pre_supercomoving_code,
        temp_supercomoving_code=temp_supercomoving_code,
        tau_supercomoving_code=result.fluid.tau_supercomoving_code,
        mu_dimensionless=mu,
    )
    return Rsim.FromComponents(result.par, result.mesh, result.fluid, result.solver)
