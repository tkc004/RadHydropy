"""Numerical solver subsystem helpers."""

import numpy as np

from radhydropy.runtime_fields import select_fluid_primitive_arrays, select_mesh_geometry_arrays


def _cfl_density_floor(par):
    return max(
        0.0,
        float(
            np.asarray(
                getattr(par, "cfl_density_floor", 0.0),
                dtype=float,
            ),
        ),
    )


def _dual_energy_enabled(par):
    return bool(getattr(par, "dual_energy", False))


def _rotational_energy_enabled(par):
    return bool(getattr(par, "gas_rotational_energy", False))


def _gravity_potential_energy_enabled(par):
    return bool(getattr(par, "gravity_potential_energy", False))


def _gravity_potential(solver, mesh, par):
    if not solver._gravity_potential_energy_enabled(par):
        return None
    gravity = solver._gravity_model(par)
    if gravity is None or not hasattr(gravity, "potential_on"):
        raise ValueError(
            "gravity_potential_energy requires a gravity model with potential_on",
        )
    geometry = mesh.geometry_state
    coordinate_runtime_code = select_mesh_geometry_arrays(geometry, par)[0]
    return np.asarray(gravity.potential_on(coordinate_runtime_code), dtype=float)


def _gravity_potential_faces(solver, mesh, par):
    if not solver._gravity_potential_energy_enabled(par):
        return None
    gravity = solver._gravity_model(par)
    if gravity is None or not hasattr(gravity, "potential_on"):
        raise ValueError(
            "gravity_potential_energy requires a gravity model with potential_on",
        )
    geometry = mesh.geometry_state
    boundary_runtime_code = select_mesh_geometry_arrays(geometry, par)[1]
    return np.asarray(gravity.potential_on(boundary_runtime_code[:-1]), dtype=float)


def _rotational_energy_density(solver, mesh, fluid, par):
    """Return opt-in rotational kinetic-energy density."""
    runtime_state = fluid.runtime_state
    rho_runtime_code = select_fluid_primitive_arrays(runtime_state, par)[0]
    rho = np.asarray(rho_runtime_code, dtype=float)
    result = np.zeros_like(rho)
    if not solver._rotational_energy_enabled(par):
        return result
    if not getattr(par, "gas_angular_momentum", False):
        raise ValueError(
            "gas_rotational_energy requires gas_angular_momentum: true",
        )
    if getattr(mesh, "coordsys", None) != "spherical":
        raise ValueError("gas_rotational_energy requires a spherical mesh")
    geometry = mesh.geometry_state
    radius_runtime_code = select_mesh_geometry_arrays(geometry, par)[0]
    radius = np.asarray(radius_runtime_code, dtype=float)
    specific = np.asarray(fluid.specific_angular_momentum_code, dtype=float)
    valid = (
        np.isfinite(rho)
        & (rho > 0.0)
        & np.isfinite(specific)
        & np.isfinite(radius)
        & (radius > 0.0)
    )
    result[valid] = 0.5 * rho[valid] * specific[valid] ** 2 / radius[valid] ** 2
    return result


def _rotational_energy_from_conserved(solver, mesh, fluid, par):
    """Return opt-in rotational kinetic energy from conserved J and M."""
    result = np.zeros_like(np.asarray(fluid.Mass_code, dtype=float))
    if not solver._rotational_energy_enabled(par):
        return result
    if not hasattr(fluid, "AngularMomentum_code"):
        return result
    mass = np.asarray(fluid.Mass_code, dtype=float)
    angular_momentum = np.asarray(fluid.AngularMomentum_code, dtype=float)
    geometry = mesh.geometry_state
    radius_runtime_code = select_mesh_geometry_arrays(geometry, par)[0]
    radius = np.abs(np.asarray(radius_runtime_code, dtype=float))
    valid = (
        np.isfinite(mass)
        & (mass > 0.0)
        & np.isfinite(angular_momentum)
        & np.isfinite(radius)
        & (radius > 0.0)
    )
    result[valid] = 0.5 * angular_momentum[valid] ** 2 / (mass[valid] * radius[valid] ** 2)
    return result


def _dual_energy_eta(par, name):
    return max(0.0, float(getattr(par, name)))
