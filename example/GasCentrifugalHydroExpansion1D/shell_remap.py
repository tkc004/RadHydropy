# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Conservative remapping of Lagrangian spherical shells to Eulerian cells."""

import numpy as np
from scipy.integrate import solve_ivp


def shell_quadrature(
    boundary_proper_code,
    samples_per_cell,
    rho_proper_code,
    central_mass_proper_code,
    rotation_factor,
):
    """Construct equal-mass quadrature shells for spherical cells.

    The quadrature is uniform in enclosed volume, so each source cell is
    represented by ``samples_per_cell`` shells with equal mass.  Returning
    shell masses explicitly keeps the analytic reference independent of the
    target Eulerian resolution.
    """
    boundary_proper_code = np.asarray(boundary_proper_code, dtype=float)
    samples_per_cell = int(samples_per_cell)
    if samples_per_cell < 1:
        raise ValueError("samples_per_cell must be positive")
    if boundary_proper_code.ndim != 1 or len(boundary_proper_code) < 2:  # noqa: PLR2004
        raise ValueError("boundary must contain at least one cell")
    if np.any(np.diff(boundary_proper_code) <= 0.0):
        raise ValueError("boundary must be strictly increasing")

    lower = boundary_proper_code[:-1, None]
    upper = boundary_proper_code[1:, None]
    fraction = (np.arange(samples_per_cell, dtype=float) + 0.5) / samples_per_cell
    lower_volume = lower**3
    upper_volume = upper**3
    radius_proper_code = (lower_volume + fraction[None, :] * (upper_volume - lower_volume)) ** (
        1.0 / 3.0
    )
    cell_volume = 4.0 * np.pi / 3.0 * (upper_volume[:, 0] - lower_volume[:, 0])
    shell_mass = np.broadcast_to(
        float(rho_proper_code) * cell_volume[:, None] / samples_per_cell,
        radius_proper_code.shape,
    ).copy()
    shell_specific_angular_momentum_code = float(rotation_factor) * np.sqrt(
        float(central_mass_proper_code) * radius_proper_code,
    )
    volume_edges = np.linspace(0.0, 1.0, samples_per_cell + 1)
    cell_edges = (
        lower_volume[:, 0, None]
        + volume_edges[None, :] * (upper_volume[:, 0, None] - lower_volume[:, 0, None])
    ) ** (1.0 / 3.0)
    shell_edge = np.concatenate((cell_edges[:, :-1].ravel(), cell_edges[-1:, -1]))
    return (
        radius_proper_code.ravel(),
        shell_mass.ravel(),
        shell_specific_angular_momentum_code.ravel(),
        shell_edge,
    )


def conservative_shell_remap(
    shell_radius_proper_code,
    shell_vel_proper_code,
    shell_specific_angular_momentum_code,
    shell_mass_proper_code,
    target_boundary_proper_code,
    shell_specific_energy_proper_code=None,
    shell_edge_proper_code=None,
):
    """Remap shell mass and mass-weighted fields into Eulerian cells.

    Each sorted shell represents a radial slab bounded by midpoint locations
    between neighboring shell centers.  Conserved shell quantities are
    distributed by radial overlap with the target cells; no primitive field is
    interpolated.  The method is conservative for all shells whose slabs
    overlap the target domain.
    """
    (
        _radius_proper_code,
        vel_proper_code,
        specific_angular_momentum_code,
        mass_proper_code,
        target_boundary_proper_code,
        specific_energy_proper_code,
        shell_edge_proper_code,
    ) = _normalize_remap_inputs(
        shell_radius_proper_code,
        shell_vel_proper_code,
        shell_specific_angular_momentum_code,
        shell_mass_proper_code,
        target_boundary_proper_code,
        shell_specific_energy_proper_code,
        shell_edge_proper_code,
    )

    cell_count = len(target_boundary_proper_code) - 1
    remapped_mass = np.zeros(cell_count, dtype=float)
    remapped_momentum = np.zeros(cell_count, dtype=float)
    remapped_angular = np.zeros(cell_count, dtype=float)
    remapped_energy = np.zeros(cell_count, dtype=float)
    _deposit_shell_overlaps(
        shell_edge_proper_code,
        mass_proper_code,
        vel_proper_code,
        specific_angular_momentum_code,
        specific_energy_proper_code,
        target_boundary_proper_code,
        remapped_mass,
        remapped_momentum,
        remapped_angular,
        remapped_energy,
    )

    valid = remapped_mass > 0.0
    remapped_velocity = np.zeros(cell_count, dtype=float)
    remapped_j = np.zeros(cell_count, dtype=float)
    remapped_velocity[valid] = remapped_momentum[valid] / remapped_mass[valid]
    remapped_j[valid] = remapped_angular[valid] / remapped_mass[valid]
    volume_proper_code = (
        4.0
        * np.pi
        / 3.0
        * (target_boundary_proper_code[1:] ** 3 - target_boundary_proper_code[:-1] ** 3)
    )
    return {
        "mass_proper_code": remapped_mass,
        "momentum_proper_code": remapped_momentum,
        "angular_momentum_proper_code": remapped_angular,
        "energy_proper_code": remapped_energy,
        "vel_proper_code": remapped_velocity,
        "specific_angular_momentum_proper_code": remapped_j,
        "rho_proper_code": np.divide(
            remapped_mass,
            volume_proper_code,
            out=np.zeros_like(remapped_mass),
            where=volume_proper_code > 0.0,
        ),
        "volume_proper_code": volume_proper_code,
    }


def _normalize_remap_inputs(
    shell_radius,
    shell_velocity,
    shell_angular_momentum,
    shell_mass,
    target_boundary,
    shell_energy,
    shell_edges,
):
    radius = np.asarray(shell_radius, dtype=float)
    velocity = np.asarray(shell_velocity, dtype=float)
    angular = np.asarray(shell_angular_momentum, dtype=float)
    mass = np.asarray(shell_mass, dtype=float)
    target = np.asarray(target_boundary, dtype=float)
    _validate_remap_arrays(radius, velocity, angular, mass, target)
    radius, velocity, angular, mass, shell_edges, order = _normalize_shell_order(
        radius,
        velocity,
        angular,
        mass,
        shell_edges,
    )
    energy = (
        0.5 * velocity**2 if shell_energy is None else np.asarray(shell_energy, dtype=float)[order]
    )
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(angular)):
        raise ValueError("shell velocity and angular momentum must be finite")
    if not np.all(np.isfinite(energy)):
        raise ValueError("shell specific energy must be finite")
    if shell_edges is None:
        shell_edges = np.empty(len(radius) + 1, dtype=float)
        if len(radius) == 1:
            half_width = max(0.5 * radius[0], np.finfo(float).tiny)
            shell_edges[:] = (radius[0] - half_width, radius[0] + half_width)
        else:
            shell_edges[1:-1] = 0.5 * (radius[:-1] + radius[1:])
            shell_edges[0] = max(0.0, radius[0] - 0.5 * (radius[1] - radius[0]))
            shell_edges[-1] = radius[-1] + 0.5 * (radius[-1] - radius[-2])
    return radius, velocity, angular, mass, target, energy, shell_edges


def _validate_remap_arrays(
    radius_proper_code,
    velocity_proper_code,
    angular_momentum_code,
    mass_proper_code,
    target_boundary_proper_code,
):
    if not (
        radius_proper_code.ndim
        == velocity_proper_code.ndim
        == angular_momentum_code.ndim
        == mass_proper_code.ndim
        == 1
    ):
        raise ValueError("shell fields must be one-dimensional")
    if not (
        len(radius_proper_code)
        == len(velocity_proper_code)
        == len(angular_momentum_code)
        == len(mass_proper_code)
    ):
        raise ValueError("shell fields must have equal lengths")
    if len(radius_proper_code) == 0 or len(target_boundary_proper_code) < 2:  # noqa: PLR2004
        raise ValueError("shell and target grids must be non-empty")
    if not np.all(np.isfinite(radius_proper_code)) or np.any(radius_proper_code <= 0.0):
        raise ValueError("shell radii must be finite and positive")
    if not np.all(np.isfinite(mass_proper_code)) or np.any(mass_proper_code < 0.0):
        raise ValueError("shell masses must be finite and non-negative")
    if np.any(np.diff(target_boundary_proper_code) <= 0.0):
        raise ValueError("target boundary must be strictly increasing")


def _normalize_shell_order(
    radius_proper_code,
    velocity_proper_code,
    angular_momentum_code,
    mass_proper_code,
    shell_edges,
):
    if shell_edges is None:
        order = np.argsort(radius_proper_code)
        return (
            *[
                values[order]
                for values in (
                    radius_proper_code,
                    velocity_proper_code,
                    angular_momentum_code,
                    mass_proper_code,
                )
            ],
            None,
            order,
        )
    order = slice(None)
    shell_edges = np.asarray(shell_edges, dtype=float)
    if len(shell_edges) != len(radius_proper_code) + 1:
        raise ValueError("shell_edge must have one more entry than shells")
    if np.any(np.diff(shell_edges) <= 0.0):
        raise ValueError("shell edges must be strictly increasing")
    if np.any(radius_proper_code <= shell_edges[:-1]) or np.any(
        radius_proper_code >= shell_edges[1:],
    ):
        raise ValueError("shell centers must lie inside shell edges")
    return (
        radius_proper_code,
        velocity_proper_code,
        angular_momentum_code,
        mass_proper_code,
        shell_edges,
        order,
    )


def _deposit_shell_overlaps(
    shell_edge_proper_code,
    mass_proper_code,
    vel_proper_code,
    specific_angular_momentum_code,
    specific_energy_proper_code,
    target_boundary_proper_code,
    remapped_mass,
    remapped_momentum,
    remapped_angular,
    remapped_energy,
):
    cell_count = len(target_boundary_proper_code) - 1
    for index in range(len(mass_proper_code)):
        slab_width = shell_edge_proper_code[index + 1] - shell_edge_proper_code[index]
        if slab_width <= 0.0 or mass_proper_code[index] == 0.0:
            continue
        left = max(shell_edge_proper_code[index], target_boundary_proper_code[0])
        right = min(shell_edge_proper_code[index + 1], target_boundary_proper_code[-1])
        if right <= left:
            continue
        first = max(0, np.searchsorted(target_boundary_proper_code, left, side="right") - 1)
        last = min(cell_count - 1, np.searchsorted(target_boundary_proper_code, right, side="left"))
        for cell in range(first, last + 1):
            overlap = max(
                0.0,
                min(right, target_boundary_proper_code[cell + 1])
                - max(left, target_boundary_proper_code[cell]),
            )
            deposited = mass_proper_code[index] * overlap / slab_width
            remapped_mass[cell] += deposited
            remapped_momentum[cell] += deposited * vel_proper_code[index]
            remapped_angular[cell] += deposited * specific_angular_momentum_code[index]
            remapped_energy[cell] += deposited * specific_energy_proper_code[index]


def centrifugal_shell_reference(
    source_boundary_proper_code,
    target_boundary_proper_code,
    time_final_proper_code,
    rho_proper_code,
    central_mass_proper_code,
    rotation_factor,
    samples_per_cell=32,
):
    """Integrate a pressureless shell ensemble and conservatively remap it."""
    (
        shell_radius_proper_code,
        shell_mass_proper_code,
        shell_specific_angular_momentum_code,
        shell_edge_proper_code,
    ) = shell_quadrature(
        source_boundary_proper_code,
        samples_per_cell,
        rho_proper_code,
        central_mass_proper_code,
        rotation_factor,
    )

    def rhs(_time, state, specific_j):
        radius_proper_code, vel_proper_code = state
        radius_proper_code = max(radius_proper_code, np.finfo(float).tiny)
        return (
            vel_proper_code,
            -float(central_mass_proper_code) / radius_proper_code**2
            + specific_j**2 / radius_proper_code**3,
        )

    shell_vel_proper_code = np.empty_like(shell_radius_proper_code)
    for index, (radius_proper_code, specific_j) in enumerate(
        zip(shell_radius_proper_code, shell_specific_angular_momentum_code, strict=False),
    ):
        solution = solve_ivp(
            lambda time_proper_code, state: rhs(
                time_proper_code,
                state,
                specific_j,
            ),
            (0.0, float(time_final_proper_code)),
            (radius_proper_code, 0.0),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        shell_vel_proper_code[index] = solution.y[1, -1]
        shell_radius_proper_code[index] = solution.y[0, -1]
    for index, radius_proper_code in enumerate(shell_edge_proper_code):
        specific_j = float(rotation_factor) * np.sqrt(
            float(central_mass_proper_code) * radius_proper_code,
        )
        solution = solve_ivp(
            lambda time_proper_code, state: rhs(
                time_proper_code,
                state,
                specific_j,
            ),
            (0.0, float(time_final_proper_code)),
            (radius_proper_code, 0.0),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        shell_edge_proper_code[index] = solution.y[0, -1]
    shell_specific_energy_proper_code = (
        0.5 * shell_vel_proper_code**2
        + 0.5 * shell_specific_angular_momentum_code**2 / shell_radius_proper_code**2
        - float(central_mass_proper_code) / shell_radius_proper_code
    )
    return conservative_shell_remap(
        shell_radius_proper_code,
        shell_vel_proper_code,
        shell_specific_angular_momentum_code,
        shell_mass_proper_code,
        target_boundary_proper_code,
        shell_specific_energy_proper_code=shell_specific_energy_proper_code,
        shell_edge_proper_code=shell_edge_proper_code,
    )
