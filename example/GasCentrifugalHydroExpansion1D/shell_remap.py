"""Conservative remapping of Lagrangian spherical shells to Eulerian cells."""

import numpy as np
from scipy.integrate import solve_ivp


def shell_quadrature(boundary, samples_per_cell, density, central_mass,
                     rotation_factor):
    """Construct equal-mass quadrature shells for spherical cells.

    The quadrature is uniform in enclosed volume, so each source cell is
    represented by ``samples_per_cell`` shells with equal mass.  Returning
    shell masses explicitly keeps the analytic reference independent of the
    target Eulerian resolution.
    """
    boundary = np.asarray(boundary, dtype=float)
    samples_per_cell = int(samples_per_cell)
    if samples_per_cell < 1:
        raise ValueError('samples_per_cell must be positive')
    if boundary.ndim != 1 or len(boundary) < 2:
        raise ValueError('boundary must contain at least one cell')
    if np.any(np.diff(boundary) <= 0.0):
        raise ValueError('boundary must be strictly increasing')

    lower = boundary[:-1, None]
    upper = boundary[1:, None]
    fraction = (np.arange(samples_per_cell, dtype=float) + 0.5) / samples_per_cell
    lower_volume = lower**3
    upper_volume = upper**3
    radius = (lower_volume + fraction[None, :] * (upper_volume - lower_volume)) ** (1.0 / 3.0)
    cell_volume = 4.0 * np.pi / 3.0 * (upper_volume[:, 0] - lower_volume[:, 0])
    shell_mass = np.broadcast_to(
        float(density) * cell_volume[:, None] / samples_per_cell,
        radius.shape,
    ).copy()
    shell_j = float(rotation_factor) * np.sqrt(float(central_mass) * radius)
    volume_edges = np.linspace(0.0, 1.0, samples_per_cell + 1)
    cell_edges = (
        lower_volume[:, 0, None]
        + volume_edges[None, :] * (upper_volume[:, 0, None] - lower_volume[:, 0, None])
    ) ** (1.0 / 3.0)
    shell_edge = np.concatenate((cell_edges[:, :-1].ravel(), cell_edges[-1:, -1]))
    return radius.ravel(), shell_mass.ravel(), shell_j.ravel(), shell_edge


def conservative_shell_remap(
    shell_radius,
    shell_velocity,
    shell_j,
    shell_mass,
    target_boundary,
    shell_specific_energy=None,
    shell_edge=None,
):
    """Remap shell mass and mass-weighted fields into Eulerian cells.

    Each sorted shell represents a radial slab bounded by midpoint locations
    between neighboring shell centers.  Conserved shell quantities are
    distributed by radial overlap with the target cells; no primitive field is
    interpolated.  The method is conservative for all shells whose slabs
    overlap the target domain.
    """
    radius = np.asarray(shell_radius, dtype=float)
    velocity = np.asarray(shell_velocity, dtype=float)
    specific_j = np.asarray(shell_j, dtype=float)
    mass = np.asarray(shell_mass, dtype=float)
    target_boundary = np.asarray(target_boundary, dtype=float)
    if not (radius.ndim == velocity.ndim == specific_j.ndim == mass.ndim == 1):
        raise ValueError('shell fields must be one-dimensional')
    if not (len(radius) == len(velocity) == len(specific_j) == len(mass)):
        raise ValueError('shell fields must have equal lengths')
    if len(radius) == 0 or len(target_boundary) < 2:
        raise ValueError('shell and target grids must be non-empty')
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError('shell radii must be finite and positive')
    if not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError('shell masses must be finite and non-negative')
    if np.any(np.diff(target_boundary) <= 0.0):
        raise ValueError('target boundary must be strictly increasing')

    if shell_edge is None:
        order = np.argsort(radius)
        radius = radius[order]
        velocity = velocity[order]
        specific_j = specific_j[order]
        mass = mass[order]
    else:
        order = slice(None)
        shell_edge = np.asarray(shell_edge, dtype=float)
        if len(shell_edge) != len(radius) + 1:
            raise ValueError('shell_edge must have one more entry than shells')
        if np.any(np.diff(shell_edge) <= 0.0):
            raise ValueError('shell edges must be strictly increasing')
        if np.any(radius <= shell_edge[:-1]) or np.any(radius >= shell_edge[1:]):
            raise ValueError('shell centers must lie inside shell edges')
    if shell_specific_energy is None:
        specific_energy = 0.5 * velocity**2
    else:
        specific_energy = np.asarray(shell_specific_energy, dtype=float)[order]
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(specific_j)):
        raise ValueError('shell velocity and angular momentum must be finite')
    if not np.all(np.isfinite(specific_energy)):
        raise ValueError('shell specific energy must be finite')

    if shell_edge is None:
        shell_edge = np.empty(len(radius) + 1, dtype=float)
        if len(radius) == 1:
            width = max(0.5 * radius[0], np.finfo(float).tiny)
            shell_edge[:] = (radius[0] - width, radius[0] + width)
        else:
            shell_edge[1:-1] = 0.5 * (radius[:-1] + radius[1:])
            shell_edge[0] = max(0.0, radius[0] - 0.5 * (radius[1] - radius[0]))
            shell_edge[-1] = radius[-1] + 0.5 * (radius[-1] - radius[-2])

    cell_count = len(target_boundary) - 1
    remapped_mass = np.zeros(cell_count, dtype=float)
    remapped_momentum = np.zeros(cell_count, dtype=float)
    remapped_angular = np.zeros(cell_count, dtype=float)
    remapped_energy = np.zeros(cell_count, dtype=float)
    for index in range(len(radius)):
        slab_width = shell_edge[index + 1] - shell_edge[index]
        if slab_width <= 0.0 or mass[index] == 0.0:
            continue
        left = max(shell_edge[index], target_boundary[0])
        right = min(shell_edge[index + 1], target_boundary[-1])
        if right <= left:
            continue
        first = max(0, np.searchsorted(target_boundary, left, side='right') - 1)
        last = min(cell_count - 1, np.searchsorted(target_boundary, right, side='left'))
        for cell in range(first, last + 1):
            overlap = max(
                0.0,
                min(right, target_boundary[cell + 1])
                - max(left, target_boundary[cell]),
            )
            fraction = overlap / slab_width
            deposited = mass[index] * fraction
            remapped_mass[cell] += deposited
            remapped_momentum[cell] += deposited * velocity[index]
            remapped_angular[cell] += deposited * specific_j[index]
            remapped_energy[cell] += deposited * specific_energy[index]

    valid = remapped_mass > 0.0
    remapped_velocity = np.zeros(cell_count, dtype=float)
    remapped_j = np.zeros(cell_count, dtype=float)
    remapped_velocity[valid] = remapped_momentum[valid] / remapped_mass[valid]
    remapped_j[valid] = remapped_angular[valid] / remapped_mass[valid]
    target_volume = 4.0 * np.pi / 3.0 * (
        target_boundary[1:]**3 - target_boundary[:-1]**3
    )
    return {
        'mass': remapped_mass,
        'momentum': remapped_momentum,
        'angular_momentum': remapped_angular,
        'energy': remapped_energy,
        'velocity': remapped_velocity,
        'specific_angular_momentum': remapped_j,
        'density': np.divide(
            remapped_mass, target_volume,
            out=np.zeros_like(remapped_mass), where=target_volume > 0.0,
        ),
        'volume': target_volume,
    }


def centrifugal_shell_reference(
    source_boundary,
    target_boundary,
    final_time,
    density,
    central_mass,
    rotation_factor,
    samples_per_cell=32,
):
    """Integrate a pressureless shell ensemble and conservatively remap it."""
    shell_radius, shell_mass, shell_j, shell_edge = shell_quadrature(
        source_boundary,
        samples_per_cell,
        density,
        central_mass,
        rotation_factor,
    )

    def rhs(_time, state, specific_j):
        position, velocity = state
        position = max(position, np.finfo(float).tiny)
        return (
            velocity,
            -float(central_mass) / position**2
            + specific_j**2 / position**3,
        )

    shell_velocity = np.empty_like(shell_radius)
    edge_velocity = np.empty_like(shell_edge)
    for index, (position, specific_j) in enumerate(zip(shell_radius, shell_j)):
        solution = solve_ivp(
            lambda time, state: rhs(time, state, specific_j),
            (0.0, float(final_time)),
            (position, 0.0),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        shell_velocity[index] = solution.y[1, -1]
        shell_radius[index] = solution.y[0, -1]
    for index, position in enumerate(shell_edge):
        specific_j = float(rotation_factor) * np.sqrt(float(central_mass) * position)
        solution = solve_ivp(
            lambda time, state: rhs(time, state, specific_j),
            (0.0, float(final_time)),
            (position, 0.0),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        shell_edge[index] = solution.y[0, -1]
    shell_specific_energy = (
        0.5 * shell_velocity**2
        + 0.5 * shell_j**2 / shell_radius**2
        - float(central_mass) / shell_radius
    )
    return conservative_shell_remap(
        shell_radius,
        shell_velocity,
        shell_j,
        shell_mass,
        target_boundary,
        shell_specific_energy=shell_specific_energy,
        shell_edge=shell_edge,
    )
