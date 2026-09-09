"""Conservative remapping of Lagrangian spherical shells to Eulerian cells."""

import numpy as np
from scipy.integrate import solve_ivp


def shell_quadrature(boundary_proper_code, samples_per_cell, rho_proper_code,
                     central_mass_proper_code,
                     rotation_factor):
    """Construct equal-mass quadrature shells for spherical cells.

    The quadrature is uniform in enclosed volume, so each source cell is
    represented by ``samples_per_cell`` shells with equal mass.  Returning
    shell masses explicitly keeps the analytic reference independent of the
    target Eulerian resolution.
    """
    boundary_proper_code = np.asarray(boundary_proper_code, dtype=float)
    samples_per_cell = int(samples_per_cell)
    if samples_per_cell < 1:
        raise ValueError('samples_per_cell must be positive')
    if boundary_proper_code.ndim != 1 or len(boundary_proper_code) < 2:
        raise ValueError('boundary must contain at least one cell')
    if np.any(np.diff(boundary_proper_code) <= 0.0):
        raise ValueError('boundary must be strictly increasing')

    lower = boundary_proper_code[:-1, None]
    upper = boundary_proper_code[1:, None]
    fraction = (np.arange(samples_per_cell, dtype=float) + 0.5) / samples_per_cell
    lower_volume = lower**3
    upper_volume = upper**3
    radius_proper_code = (lower_volume + fraction[None, :] * (upper_volume - lower_volume)) ** (1.0 / 3.0)
    cell_volume = 4.0 * np.pi / 3.0 * (upper_volume[:, 0] - lower_volume[:, 0])
    shell_mass = np.broadcast_to(
        float(rho_proper_code) * cell_volume[:, None] / samples_per_cell,
        radius_proper_code.shape,
    ).copy()
    shell_specific_angular_momentum_code = float(rotation_factor) * np.sqrt(
        float(central_mass_proper_code) * radius_proper_code
    )
    volume_edges = np.linspace(0.0, 1.0, samples_per_cell + 1)
    cell_edges = (
        lower_volume[:, 0, None]
        + volume_edges[None, :] * (upper_volume[:, 0, None] - lower_volume[:, 0, None])
    ) ** (1.0 / 3.0)
    shell_edge = np.concatenate((cell_edges[:, :-1].ravel(), cell_edges[-1:, -1]))
    return (
        radius_proper_code.ravel(), shell_mass.ravel(),
        shell_specific_angular_momentum_code.ravel(), shell_edge,
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
    radius_proper_code = np.asarray(shell_radius_proper_code, dtype=float)
    vel_proper_code = np.asarray(shell_vel_proper_code, dtype=float)
    specific_angular_momentum_code = np.asarray(
        shell_specific_angular_momentum_code, dtype=float
    )
    mass_proper_code = np.asarray(shell_mass_proper_code, dtype=float)
    target_boundary_proper_code = np.asarray(target_boundary_proper_code, dtype=float)
    if not (radius_proper_code.ndim == vel_proper_code.ndim ==
            specific_angular_momentum_code.ndim == mass_proper_code.ndim == 1):
        raise ValueError('shell fields must be one-dimensional')
    if not (len(radius_proper_code) == len(vel_proper_code) ==
            len(specific_angular_momentum_code) == len(mass_proper_code)):
        raise ValueError('shell fields must have equal lengths')
    if len(radius_proper_code) == 0 or len(target_boundary_proper_code) < 2:
        raise ValueError('shell and target grids must be non-empty')
    if not np.all(np.isfinite(radius_proper_code)) or np.any(radius_proper_code <= 0.0):
        raise ValueError('shell radii must be finite and positive')
    if not np.all(np.isfinite(mass_proper_code)) or np.any(mass_proper_code < 0.0):
        raise ValueError('shell masses must be finite and non-negative')
    if np.any(np.diff(target_boundary_proper_code) <= 0.0):
        raise ValueError('target boundary must be strictly increasing')

    if shell_edge_proper_code is None:
        order = np.argsort(radius_proper_code)
        radius_proper_code = radius_proper_code[order]
        vel_proper_code = vel_proper_code[order]
        specific_angular_momentum_code = specific_angular_momentum_code[order]
        mass_proper_code = mass_proper_code[order]
    else:
        order = slice(None)
        shell_edge_proper_code = np.asarray(shell_edge_proper_code, dtype=float)
        if len(shell_edge_proper_code) != len(radius_proper_code) + 1:
            raise ValueError('shell_edge must have one more entry than shells')
        if np.any(np.diff(shell_edge_proper_code) <= 0.0):
            raise ValueError('shell edges must be strictly increasing')
        if (np.any(radius_proper_code <= shell_edge_proper_code[:-1]) or
                np.any(radius_proper_code >= shell_edge_proper_code[1:])):
            raise ValueError('shell centers must lie inside shell edges')
    if shell_specific_energy_proper_code is None:
        specific_energy_proper_code = 0.5 * vel_proper_code**2
    else:
        specific_energy_proper_code = np.asarray(
            shell_specific_energy_proper_code, dtype=float
        )[order]
    if (not np.all(np.isfinite(vel_proper_code)) or
            not np.all(np.isfinite(specific_angular_momentum_code))):
        raise ValueError('shell velocity and angular momentum must be finite')
    if not np.all(np.isfinite(specific_energy_proper_code)):
        raise ValueError('shell specific energy must be finite')

    if shell_edge_proper_code is None:
        shell_edge_proper_code = np.empty(len(radius_proper_code) + 1, dtype=float)
        if len(radius_proper_code) == 1:
            width = max(0.5 * radius_proper_code[0], np.finfo(float).tiny)
            shell_edge_proper_code[:] = (radius_proper_code[0] - width, radius_proper_code[0] + width)
        else:
            shell_edge_proper_code[1:-1] = 0.5 * (radius_proper_code[:-1] + radius_proper_code[1:])
            shell_edge_proper_code[0] = max(0.0, radius_proper_code[0] - 0.5 * (radius_proper_code[1] - radius_proper_code[0]))
            shell_edge_proper_code[-1] = radius_proper_code[-1] + 0.5 * (radius_proper_code[-1] - radius_proper_code[-2])

    cell_count = len(target_boundary_proper_code) - 1
    remapped_mass = np.zeros(cell_count, dtype=float)
    remapped_momentum = np.zeros(cell_count, dtype=float)
    remapped_angular = np.zeros(cell_count, dtype=float)
    remapped_energy = np.zeros(cell_count, dtype=float)
    for index in range(len(radius_proper_code)):
        slab_width = shell_edge_proper_code[index + 1] - shell_edge_proper_code[index]
        if slab_width <= 0.0 or mass_proper_code[index] == 0.0:
            continue
        left = max(shell_edge_proper_code[index], target_boundary_proper_code[0])
        right = min(shell_edge_proper_code[index + 1], target_boundary_proper_code[-1])
        if right <= left:
            continue
        first = max(0, np.searchsorted(target_boundary_proper_code, left, side='right') - 1)
        last = min(cell_count - 1, np.searchsorted(target_boundary_proper_code, right, side='left'))
        for cell in range(first, last + 1):
            overlap = max(
                0.0,
                min(right, target_boundary_proper_code[cell + 1])
                - max(left, target_boundary_proper_code[cell]),
            )
            fraction = overlap / slab_width
            deposited = mass_proper_code[index] * fraction
            remapped_mass[cell] += deposited
            remapped_momentum[cell] += deposited * vel_proper_code[index]
            remapped_angular[cell] += deposited * specific_angular_momentum_code[index]
            remapped_energy[cell] += deposited * specific_energy_proper_code[index]

    valid = remapped_mass > 0.0
    remapped_velocity = np.zeros(cell_count, dtype=float)
    remapped_j = np.zeros(cell_count, dtype=float)
    remapped_velocity[valid] = remapped_momentum[valid] / remapped_mass[valid]
    remapped_j[valid] = remapped_angular[valid] / remapped_mass[valid]
    volume_proper_code = 4.0 * np.pi / 3.0 * (
        target_boundary_proper_code[1:]**3 - target_boundary_proper_code[:-1]**3
    )
    return {
        'mass_proper_code': remapped_mass,
        'momentum_proper_code': remapped_momentum,
        'angular_momentum_proper_code': remapped_angular,
        'energy_proper_code': remapped_energy,
        'vel_proper_code': remapped_velocity,
        'specific_angular_momentum_proper_code': remapped_j,
        'rho_proper_code': np.divide(
            remapped_mass, volume_proper_code,
            out=np.zeros_like(remapped_mass), where=volume_proper_code > 0.0,
        ),
        'volume_proper_code': volume_proper_code,
    }


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
    shell_radius_proper_code, shell_mass_proper_code, shell_specific_angular_momentum_code, shell_edge_proper_code = shell_quadrature(
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
        zip(shell_radius_proper_code, shell_specific_angular_momentum_code)
    ):
        solution = solve_ivp(
            lambda time_proper_code, state: rhs(
                time_proper_code, state, specific_j
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
            float(central_mass_proper_code) * radius_proper_code
        )
        solution = solve_ivp(
            lambda time_proper_code, state: rhs(
                time_proper_code, state, specific_j
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
