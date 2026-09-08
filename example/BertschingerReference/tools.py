"""Reference helpers for the collisionless Bertschinger epsilon=1 problem."""

import h5py
import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits
from example import example_utils as eu


def code_units_from_config(config):
    return CodeUnits.from_mapping(config['par']['units']['CodeUnits'])


def load_reference_config(filename):
    """Load the complete nested Bertschinger example configuration."""
    return eu.load_nested_example_config(filename)


def make_scale_free_shells(config):
    """Create the EdS epsilon=1 scale-free radial shell perturbation.

    For epsilon=1, Delta M/M is proportional to M^{-1}, hence Delta M is
    constant. It is represented as a central fixed excess mass while the
    shells sample the homogeneous background. This is the standard cold,
    radial secondary-infall construction before shell crossing.
    """
    initial_condition = config['initial_condition']
    code_unit_system = config['_code_units']
    cosmology = config['_cosmology']
    number = int(initial_condition['number_of_shells'])
    qmin = float(initial_condition['radius_inner_dimensionless'])
    qmax = float(initial_condition['radius_outer_dimensionless'])
    boundaries = np.linspace(qmin**3, qmax**3, number + 1)**(1.0 / 3.0)
    radius_comoving_code = 0.5 * (boundaries[:-1] + boundaries[1:])
    volume_comoving_code = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    time_cosmic_code = float(initial_condition['time_cosmic'])
    scale_factor_dimensionless = float(cosmology.scale_factor(time_cosmic_code))
    hubble_code = float(cosmology.hubble(time_cosmic_code))
    rho_comoving_code = float(cosmology.background_density(time_cosmic_code)) * scale_factor_dimensionless**3
    mass_comoving_code = rho_comoving_code * volume_comoving_code
    perturbation_amplitude = float(initial_condition['perturbation_amplitude'])
    delta_mass = 4.0 * np.pi / 3.0 * rho_comoving * perturbation_amplitude
    delta = perturbation_amplitude / radius_comoving_code**3
    vel_supercomoving_code = -scale_factor_dimensionless**2 * hubble_code * delta * radius_comoving_code / 3.0
    shells = DarkMatterShells(
        radius=radius_comoving_code,
        velocity=vel_supercomoving_code,
        mass=mass_comoving_code,
        shell_id=np.arange(number),
        fixed_enclosed_mass=delta_mass,
        softening=float(initial_condition['softening']),
        code_units=code_unit_system,
    )
    return shells, delta_mass


def physical_velocity(shells, cosmic_time, cosmology):
    a = float(cosmology.scale_factor(cosmic_time))
    hubble = float(cosmology.hubble(cosmic_time))
    return hubble * a * shells.radius + shells.velocity / a


def turnaround_radius(shells, cosmic_time, cosmology):
    """Interpolate the physical radius where the radial velocity vanishes."""
    radius_proper_code = float(cosmology.scale_factor(cosmic_time)) * shells.radius
    vel_peculiar_proper_code = physical_velocity(shells, cosmic_time, cosmology)
    # The sorted shell array is ordered from the collapsed/infalling region to
    # the expanding background, so the first turnaround interface is usually
    # a negative-to-positive velocity transition after shell crossing.
    crossing = np.flatnonzero(vel_peculiar_proper_code[:-1] * vel_peculiar_proper_code[1:] <= 0.0)
    if crossing.size == 0:
        raise RuntimeError('no turnaround shell in the requested output')
    i = int(crossing[0])
    fraction = vel_peculiar_proper_code[i] / (vel_peculiar_proper_code[i] - vel_peculiar_proper_code[i + 1])
    return float(radius_proper_code[i] + fraction * (radius_proper_code[i + 1] - radius_proper_code[i]))


def similarity_profiles(shells, cosmic_time, cosmology, bins=256):
    """Deposit shell mass into dimensionless Bertschinger profiles."""
    rta = turnaround_radius(shells, cosmic_time, cosmology)
    a = float(cosmology.scale_factor(cosmic_time))
    rho_background = float(cosmology.background_density(cosmic_time))
    radius_proper_code = a * shells.radius
    vel_peculiar_proper_code = physical_velocity(shells, cosmic_time, cosmology)
    lam_edges = np.geomspace(max(radius_proper_code.min() / rta, 1.0e-5),
                             radius_proper_code.max() / rta, bins + 1)
    lam = np.sqrt(lam_edges[:-1] * lam_edges[1:])
    shell_index = np.clip(np.searchsorted(lam_edges, radius_proper_code / rta) - 1, 0, bins - 1)
    shell_mass = np.bincount(shell_index, weights=shells.mass, minlength=bins)
    shell_volume = 4.0 * np.pi / 3.0 * rta**3 * np.diff(lam_edges**3)
    density_contrast = shell_mass / shell_volume / rho_background
    velocity_scaled = np.full(bins, np.nan)
    for index in range(bins):
        selected = shell_index == index
        if np.any(selected):
            velocity_scaled[index] = np.average(
                vel_peculiar_proper_code[selected] / (rta / cosmic_time),
                weights=shells.mass[selected],
            )
    cumulative = np.cumsum(shell_mass)
    mass_scaled = cumulative / ((4.0 * np.pi / 3.0) * rho_background * rta**3)
    return {
        'lambda': lam,
        'rho_proper': density_contrast,
        'vel_proper': velocity_scaled,
        'mass': mass_scaled,
        'turnaround_radius': rta,
    }


def write_reference(filename, profiles, metadata):
    with h5py.File(filename, 'w') as handle:
        header = handle.create_group('Header')
        for key, value in metadata.items():
            header.attrs[key] = value
        for key, value in profiles.items():
            if key != 'turnaround_radius':
                handle.create_dataset(key, data=np.asarray(value))
        header.attrs['TurnaroundRadius'] = float(profiles['turnaround_radius'])
