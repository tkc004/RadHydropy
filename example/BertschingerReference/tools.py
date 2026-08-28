"""Reference helpers for the collisionless Bertschinger epsilon=1 problem."""

import h5py
import numpy as np

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def load_units(runparams):
    return CodeUnits.from_mapping(runparams['CodeUnits'])


def make_scale_free_shells(icparams, units, cosmology):
    """Create the EdS epsilon=1 scale-free radial shell perturbation.

    For epsilon=1, Delta M/M is proportional to M^{-1}, hence Delta M is
    constant. It is represented as a central fixed excess mass while the
    shells sample the homogeneous background. This is the standard cold,
    radial secondary-infall construction before shell crossing.
    """
    number = int(icparams['number_of_shells'])
    qmin = float(icparams['inner_radius'])
    qmax = float(icparams['outer_radius'])
    boundaries = np.linspace(qmin**3, qmax**3, number + 1)**(1.0 / 3.0)
    radius = 0.5 * (boundaries[:-1] + boundaries[1:])
    volume = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    cosmic_time = float(icparams['initial_cosmic_time'])
    a = float(cosmology.scale_factor(cosmic_time))
    hubble = float(cosmology.hubble(cosmic_time))
    rho_comoving = float(cosmology.background_density(cosmic_time)) * a**3
    mass = rho_comoving * volume
    perturbation_amplitude = float(icparams['perturbation_amplitude'])
    delta_mass = 4.0 * np.pi / 3.0 * rho_comoving * perturbation_amplitude
    delta = perturbation_amplitude / radius**3
    velocity = -a**2 * hubble * delta * radius / 3.0
    shells = DarkMatterShells(
        radius=radius,
        velocity=velocity,
        mass=mass,
        shell_id=np.arange(number),
        fixed_enclosed_mass=delta_mass,
        softening=float(icparams['softening']),
        code_units=units,
    )
    return shells, delta_mass


def physical_velocity(shells, cosmic_time, cosmology):
    a = float(cosmology.scale_factor(cosmic_time))
    hubble = float(cosmology.hubble(cosmic_time))
    return hubble * a * shells.radius + shells.velocity / a


def turnaround_radius(shells, cosmic_time, cosmology):
    """Interpolate the physical radius where the radial velocity vanishes."""
    proper_radius = float(cosmology.scale_factor(cosmic_time)) * shells.radius
    velocity = physical_velocity(shells, cosmic_time, cosmology)
    # The sorted shell array is ordered from the collapsed/infalling region to
    # the expanding background, so the first turnaround interface is usually
    # a negative-to-positive velocity transition after shell crossing.
    crossing = np.flatnonzero(velocity[:-1] * velocity[1:] <= 0.0)
    if crossing.size == 0:
        raise RuntimeError('no turnaround shell in the requested output')
    i = int(crossing[0])
    fraction = velocity[i] / (velocity[i] - velocity[i + 1])
    return float(proper_radius[i] + fraction * (proper_radius[i + 1] - proper_radius[i]))


def similarity_profiles(shells, cosmic_time, cosmology, bins=256):
    """Deposit shell mass into dimensionless Bertschinger profiles."""
    rta = turnaround_radius(shells, cosmic_time, cosmology)
    a = float(cosmology.scale_factor(cosmic_time))
    rho_background = float(cosmology.background_density(cosmic_time))
    proper_radius = a * shells.radius
    velocity = physical_velocity(shells, cosmic_time, cosmology)
    lam_edges = np.geomspace(max(proper_radius.min() / rta, 1.0e-5),
                             proper_radius.max() / rta, bins + 1)
    lam = np.sqrt(lam_edges[:-1] * lam_edges[1:])
    shell_index = np.clip(np.searchsorted(lam_edges, proper_radius / rta) - 1, 0, bins - 1)
    shell_mass = np.bincount(shell_index, weights=shells.mass, minlength=bins)
    shell_volume = 4.0 * np.pi / 3.0 * rta**3 * np.diff(lam_edges**3)
    density_contrast = shell_mass / shell_volume / rho_background
    velocity_scaled = np.full(bins, np.nan)
    for index in range(bins):
        selected = shell_index == index
        if np.any(selected):
            velocity_scaled[index] = np.average(
                velocity[selected] / (rta / cosmic_time),
                weights=shells.mass[selected],
            )
    cumulative = np.cumsum(shell_mass)
    mass_scaled = cumulative / ((4.0 * np.pi / 3.0) * rho_background * rta**3)
    return {
        'lambda': lam,
        'density': density_contrast,
        'velocity': velocity_scaled,
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
