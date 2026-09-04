"""Initial conditions and diagnostics for the PIE radiative shock tube."""

from types import SimpleNamespace

import numpy as np
import unyt


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
BOLTZMANN_ERG_cgs_K = unyt.kb.to_value(unyt.erg / unyt.K)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


class Simwrap:
    """Build two uniform streams moving toward the central discontinuity."""

    def __init__(self, config, code_units, hydrogen_mass_fraction):
        icparams = config['initial_condition']
        par = config['par']
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordinate_system'],
            current_time=icparams['current_time'],
            box_size=icparams['box_size'],
        )
        grid_cells = int(par['mesh']['grid_cells'])
        boxsize = icparams['box_size'] * np.ones(1)
        self.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=par['mesh']['ghost_cells'])
        self.par.time = icparams['current_time'] * np.ones(1)
        self.mesh.boundary = np.linspace(
            0.0 * boxsize[0], boxsize[0], grid_cells + 1,
        )
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])
        velocity = icparams['collision_velocity']
        self.fluid.vel_code = np.where(
            coordinate < 0.5 * boxsize[0], velocity, -velocity
        )
        rho = icparams['hydrogen_density'] * unyt.mp / hydrogen_mass_fraction
        self.fluid.rho_code = np.ones(grid_cells) * rho
        self.fluid.temp_code = np.ones(grid_cells) * icparams['initial_temperature']
        self.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']


def _physical_cells(data, header):
    noghost = int(header.attrs.get('GhostCells', 0))
    nogrid = int(header.attrs['GridCells'])
    return slice(noghost, noghost + nogrid)


def load_snapshot(filename):
    import h5py

    with h5py.File(filename, 'r') as handle:
        data = handle['Data']
        header = handle['Header']
        physical = _physical_cells(data, header)
        noghost = int(header.attrs.get('GhostCells', 0))
        nogrid = int(header.attrs['GridCells'])
        boundary = np.asarray(data['Boundary'][()])[noghost:noghost + nogrid + 1]
        return {
            'time_Myr': float(header.attrs['Time']) / SECONDS_PER_MYR,
            'boundary_cgs_cm': boundary,
            'density_cgs_g_cm3': np.asarray(data['Density'][()])[physical],
            'velocity_cgs_cm_s': np.asarray(data['Velocity'][()])[physical],
            'temperature_cgs_K': np.asarray(data['Temperature'][()])[physical],
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
    temperature,
    density_cgs_g_cm3,
    hydrogen_mass_fraction,
    mu,
    gamma,
    metallicity,
    redshift,
    post_velocity_cgs_cm_s,
):
    """Estimate l_cool = u_post * t_cool using the net PIE rate."""
    hydrogen_density = density_cgs_g_cm3 * hydrogen_mass_fraction / PROTON_MASS_G
    heating, cooling = table.rates(
        temperature, hydrogen_density, metallicity=metallicity, redshift=redshift
    )
    net_cooling = max(float(np.asarray(cooling) - np.asarray(heating)), 1.0e-99)
    thermal_energy = 1.5 * density_cgs_g_cm3 * BOLTZMANN_ERG_cgs_K * temperature / (
        mu * PROTON_MASS_G
    )
    cooling_time_s = thermal_energy / net_cooling
    return post_velocity_cgs_cm_s * cooling_time_s
