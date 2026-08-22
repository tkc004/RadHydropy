"""Initial conditions and diagnostics for a gravity-free spherical PIE shock."""

from types import SimpleNamespace

import h5py
import numpy as np
import unyt


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


class Simwrap:
    """Build opposing radial streams that collide in a spherical shell."""

    def __init__(self, icparams, code_units, hydrogen_mass_fraction):
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.par.CodeUnits = code_units
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(icparams['nogrid'])
        self.par.coordsys = 'spherical'
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)

        self.mesh.boundary = np.linspace(
            icparams['rmin'], icparams['rmax'], self.par.nogrid + 1
        )
        rho = (
            icparams['hydrogen_density'] * unyt.mp
            / hydrogen_mass_fraction
        )
        self.fluid.rho = np.ones(self.par.nogrid) * rho
        coordinate = 0.5 * (self.mesh.boundary[1:] + self.mesh.boundary[:-1])
        midpoint = 0.5 * (icparams['rmin'] + icparams['rmax'])
        self.fluid.vel = np.where(
            coordinate < midpoint,
            icparams['outflow_velocity'],
            icparams['inflow_velocity'],
        )
        self.fluid.temp = np.ones(self.par.nogrid) * icparams['inflow_temperature']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']


def physical_cells(header):
    noghost = int(header.attrs.get('noghost', 0))
    nogrid = int(header.attrs['nogrid'])
    return slice(noghost, noghost + nogrid)


def load_snapshot(filename):
    with h5py.File(filename, 'r') as handle:
        data = handle['Data']
        header = handle['Header']
        physical = physical_cells(header)
        noghost = int(header.attrs.get('noghost', 0))
        nogrid = int(header.attrs['nogrid'])
        boundary = np.asarray(
            data['Boundary'][()]
        )[noghost:noghost + nogrid + 1]
        return {
            'time_Myr': float(header.attrs['Time']) / SECONDS_PER_MYR,
            'boundary_cm': boundary,
            'density_g_cm3': np.asarray(data['Density'][()])[physical],
            'velocity_cm_s': np.asarray(data['Velocity'][()])[physical],
            'temperature_K': np.asarray(data['Temperature'][()])[physical],
        }


def shock_radius(snapshot):
    """Locate the strongest compression near the colliding-stream interface."""
    boundary = snapshot['boundary_cm']
    centers = 0.5 * (boundary[1:] + boundary[:-1])
    density = np.maximum(snapshot['density_g_cm3'], 1.0e-99)
    gradient = np.abs(np.diff(np.log(density)))
    start = max(2, int(0.2 * len(gradient)))
    stop = min(len(gradient) - 1, int(0.9 * len(gradient)))
    index = start + int(np.argmax(gradient[start:stop]))
    return float(centers[index] / KPC_CM)


def shock_history(filenames, output_interval_myr=None):
    rows = []
    for filename in filenames:
        snapshot = load_snapshot(filename)
        if output_interval_myr is None:
            time_myr = snapshot['time_Myr']
        else:
            # Current HDF5 output headers do not preserve the evolving time
            # for this non-cosmological run.  The numbered output and the
            # configured fixed cadence are unambiguous.
            try:
                output_index = int(filename.stem.rsplit('_', 1)[1])
            except (AttributeError, IndexError, ValueError):
                output_index = len(rows)
            time_myr = output_index * float(output_interval_myr)
        rows.append((time_myr, shock_radius(snapshot)))
    return np.asarray(rows, dtype=float)


def estimate_cooling_length(snapshot, table, metallicity, hydrogen_mass_fraction, mu):
    """Estimate post-shock cooling length and cooling time from one snapshot."""
    radius = snapshot['boundary_cm']
    centers = 0.5 * (radius[1:] + radius[:-1])
    shock_kpc = shock_radius(snapshot)
    shock_index = int(np.argmin(np.abs(centers / KPC_CM - shock_kpc)))
    left = slice(max(0, shock_index - 4), shock_index)
    right = slice(shock_index + 1, min(len(centers), shock_index + 5))
    left_temperature = float(np.median(snapshot['temperature_K'][left]))
    right_temperature = float(np.median(snapshot['temperature_K'][right]))
    post_slice = right if right_temperature >= left_temperature else left
    density = float(np.median(snapshot['density_g_cm3'][post_slice]))
    temperature = float(np.median(snapshot['temperature_K'][post_slice]))
    velocity = float(np.median(np.abs(snapshot['velocity_cm_s'][post_slice])))
    n_h = hydrogen_mass_fraction * density / PROTON_MASS_G
    heating, cooling = table.rates(
        temperature, n_h, metallicity=metallicity, redshift=0.0
    )
    net_rate = max(float(np.asarray(cooling) - np.asarray(heating)), 1.0e-99)
    thermal_energy = 1.5 * density * 1.380649e-16 * temperature / (
        mu * PROTON_MASS_G
    )
    cooling_time_s = thermal_energy / net_rate
    cooling_length_cm = abs(velocity) * cooling_time_s
    cell_width_cm = float(np.median(np.diff(radius)))
    return {
        'shock_radius_kpc': shock_kpc,
        'post_density_g_cm3': density,
        'post_temperature_K': temperature,
        'cooling_time_Myr': cooling_time_s / SECONDS_PER_MYR,
        'cooling_length_kpc': cooling_length_cm / KPC_CM,
        'cooling_cells': cooling_length_cm / cell_width_cm,
    }
