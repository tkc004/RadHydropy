"""Initial conditions, diagnostics, and plotting for the adiabatic benchmark."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from types import SimpleNamespace

import radhydropy.io as rio
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.units import CodeUnits, code_quantity_to_cgs, time_seconds

NFW_TOOLS_PATH = __file__.replace(
    'NFWVirialShockAdiabatic1D/tools.py',
    'NFWHydrostaticEquilibrium1D/tools.py',
)
import importlib.util
NFW_SPEC = importlib.util.spec_from_file_location('nfw_halo_tools_adiabatic', NFW_TOOLS_PATH)
NFW = importlib.util.module_from_spec(NFW_SPEC)
assert NFW_SPEC.loader is not None
NFW_SPEC.loader.exec_module(NFW)


def nfw_halo_parameters(*args, **kwargs):
    return NFW.nfw_halo_parameters(*args, **kwargs)


def virial_temperature(*args, **kwargs):
    return NFW.virial_temperature(*args, **kwargs)


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def cosmic_mean_baryon_density(h0, omega_b, redshift):
    h0_cgs = h0.to(1.0 / unyt.s)
    rho_critical = 3.0 * h0_cgs**2 / (
        8.0 * np.pi * unyt.physical_constants.gravitational_constant
    )
    return (
        float(omega_b) * rho_critical * (1.0 + float(redshift))**3
    ).to(unyt.g / unyt.cm**3)


def hubble_rate(h0, omega_m, omega_lambda, redshift):
    return h0 * np.sqrt(
        float(omega_m) * (1.0 + float(redshift))**3
        + float(omega_lambda)
    )


class Simwrap:
    def __init__(self, config, code_units=None):
        icparams = config['initial_condition']
        grid_cells = int(config['par']['mesh']['grid_cells'])
        box_size = icparams['boxsize']
        time_value = icparams['time']
        radius_min = icparams['rmin']
        radius_max = icparams['rmax']
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        box_size = np.ones(1) * box_size
        self.par.time = np.ones(1) * time_value
        self.par.simulation = SimpleNamespace(
            coordinate_system='spherical',
            current_time=self.par.time,
            box_size=box_size,
        )
        self.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
        self.mesh.boundary = np.linspace(
            radius_min, radius_max, grid_cells + 1
        )
        self.mesh.coordinate = NFW.spherical_cell_centers(self.mesh.boundary)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = 4.0 * np.pi / 3.0 * (
            self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3
        )
        mean_density = cosmic_mean_baryon_density(
            icparams['h0'], icparams['omega_b'], icparams['initial_redshift']
        )
        expansion_rate = hubble_rate(
            icparams['h0'], icparams['omega_m'], icparams['omega_lambda'],
            icparams['initial_redshift']
        )
        cmb_temperature = icparams.get(
            'cmb_temperature_0', icparams['initial_temperature']
        )
        self.fluid.temp_code = np.ones(grid_cells) * cmb_temperature * (
            1.0 + float(icparams['initial_redshift'])
        )
        self.fluid.mu = np.ones(grid_cells) * icparams['mu']
        self.fluid.vel_code = expansion_rate * self.mesh.coordinate
        self.fluid.rho_code = np.ones(grid_cells) * mean_density


def _snapshot_profiles(filename, config):
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    rout = Simwrap(config, code_units=code_units)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, filename)
    boundary_cgs_cm = code_quantity_to_cgs(
        rout.mesh.boundary, code_units, 'length_cgs_cm'
    ) * unyt.cm
    radius = NFW.spherical_cell_centers(boundary_cgs_cm)
    nghost = int(config['par']['mesh']['ghost_cells'])
    radius = radius[nghost:-nghost]
    density = code_quantity_to_cgs(
        rout.fluid.rho_code[nghost:-nghost], code_units, 'density_cgs_g_cm3'
    )
    temperature = code_quantity_to_cgs(
        rout.fluid.temp_code[nghost:-nghost], code_units, 'temperature_cgs_K'
    )
    velocity = code_quantity_to_cgs(
        rout.fluid.vel_code[nghost:-nghost], code_units, 'velocity_cgs_cm_s'
    ) / 1.0e5
    time_myr = time_seconds(rout.fluid.time, code_units) / (1.0e6 * 365.25 * 86400.0)
    return time_myr, radius.to_value(unyt.kpc), density, temperature, velocity


def rankine_hugoniot_ratios(mach_number, gamma=5.0 / 3.0):
    mach_squared = np.asarray(mach_number, dtype=float)**2
    density_ratio = (gamma + 1.0) * mach_squared / (
        (gamma - 1.0) * mach_squared + 2.0
    )
    pressure_ratio = (2.0 * gamma * mach_squared - (gamma - 1.0)) / (
        gamma + 1.0
    )
    return density_ratio, pressure_ratio / density_ratio


def _locate_shock(radius, temperature, virial_radius_kpc):
    gradient = np.abs(
        np.diff(np.log(np.maximum(temperature, 1.0))) / np.diff(radius)
    )
    candidate = (
        (radius[:-1] > 0.5 * virial_radius_kpc)
        & (radius[:-1] < 3.5 * virial_radius_kpc)
    )
    if not np.any(candidate):
        raise RuntimeError('No shock candidate found in virial region')
    indices = np.flatnonzero(candidate)
    index = indices[np.argmax(gradient[candidate])]
    return int(index), float(radius[index])


def rankine_hugoniot_diagnostics(filenames, config, _unused, halo):
    profiles = [
        _snapshot_profiles(filename, config)
        for filename in filenames
    ]
    if len(profiles) < 3:
        return []
    gamma = float(config['par']['hydrodynamics']['gamma'])
    mu = float(config['initial_condition']['mu'])
    virial_radius_kpc = halo['virial_radius'].to_value(unyt.kpc)
    shock_positions = []
    shock_indices = []
    for _, radius, _, temperature, _ in profiles:
        index, position = _locate_shock(radius, temperature, virial_radius_kpc)
        shock_indices.append(index)
        shock_positions.append(position)

    rows = []
    for snapshot_index in range(1, len(profiles) - 1):
        time_myr, radius, density, temperature, velocity = profiles[snapshot_index]
        previous_time = profiles[snapshot_index - 1][0]
        next_time = profiles[snapshot_index + 1][0]
        dt_myr = next_time - previous_time
        if dt_myr <= 0.0:
            continue
        shock_speed = (
            shock_positions[snapshot_index + 1]
            - shock_positions[snapshot_index - 1]
        ) / dt_myr * 977.792221
        index = shock_indices[snapshot_index]
        if index < 5 or index + 5 > len(radius):
            continue
        upstream = slice(index + 2, index + 5)
        downstream = slice(index - 4, index - 1)
        rho_upstream = float(np.median(density[upstream]))
        rho_downstream = float(np.median(density[downstream]))
        temp_upstream = float(np.median(temperature[upstream]))
        temp_downstream = float(np.median(temperature[downstream]))
        velocity_upstream = float(np.median(velocity[upstream]))
        sound_speed = np.sqrt(
            gamma * BOLTZMANN_CONSTANT_CGS * temp_upstream
            / (mu * PROTON_MASS_CGS)
        ) / 1.0e5
        mach_number = abs(velocity_upstream - shock_speed) / max(sound_speed, 1.0e-30)
        predicted_density, predicted_temperature = rankine_hugoniot_ratios(
            mach_number, gamma
        )
        rows.append({
            'time_Myr': time_myr,
            'shock_radius_kpc': shock_positions[snapshot_index],
            'shock_radius_over_R200': shock_positions[snapshot_index] / virial_radius_kpc,
            'shock_speed_km_s': shock_speed,
            'mach_number': mach_number,
            'measured_density_ratio': rho_downstream / max(rho_upstream, 1.0e-99),
            'predicted_density_ratio': float(predicted_density),
            'measured_temperature_ratio': temp_downstream / max(temp_upstream, 1.0e-99),
            'predicted_temperature_ratio': float(predicted_temperature),
        })
    return rows


def write_rankine_hugoniot_report(rows, filename):
    header = (
        'time_Myr shock_radius_kpc shock_radius_over_R200 shock_speed_km_s '
        'Mach rho_ratio_measured rho_ratio_RH T_ratio_measured T_ratio_RH\n'
    )
    with open(filename, 'w', encoding='utf-8') as report:
        report.write(header)
        for row in rows:
            report.write(
                '%(time_Myr).8g %(shock_radius_kpc).8g '
                '%(shock_radius_over_R200).8g %(shock_speed_km_s).8g '
                '%(mach_number).8g %(measured_density_ratio).8g '
                '%(predicted_density_ratio).8g %(measured_temperature_ratio).8g '
                '%(predicted_temperature_ratio).8g\n' % row
            )


def plot_snapshots(filenames, config, _unused, halo, figure_filename):
    icparams = config['initial_condition']
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(filenames)))
    virial_radius = halo['virial_radius'].to_value(unyt.kpc)
    virial_temperature = NFW.virial_temperature(
        halo, icparams['mu']
    ).to_value(unyt.K)
    for color, filename in zip(colors, filenames):
        time_myr, radius, density, temperature, _ = _snapshot_profiles(
            filename, config
        )
        label = f'{time_myr:.0f} Myr'
        axes[0].plot(radius, density, color=color, label=label)
        axes[1].plot(radius, temperature, color=color, label=label)
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xlabel('r [kpc]')
    axes[1].set_xlabel('r [kpc]')
    axes[0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
    axes[1].set_ylabel('T [K]')
    for axis in axes:
        axis.axvline(virial_radius, color='black', ls=':', alpha=0.6)
        axis.axvline(2.0 * virial_radius, color='black', ls='--', alpha=0.6)
        axis.grid(True, which='both', alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].text(virial_radius, 0.04, 'R200', transform=axes[0].get_xaxis_transform(), ha='center')
    axes[0].text(2.0 * virial_radius, 0.04, '2R200', transform=axes[0].get_xaxis_transform(), ha='center')
    axes[1].axhline(virial_temperature, color='red', ls=':', label=r'$T_{vir}$')
    axes[1].legend(frameon=False, fontsize=8)
    mass = halo['mass'].to_value(unyt.Msun)
    fig.suptitle('Adiabatic virial shock around %.2g Msun NFW halo' % mass)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
