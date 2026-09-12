"""Initial conditions and plotting for the NFW virial-shock example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs, time_seconds, quantity_to_value
from radhydropy.rsim import Rsim
from radhydropy.initial_condition_writer import InitialConditionWriter
from example.NFWHydrostaticEquilibrium1D import tools as NFW


def cosmic_mean_baryon_density(h0, omega_b, redshift):
    """Return the cosmic-mean baryon density at the initial redshift."""
    h0_cgs = h0.to(1.0 / unyt.s)
    rho_critical = 3.0 * h0_cgs**2 / (
        8.0 * np.pi * unyt.physical_constants.gravitational_constant
    )
    return (
        float(omega_b) * rho_critical * (1.0 + float(redshift))**3
    ).to(unyt.g / unyt.cm**3)


def hubble_rate(h0, omega_m, omega_lambda, redshift):
    """Return a flat-background Hubble rate at the initial redshift."""
    return h0 * np.sqrt(
        float(omega_m) * (1.0 + float(redshift))**3
        + float(omega_lambda)
    )


def build_initial_condition(config):
    initial_condition = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    boundary_unyt = np.linspace(
        initial_condition['radius_inner_proper'],
        initial_condition['radius_outer_proper'],
        grid_cells + 1,
    )
    coordinate_unyt = NFW.spherical_cell_centers(boundary_unyt)
    mean_density = cosmic_mean_baryon_density(
        initial_condition['h0'],
        initial_condition['omega_b'],
        initial_condition['initial_redshift'],
    )
    expansion_rate = hubble_rate(
        initial_condition['h0'],
        initial_condition['omega_m'],
        initial_condition['omega_lambda'],
        initial_condition['initial_redshift'],
    )
    cmb_temperature = initial_condition.get('cmb_temperature_0', initial_condition['temperature_proper'])
    temperature_proper_cgs_K = cmb_temperature * (1.0 + float(initial_condition['initial_redshift']))
    writer = InitialConditionWriter(par_config=config['par'], code_units=code_units)
    writer.mesh.boundary_radarray = writer.radarray(boundary_unyt)
    writer.mesh.x_radarray = writer.radarray(coordinate_unyt)
    writer.fluid.rho_radarray = writer.radarray(np.ones(grid_cells) * mean_density)
    writer.fluid.vel_radarray = writer.radarray(expansion_rate * coordinate_unyt)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * temperature_proper_cgs_K)
    writer.simulation.fluid.mu = np.full(grid_cells, float(initial_condition['mu']))
    return writer

def _snapshot_profiles(filename, config):
    rout = rio.loadhdf5(config, filename)
    boundary_cgs_cm = rout.mesh.boundary_radarray.to(unyt.cm)
    radius_proper_cgs_cm = NFW.spherical_cell_centers(boundary_cgs_cm)
    nghost = int(config['par']['mesh']['ghost_cells'])
    active_slice = slice(nghost, -nghost if nghost else None)
    radius_proper_cgs_cm = radius_proper_cgs_cm[active_slice]
    density_proper_cgs_g_cm3 = rout.fluid.rho_radarray.to(unyt.g / unyt.cm**3).value[active_slice]
    temperature_proper_cgs_K = rout.fluid.temp_radarray.to(unyt.K).value[active_slice]
    vel_peculiar_proper_km_s = rout.fluid.vel_radarray.to(unyt.km / unyt.s).value[active_slice]
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    time_proper_Myr = time_seconds(rout.fluid.time_proper_code, code_units) / float(
        (1.0 * unyt.Myr).to_value(unyt.s)
    )
    radius_proper_kpc = radius_proper_cgs_cm.to_value(unyt.kpc)
    return time_proper_Myr, radius_proper_kpc, density_proper_cgs_g_cm3, temperature_proper_cgs_K, vel_peculiar_proper_km_s


def rankine_hugoniot_ratios(mach_number, gamma=5.0 / 3.0):
    """Return finite-Mach Rankine--Hugoniot density and temperature ratios."""
    mach_squared = np.asarray(mach_number, dtype=float)**2
    density_ratio = (
        (gamma + 1.0) * mach_squared
        / ((gamma - 1.0) * mach_squared + 2.0)
    )
    pressure_ratio = (
        2.0 * gamma * mach_squared - (gamma - 1.0)
    ) / (gamma + 1.0)
    return density_ratio, pressure_ratio / density_ratio


def rankine_hugoniot_diagnostics(filenames, config):
    """Compare detected shock jumps with Rankine--Hugoniot predictions."""
    profiles = [_snapshot_profiles(filename, config) for filename in filenames]
    if len(profiles) < 3:
        return []
    gamma = float(config['par']['hydrodynamics']['gamma'])
    mu = float(config['initial_condition']['mu'])
    shock_positions = []
    shock_indices = []
    for _, radius_proper_cgs_cm, _, temperature_proper_cgs_K, _ in profiles:
        gradient = np.abs(
            np.diff(np.log(np.maximum(temperature_proper_cgs_K, 1.0))) / np.diff(radius_proper_cgs_cm)
        )
        # The accretion shock is the inner, hot-side edge of the infalling
        # shell. Exclude only ghost-adjacent cells and the outer boundary edge;
        # restricting to the outer half would select the reflecting boundary.
        lower = 2
        upper = max(lower + 1, len(gradient) - 5)
        index = lower + int(np.argmax(gradient[lower:upper]))
        shock_indices.append(index)
        shock_positions.append(radius_proper_cgs_cm[index])

    rows = []
    kpc_per_myr_to_km_s = 977.792221
    for snapshot_index in range(1, len(profiles) - 1):
        time_proper_Myr, radius_proper_cgs_cm, density_proper_cgs_g_cm3, temperature_proper_cgs_K, vel_peculiar_proper_cgs_cm_s = profiles[snapshot_index]
        previous_time = profiles[snapshot_index - 1][0]
        next_time = profiles[snapshot_index + 1][0]
        dt_myr = next_time - previous_time
        if dt_myr <= 0.0:
            continue
        shock_speed = (
            shock_positions[snapshot_index + 1]
            - shock_positions[snapshot_index - 1]
        ) / dt_myr * kpc_per_myr_to_km_s
        index = shock_indices[snapshot_index]
        if index < 5 or index + 5 > len(radius_proper_cgs_cm):
            continue
        upstream = slice(index + 2, index + 5)
        downstream = slice(index - 4, index - 1)
        density_upstream_proper_cgs_g_cm3 = float(np.median(density_proper_cgs_g_cm3[upstream]))
        density_downstream_proper_cgs_g_cm3 = float(np.median(density_proper_cgs_g_cm3[downstream]))
        temperature_upstream_proper_cgs_K = float(np.median(temperature_proper_cgs_K[upstream]))
        temperature_downstream_proper_cgs_K = float(np.median(temperature_proper_cgs_K[downstream]))
        vel_upstream_peculiar_proper_cgs_cm_s = float(np.median(vel_peculiar_proper_cgs_cm_s[upstream]))
        vel_downstream_peculiar_proper_cgs_cm_s = float(np.median(vel_peculiar_proper_cgs_cm_s[downstream]))
        relative_upstream_peculiar_proper_cgs_cm_s = abs(vel_upstream_peculiar_proper_cgs_cm_s - shock_speed)
        sound_speed = np.sqrt(
            gamma * 1.380649e-16 * temperature_upstream_proper_cgs_K
            / (mu * 1.67262192369e-24)
        ) / 1.0e5
        mach_number = relative_upstream_peculiar_proper_cgs_cm_s / max(sound_speed, 1.0e-30)
        predicted_density, predicted_temperature = rankine_hugoniot_ratios(
            mach_number,
            gamma,
        )
        rows.append({
            'time_proper_Myr': time_proper_Myr,
            'shock_radius_proper_kpc': shock_positions[snapshot_index],
            'shock_speed_proper_km_s': shock_speed,
            'mach_number_dimensionless': mach_number,
            'density_ratio_measured_dimensionless': density_downstream_proper_cgs_g_cm3 / max(density_upstream_proper_cgs_g_cm3, 1.0e-99),
            'density_ratio_predicted_dimensionless': float(predicted_density),
            'temperature_ratio_measured_dimensionless': temperature_downstream_proper_cgs_K / max(temperature_upstream_proper_cgs_K, 1.0e-99),
            'temperature_ratio_predicted_dimensionless': float(predicted_temperature),
            'vel_downstream_peculiar_proper_km_s': vel_downstream_peculiar_proper_cgs_cm_s / 1.0e5,
        })
    return rows


def write_rankine_hugoniot_report(rows, filename):
    """Write the shock-jump comparison as a text table."""
    header = (
        'time_proper_Myr shock_radius_proper_kpc shock_speed_proper_km_s mach_number_dimensionless '
        'density_ratio_measured_dimensionless density_ratio_predicted_dimensionless '
        'temperature_ratio_measured_dimensionless temperature_ratio_predicted_dimensionless '
        'vel_downstream_peculiar_proper_km_s'
    )
    with open(filename, 'w', encoding='utf-8') as report:
        report.write(header + '\n')
        for row in rows:
            report.write(
                '%(time_proper_Myr).8g %(shock_radius_proper_kpc).8g '
                '%(shock_speed_proper_km_s).8g %(mach_number_dimensionless).8g '
                '%(density_ratio_measured_dimensionless).8g %(density_ratio_predicted_dimensionless).8g '
                '%(temperature_ratio_measured_dimensionless).8g %(temperature_ratio_predicted_dimensionless).8g '
                '%(vel_downstream_peculiar_proper_km_s).8g\n' % row
            )


def plot_snapshots(filenames, config, figure_filename):
    """Plot density and temperature profiles from all saved snapshots."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(filenames)))
    initial_condition = config['initial_condition']
    halo = NFW.nfw_halo_parameters(
        initial_condition['halo_mass'],
        initial_condition['concentration'],
        initial_condition['redshift'],
        initial_condition['overdensity'],
        initial_condition['h0'],
    )
    virial_radius_proper_kpc = halo['radius_virial_proper_kpc_unyt'].to_value(unyt.kpc)
    for color, filename in zip(colors, filenames):
        time_proper_Myr, radius_proper_kpc, density_proper_cgs_g_cm3, temperature_proper_cgs_K, _ = _snapshot_profiles(
            filename,
            config,
        )
        label = f'{time_proper_Myr:.0f} Myr'
        axes[0].plot(radius_proper_kpc, density_proper_cgs_g_cm3, color=color, label=label)
        axes[1].plot(radius_proper_kpc, temperature_proper_cgs_K, color=color, label=label)
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xlabel('r [kpc]')
    axes[1].set_xlabel('r [kpc]')
    axes[0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
    axes[1].set_ylabel('T [K]')
    for axis in axes:
        axis.axvline(virial_radius_proper_kpc, color='black', ls=':', alpha=0.6)
        axis.axvline(2.0 * virial_radius_proper_kpc, color='black', ls='--', alpha=0.6)
        axis.grid(True, which='both', alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].text(virial_radius_proper_kpc, 0.04, 'R200', transform=axes[0].get_xaxis_transform(), ha='center')
    axes[0].text(2.0 * virial_radius_proper_kpc, 0.04, '2R200', transform=axes[0].get_xaxis_transform(), ha='center')
    axes[1].axhline(2.7255, color='black', ls='--', alpha=0.6, label='CMB')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Virial shock around a 1e8 Msun NFW halo')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
