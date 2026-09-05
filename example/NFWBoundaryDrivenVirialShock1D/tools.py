"""Helpers for a boundary-driven virial shock in a fixed NFW halo."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS


NFW_TOOLS = Path(__file__).resolve().parents[1] / 'NFWHydrostaticEquilibrium1D' / 'tools.py'
SPEC = importlib.util.spec_from_file_location('boundary_accretion_nfw_tools', NFW_TOOLS)
NFW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NFW)

KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KM_S_TO_CM_S = 1.0e5
GAMMA_CRITICAL = 10.0 / 7.0

nfw_halo_parameters = NFW.nfw_halo_parameters
virial_temperature = NFW.virial_temperature
spherical_cell_centers = NFW.spherical_cell_centers


def pie_equilibrium_temperature(
    density_cgs_g_cm3, table, hydrogen_mass_fraction, metallicity, redshift
):
    """Return the lowest stable HM12 PIE equilibrium on the table grid."""
    density = np.atleast_1d(np.asarray(density_cgs_g_cm3, dtype=float))
    n_h = hydrogen_mass_fraction * density / PROTON_MASS_CGS
    log_temperature = np.asarray(table.log_temperature, dtype=float)
    temperature = 10.0 ** log_temperature
    heating, cooling = table.rates(
        temperature[:, None], n_h[None, :],
        metallicity=metallicity, redshift=redshift,
    )
    difference = np.asarray(heating) - np.asarray(cooling)
    crossing = difference[:-1] * difference[1:] <= 0.0
    result = np.empty(density.size)
    for cell in range(density.size):
        candidates = np.flatnonzero(crossing[:, cell])
        if candidates.size:
            index = int(candidates[0])
            lower = difference[index, cell]
            upper = difference[index + 1, cell]
            fraction = -lower / (upper - lower) if upper != lower else 0.0
            result[cell] = 10.0 ** (
                log_temperature[index]
                + fraction * (log_temperature[index + 1] - log_temperature[index])
            )
        else:
            result[cell] = temperature[np.argmin(np.abs(difference[:, cell]))]
    return result


def inflow_density(mass_accretion_rate, radius, velocity):
    """Return density from ``Mdot = 4 pi r^2 rho |v|``."""
    mdot = mass_accretion_rate.to_value(unyt.g / unyt.s)
    radius_cgs_cm = radius.to_value(unyt.cm)
    speed = abs(velocity.to_value(unyt.cm / unyt.s))
    return mdot / (4.0 * np.pi * radius_cgs_cm**2 * speed) * unyt.g / unyt.cm**3


def boundary_inflow_state(icparams, halo, table, par_config):
    """Return the maintained outer-boundary state in physical units."""
    radius = float(icparams['outer_radius_over_R200']) * halo['virial_radius']
    velocity = (
        -float(icparams['inflow_velocity_over_V200']) * halo['virial_velocity']
    ).to(unyt.km / unyt.s)
    density = inflow_density(icparams['baryon_accretion_rate'], radius, velocity)
    temperature = pie_equilibrium_temperature(
        [density.to_value(unyt.g / unyt.cm**3)], table,
        float(par_config['chemistry']['hydrogen_mass_fraction']),
        float(par_config['thermochemistry']['metallicity']),
        float(par_config['thermochemistry']['metal_pie_redshift']),
    )[0] * unyt.K
    return {
        'inflow_density': density,
        'inflow_velocity': velocity,
        'inflow_temperature': temperature,
        'inflow_mu': float(icparams['mu']),
    }


def build_initial_condition(config):
    """Build a hot NFW atmosphere with a steady cold PIE inflow."""
    icparams = config['initial_condition']
    par_config = config['par']
    code_units = config['_code_units']
    table = config['_pie_table']
    sim = SimpleNamespace(
        par=SimpleNamespace(units=SimpleNamespace(CodeUnits=code_units)),
        mesh=SimpleNamespace(),
        fluid=SimpleNamespace(),
    )
    grid_cells = int(par_config['mesh']['grid_cells'])
    halo = nfw_halo_parameters(
        icparams['halo_mass'], icparams['concentration'],
        icparams['redshift'], icparams['overdensity'], icparams['h0'],
    )
    r200 = halo['virial_radius']
    inner = float(icparams['inner_radius_over_R200']) * r200
    outer = float(icparams['outer_radius_over_R200']) * r200
    sim.par.simulation = SimpleNamespace(
        coordinate_system='spherical', current_time=icparams['current_time'], box_size=outer,
    )
    sim.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
    sim.mesh.boundary = np.geomspace(inner.to_value(unyt.kpc), outer.to_value(unyt.kpc), grid_cells + 1) * unyt.kpc
    sim.mesh.coordinate = spherical_cell_centers(sim.mesh.boundary)
    radius = sim.mesh.coordinate
    inflow_velocity = (-float(icparams['inflow_velocity_over_V200']) * halo['virial_velocity']).to(unyt.cm / unyt.s)
    mdot = icparams['baryon_accretion_rate']
    rho_cold = inflow_density(mdot, radius, inflow_velocity)
    temperature_cold = pie_equilibrium_temperature(
        rho_cold.to_value(unyt.g / unyt.cm**3), table,
        float(par_config['chemistry']['hydrogen_mass_fraction']),
        float(par_config['thermochemistry']['metallicity']),
        float(par_config['thermochemistry']['metal_pie_redshift']),
    ) * unyt.K
    transition = float(icparams['atmosphere_radius_over_R200']) * r200
    transition_density = inflow_density(mdot, transition, inflow_velocity)
    ram_pressure = transition_density * inflow_velocity**2
    hot_pressure = float(icparams['atmosphere_ram_pressure_fraction']) * ram_pressure
    hot_temperature = virial_temperature(halo, float(icparams['mu']))
    rho_transition = (
        hot_pressure.to_value(unyt.erg / unyt.cm**3) * float(icparams['mu']) * PROTON_MASS_CGS
        / (BOLTZMANN_CONSTANT_CGS * hot_temperature.to_value(unyt.K))
    ) * unyt.g / unyt.cm**3
    potential = NFW.nfw_potential(radius, halo['scale_density'], halo['scale_radius']).to_value(unyt.cm**2 / unyt.s**2)
    potential_transition = NFW.nfw_potential(transition, halo['scale_density'], halo['scale_radius']).to_value(unyt.cm**2 / unyt.s**2)
    beta = float(icparams['mu']) * PROTON_MASS_CGS / (BOLTZMANN_CONSTANT_CGS * hot_temperature.to_value(unyt.K))
    rho_hot = rho_transition * np.exp(-beta * (potential - potential_transition))
    width = float(icparams['transition_width_over_R200']) * r200
    weight = 0.5 * (1.0 + np.tanh(((radius - transition) / width).to_value(unyt.dimensionless)))
    log_density = (
        (1.0 - weight) * np.log(rho_hot.to_value(unyt.g / unyt.cm**3))
        + weight * np.log(rho_cold.to_value(unyt.g / unyt.cm**3))
    )
    sim.fluid.rho_code = np.exp(log_density) * unyt.g / unyt.cm**3
    sim.fluid.vel_code = weight * inflow_velocity
    sim.fluid.temp_code = (
        (1.0 - weight) * hot_temperature.to_value(unyt.K)
        + weight * temperature_cold.to_value(unyt.K)
    ) * unyt.K
    sim.fluid.mu = np.ones(grid_cells) * float(icparams['mu'])
    sim.fluid.xHI = np.ones(grid_cells)
    return sim

def load_snapshot(filename):
    """Load physical cells from a RadHydropy snapshot in CGS units."""
    with h5py.File(filename, 'r') as handle:
        header = handle['Header'].attrs
        header_group = handle['Header']
        data = handle['Data']
        nogrid = int(header['GridCells'])
        noghost = int(header.get('GhostCells', (len(data['rho_code']) - nogrid) // 2))
        physical = slice(noghost, noghost + nogrid)
        boundary = np.asarray(data['boundary'])[noghost:noghost + nogrid + 1]
        centers = 0.75 * (
            boundary[1:]**4 - boundary[:-1]**4
        ) / (boundary[1:]**3 - boundary[:-1]**3)
        return {
            'time_Myr': float(header_group['time_code'][()]) / SECONDS_PER_MYR,
            'radius_kpc': centers / KPC_CM,
            'density_cgs_g_cm3': np.asarray(data['rho_code'])[physical],
            'velocity_km_s': np.asarray(data['vel_code'])[physical] / 1.0e5,
            'temperature_cgs_K': np.asarray(data['temp_code'])[physical],
        }


def locate_shock(snapshot, r200_kpc):
    """Locate the strongest entropy-producing compression near the halo."""
    radius = snapshot['radius_kpc']
    density = np.maximum(snapshot['density_cgs_g_cm3'], 1.0e-99)
    temperature = np.maximum(snapshot['temperature_cgs_K'], 1.0)
    pressure = density * temperature
    entropy = pressure / density**(5.0 / 3.0)
    # Radius increases with array index, so a compressed downstream (inner)
    # state is a *negative* outward density gradient.
    score = np.abs(np.diff(np.log(entropy))) + np.maximum(
        -np.diff(np.log(density)), 0.0
    )
    density_ratio = density[:-1] / density[1:]
    temperature_ratio = temperature[:-1] / temperature[1:]
    velocity = snapshot.get('velocity_km_s')
    decelerating = (
        np.ones_like(density_ratio, dtype=bool)
        if velocity is None else np.asarray(velocity[:-1] - velocity[1:] > 0.0)
    )
    candidate = (
        (radius[:-1] > 0.1 * r200_kpc)
        & (radius[:-1] < 2.0 * r200_kpc)
        & (density_ratio > 1.02)
        & (temperature_ratio > 1.02)
        & decelerating
    )
    if not np.any(candidate):
        return None
    indices = np.flatnonzero(candidate)
    index = int(indices[np.argmax(score[candidate])])
    if score[index] < 0.08:
        return None
    return index


def shock_history(filenames, halo, times_myr=None):
    """Return shock radius and jump diagnostics from saved snapshots."""
    r200 = halo['virial_radius'].to_value(unyt.kpc)
    rows = []
    for file_index, filename in enumerate(filenames):
        snapshot = load_snapshot(filename)
        if times_myr is not None:
            snapshot['time_Myr'] = float(times_myr[file_index])
        index = locate_shock(snapshot, r200)
        if index is None or index < 3 or index + 4 >= len(snapshot['radius_kpc']):
            continue
        inner = slice(index - 3, index)
        outer = slice(index + 1, index + 4)
        rho_in = float(np.median(snapshot['density_cgs_g_cm3'][inner]))
        rho_out = float(np.median(snapshot['density_cgs_g_cm3'][outer]))
        temp_in = float(np.median(snapshot['temperature_cgs_K'][inner]))
        temp_out = float(np.median(snapshot['temperature_cgs_K'][outer]))
        rows.append({
            'time_Myr': snapshot['time_Myr'],
            'shock_radius_kpc': float(snapshot['radius_kpc'][index]),
            'shock_radius_over_R200': float(snapshot['radius_kpc'][index] / r200),
            'density_ratio': rho_in / max(rho_out, 1.0e-99),
            'temperature_ratio': temp_in / max(temp_out, 1.0),
            'velocity_inner_km_s': float(np.median(snapshot['velocity_km_s'][inner])),
            'velocity_outer_km_s': float(np.median(snapshot['velocity_km_s'][outer])),
        })
    return rows


def write_report(rows, filename):
    with Path(filename).open('w', encoding='utf-8') as stream:
        stream.write(
            'time_Myr shock_radius_kpc shock_radius_over_R200 density_ratio '
            'temperature_ratio velocity_inner_km_s velocity_outer_km_s\n'
        )
        for row in rows:
            stream.write(
                '%(time_Myr).8g %(shock_radius_kpc).8g '
                '%(shock_radius_over_R200).8g %(density_ratio).8g '
                '%(temperature_ratio).8g %(velocity_inner_km_s).8g '
                '%(velocity_outer_km_s).8g\n' % row
            )


def _gas_pressure(density, temperature, mu):
    return density * BOLTZMANN_CONSTANT_CGS * temperature / (mu * PROTON_MASS_CGS)


def _pie_net_rate(table, density, temperature, runparams):
    n_h = float(runparams['chemistry']['hydrogen_mass_fraction']) * density / PROTON_MASS_CGS
    heating, cooling = table.rates(
        temperature, n_h,
        metallicity=float(runparams['thermochemistry']['metallicity']),
        redshift=float(runparams['thermochemistry']['metal_pie_redshift']),
    )
    return float(np.asarray(cooling)) - float(np.asarray(heating))


def pie_stability_diagnostics(
    filenames, times_myr, halo, table, runparams, mu,
):
    """Compare simulated post-shock states with finite-Mach estimates."""
    profiles = [load_snapshot(name) for name in filenames]
    r200 = halo['virial_radius'].to_value(unyt.kpc)
    indices = [locate_shock(profile, r200) for profile in profiles]
    radii = [
        None if index is None else profile['radius_kpc'][index]
        for profile, index in zip(profiles, indices)
    ]
    gamma = float(runparams['hydrodynamics']['gamma'])
    downstream = []
    for profile, index in zip(profiles, indices):
        if index is None or index < 8 or index + 5 >= len(profile['radius_kpc']):
            downstream.append(None)
            continue
        band = slice(index - 8, index - 3)
        rho = float(np.median(profile['density_cgs_g_cm3'][band]))
        temperature = float(np.median(profile['temperature_cgs_K'][band]))
        downstream.append((rho, temperature, _gas_pressure(rho, temperature, mu)))

    rows = []
    for i in range(len(profiles)):
        if any(value is None for value in (
            indices[i], radii[i], downstream[i],
        )):
            continue
        if i > 0 and i + 1 < len(profiles) and radii[i - 1] is not None and radii[i + 1] is not None:
            dt_myr = float(times_myr[i + 1] - times_myr[i - 1])
            shock_speed = (radii[i + 1] - radii[i - 1]) / dt_myr * 977.792221
        elif i > 0 and radii[i - 1] is not None:
            dt_myr = float(times_myr[i] - times_myr[i - 1])
            shock_speed = (radii[i] - radii[i - 1]) / dt_myr * 977.792221
        elif i + 1 < len(profiles) and radii[i + 1] is not None:
            dt_myr = float(times_myr[i + 1] - times_myr[i])
            shock_speed = (radii[i + 1] - radii[i]) / dt_myr * 977.792221
        else:
            shock_speed = 0.0
        profile = profiles[i]
        index = indices[i]
        upstream = slice(index + 2, index + 5)
        rho0 = float(np.median(profile['density_cgs_g_cm3'][upstream]))
        temp0 = float(np.median(profile['temperature_cgs_K'][upstream]))
        velocity0 = float(np.median(profile['velocity_km_s'][upstream]))
        relative_speed = abs(velocity0 - shock_speed)
        sound_speed = np.sqrt(
            gamma * BOLTZMANN_CONSTANT_CGS * max(temp0, 1.0)
            / (float(mu) * PROTON_MASS_CGS)
        ) / KM_S_TO_CM_S
        mach = relative_speed / max(sound_speed, 1.0e-30)
        mach2 = mach**2
        density_ratio = (gamma + 1.0) * mach2 / (
            (gamma - 1.0) * mach2 + 2.0
        )
        pressure_ratio = (
            2.0 * gamma * mach2 - (gamma - 1.0)
        ) / (gamma + 1.0)
        rho_analytic = rho0 * density_ratio
        pressure_analytic = _gas_pressure(rho0, temp0, mu) * pressure_ratio
        temp_analytic = temp0 * pressure_ratio / max(density_ratio, 1.0e-30)
        ram_pressure = rho0 * (relative_speed * KM_S_TO_CM_S)**2

        rho1, temp1, pressure1 = downstream[i]
        net_rate = _pie_net_rate(table, rho1, temp1, runparams)
        analytic_net_rate = _pie_net_rate(
            table, rho_analytic, temp_analytic, runparams
        )
        energy1 = pressure1 / (gamma - 1.0)
        energy_analytic = pressure_analytic / (gamma - 1.0)
        cooling_time = (
            energy1 / net_rate / SECONDS_PER_MYR if net_rate > 0.0 else np.inf
        )
        analytic_cooling_time = (
            energy_analytic / analytic_net_rate / SECONDS_PER_MYR
            if analytic_net_rate > 0.0 else np.inf
        )

        # Birnboim & Dekel's effective index follows a compressed fluid
        # element. Estimate dln(rho)/dt=-div(v) directly from the spherical
        # upstream flow, avoiding a fixed-Eulerian-band time derivative.
        radius_cgs_cm = profile['radius_kpc'] * KPC_CM
        velocity_cgs_cm_s = profile['velocity_km_s'] * KM_S_TO_CM_S
        divergence = np.gradient(
            radius_cgs_cm**2 * velocity_cgs_cm_s, radius_cgs_cm
        ) / radius_cgs_cm**2
        compression_rate = -float(np.median(divergence[upstream]))
        # Birnboim & Dekel's local definition follows directly from
        # P=(gamma-1)*rho*e and de/dt=P/rho**2*d(rho)/dt-q:
        # gamma_eff = gamma - rho*q/(dot(rho)*e).  Keep q as the net
        # volumetric cooling rate and e as specific internal energy here;
        # this avoids hiding the density and energy conventions inside a
        # cooling/compression-timescale ratio.
        density_rate = rho1 * compression_rate
        analytic_density_rate = rho_analytic * compression_rate
        specific_energy1 = energy1 / max(rho1, 1.0e-99)
        specific_energy_analytic = energy_analytic / max(rho_analytic, 1.0e-99)
        gamma_eff = (
            gamma - rho1 * net_rate / (density_rate * specific_energy1)
            if compression_rate > 1.0e-30 else np.nan
        )
        gamma_eff_analytic = (
            gamma - rho_analytic * analytic_net_rate
            / (analytic_density_rate * specific_energy_analytic)
            if compression_rate > 1.0e-30 else np.nan
        )
        rows.append({
            'time_Myr': float(times_myr[i]),
            'shock_radius_over_R200': float(radii[i] / r200),
            'shock_speed_km_s': float(shock_speed),
            'mach_number': float(mach),
            'postshock_pressure_cgs_erg_cm3': float(pressure1),
            'analytic_postshock_pressure_cgs_erg_cm3': float(pressure_analytic),
            'ram_pressure_cgs_erg_cm3': float(ram_pressure),
            'postshock_to_ram_pressure': float(pressure1 / max(ram_pressure, 1.0e-99)),
            'analytic_postshock_to_ram_pressure': float(
                pressure_analytic / max(ram_pressure, 1.0e-99)
            ),
            'postshock_temperature_cgs_K': float(temp1),
            'analytic_postshock_temperature_cgs_K': float(temp_analytic),
            'cooling_time_Myr': float(cooling_time),
            'analytic_cooling_time_Myr': float(analytic_cooling_time),
            'gamma_eff': float(gamma_eff),
            'gamma_eff_analytic': float(gamma_eff_analytic),
            'gamma_critical': GAMMA_CRITICAL,
        })
    return rows


def write_stability_report(rows, filename):
    keys = (
        'time_Myr', 'shock_radius_over_R200', 'shock_speed_km_s', 'mach_number',
        'postshock_pressure_cgs_erg_cm3', 'analytic_postshock_pressure_cgs_erg_cm3',
        'ram_pressure_cgs_erg_cm3', 'postshock_to_ram_pressure',
        'analytic_postshock_to_ram_pressure',
        'postshock_temperature_cgs_K', 'analytic_postshock_temperature_cgs_K',
        'cooling_time_Myr', 'analytic_cooling_time_Myr', 'gamma_eff',
        'gamma_eff_analytic', 'gamma_critical',
    )
    with Path(filename).open('w', encoding='utf-8') as stream:
        stream.write(' '.join(keys) + '\n')
        for row in rows:
            stream.write(' '.join(f"{row[key]:.8g}" for key in keys) + '\n')


def plot_stability_diagnostics(rows, filename):
    """Plot pressure, ram-pressure balance, temperature, cooling, and gamma."""
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.0), sharex=True)
    if not rows:
        axes[0, 0].text(0.5, 0.5, 'No time-resolved shock diagnostics',
                        ha='center', va='center')
        for axis in axes.flat:
            axis.set_axis_off()
    else:
        time = np.asarray([row['time_Myr'] for row in rows])
        panels = (
            ('postshock_pressure_cgs_erg_cm3', 'analytic_postshock_pressure_cgs_erg_cm3',
             r'$P_1$ [erg cm$^{-3}$]', True),
            ('postshock_to_ram_pressure', 'analytic_postshock_to_ram_pressure',
             r'$P_1/P_{\rm ram}$', False),
            ('postshock_temperature_cgs_K', 'analytic_postshock_temperature_cgs_K',
             r'$T_1$ [K]', True),
            ('cooling_time_Myr', 'analytic_cooling_time_Myr',
             r'$t_{\rm cool}$ [Myr]', True),
            ('gamma_eff', 'gamma_eff_analytic', r'$\gamma_{\rm eff}$', False),
            ('shock_radius_over_R200', 'shock_radius_over_R200',
             r'$r_{\rm shock}/R_{200}$', False),
        )
        for axis, (measured, analytic, ylabel, logarithmic) in zip(axes.flat, panels):
            axis.plot(time, [row[measured] for row in rows], 'o-', label='simulation')
            axis.plot(time, [row[analytic] for row in rows], '--', label='analytic')
            axis.set_ylabel(ylabel)
            if logarithmic:
                axis.set_yscale('log')
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
        axes[1, 1].axhline(5.0 / 3.0, color='black', ls=':', label=r'$5/3$')
        axes[1, 1].axhline(GAMMA_CRITICAL, color='red', ls='--', label=r'$10/7$')
        axes[1, 1].legend(frameon=False, fontsize=8)
        for axis in axes[1]:
            axis.set_xlabel('total time [Myr]')
    fig.suptitle('Boundary-driven HM12 PIE shock diagnostics')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def plot_comparison(
    adiabatic_files, pie_files, halo, filename,
    adiabatic_times_myr=None, pie_times_myr=None,
):
    """Plot profiles and shock histories for the settling and PIE stages."""
    stages = [
        ('adiabatic settling', adiabatic_files, adiabatic_times_myr),
        ('PIE restart', pie_files, pie_times_myr),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.5), squeeze=False)
    r200 = halo['virial_radius'].to_value(unyt.kpc)
    for row, (label, files, times_myr) in enumerate(stages):
        selected = np.unique(np.linspace(0, len(files) - 1, 6, dtype=int))
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(selected)))
        for color, index in zip(colors, selected):
            snapshot = load_snapshot(files[index])
            if times_myr is not None:
                snapshot['time_Myr'] = float(times_myr[index])
            radius = snapshot['radius_kpc'] / r200
            plot_label = f"{snapshot['time_Myr']:.0f} Myr"
            axes[row, 0].plot(radius, snapshot['density_cgs_g_cm3'], color=color,
                              label=plot_label)
            axes[row, 1].plot(radius, snapshot['temperature_cgs_K'], color=color)
            axes[row, 2].plot(radius, snapshot['velocity_km_s'], color=color)
        history = shock_history(files, halo, times_myr=times_myr)
        if history:
            axes[row, 3].plot(
                [item['time_Myr'] for item in history],
                [item['shock_radius_over_R200'] for item in history],
                marker='o', ms=3,
            )
        axes[row, 0].set_ylabel(f'{label}\n' + r'$\rho$ [g cm$^{-3}$]')
        axes[row, 0].set_yscale('log')
        axes[row, 1].set_yscale('log')
        axes[row, 1].set_ylabel('T [K]')
        axes[row, 2].set_ylabel('velocity [km s$^{-1}$]')
        axes[row, 3].set_ylabel(r'$r_{\rm shock}/R_{200}$')
        axes[row, 0].legend(frameon=False, fontsize=8, ncol=2)
        for column in range(3):
            axes[row, column].set_xlim(0.0, 2.0)
            axes[row, column].axvline(1.0, color='black', ls=':', alpha=0.6)
            axes[row, column].set_xlabel(r'$r/R_{200}$')
        axes[row, 3].set_xlabel('time [Myr]')
        for axis in axes[row]:
            axis.grid(alpha=0.25)
    axes[0, 0].set_title('Density')
    axes[0, 1].set_title('Temperature')
    axes[0, 2].set_title('Radial velocity')
    axes[0, 3].set_title('Shock history')
    fig.suptitle('Boundary-driven accretion shock in a fixed NFW halo')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)
