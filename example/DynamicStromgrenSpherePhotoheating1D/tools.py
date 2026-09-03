"""Utilities for the dynamic photoheated Stromgren sphere example."""

import glob
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import example_utils as eu

from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
import radhydropy.io as rio
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value

IONIZATION_FRONT_NEUTRAL_FRACTION = 0.5


def _to_kpc(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.kpc), dtype=float)
    code = getattr(par, 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'length_cm') / (1.0 * unyt.kpc).to_value(unyt.cm),
        dtype=float,
    )


def _to_myr(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.Myr), dtype=float)
    code = getattr(par, 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'time_s') / (1.0 * unyt.Myr).to_value(unyt.s),
        dtype=float,
    )


def _to_km_s(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.km / unyt.s), dtype=float)
    code = getattr(par, 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'velocity_cm_s') / (1.0 * unyt.km).to_value(unyt.cm),
        dtype=float,
    )


def _to_number_density(values, par):
    if hasattr(values, 'to_value'):
        density = np.asarray(values.to_value(unyt.g / unyt.cm**3), dtype=float)
        return density / (1.0 * unyt.mp).to_value(unyt.g)
    code = getattr(par, 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(
        code_quantity_to_cgs(values, code, 'density_g_cm3') / (1.0 * unyt.mp).to_value(unyt.g),
        dtype=float,
    )


def _to_pressure(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.g / unyt.cm / unyt.s**2), dtype=float)
    code = getattr(par, 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(code_quantity_to_cgs(values, code, 'pressure_erg_cm3'), dtype=float)


def _to_temperature(values, par):
    if hasattr(values, 'to_value'):
        return np.asarray(values.to_value(unyt.K), dtype=float)
    code = getattr(par, 'CodeUnits', None)
    if code is None:
        return np.asarray(values, dtype=float)
    return np.asarray(code_quantity_to_cgs(values, code, 'temperature_K'), dtype=float)


def load_parameters(config_filename, rundir=None, clean_previous_outputs=True):
    from radhydropy.example_config import load_example_parameters

    config_filename = Path(config_filename)
    runparams, icparams = load_example_parameters(config_filename, rundir)
    aliases = {
        'alpha_B_coefficient': 'hydrogen_alpha_B',
        'sigma_gamma': 'hydrogen_sigma_gamma',
        'epsilon_gamma': 'hydrogen_epsilon_gamma',
        'source_photon_rate': 'radiative_transfer_source_photon_rate',
    }
    for alias, source in aliases.items():
        if alias not in runparams and source in runparams:
            runparams[alias] = runparams[source]
    config_dir = config_filename.resolve().parent
    for key in (
        'density_reference_filename',
        'velocity_reference_filename',
        'pressure_reference_filename',
        'neutral_fraction_reference_filename',
    ):
        if key in icparams:
            value = Path(icparams[key])
            if not value.is_absolute():
                icparams[key] = str(config_dir / value)
    if clean_previous_outputs:
        eu.clean_previous_outputs(runparams)
    return runparams, icparams


def build_static_problem(config):
    code_units_obj = CodeUnits.from_mapping(config.get('CodeUnits'))
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=config['number_of_cells'],
        noghost=2,
        boxsize=config['boxsize'],
        verbose=config.get('verbose', 0),
        area=config['area'],
        EOStype='polytropic',
        gamma=5.0 / 3.0,
        CFL=config.get('hydro_cfl', config['CFL']),
        order=0,
        dtmin=config['dtmin'],
        dtmax=config['dtmax'],
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_xHI_initial=1.0,
        hydrogen_xHI_inflow=1.0,
        hydrogen_xHI_outflow=1.0,
        hydrogen_source_CFL=config.get(
            'source_cfl', config.get('hydrogen_source_CFL', 0.1)
        ),
        hydrogen_source_dtmin=config['hydrogen_source_dtmin'],
        hydrogen_update_mu=True,
        hydrogen_thermal_coupling=True,
        hydrogen_recombination=True,
        hydrogen_collisional_ionization=False,
        hydrogen_alpha_B=config['hydrogen_alpha_B'],
        hydrogen_beta=config['hydrogen_beta'],
        hydrogen_radiation_field=False,
        hydrogen_radiation_evolution=False,
        hydrogen_ngamma_initial=config['hydrogen_ngamma_initial'],
        hydrogen_sigma_gamma=config['hydrogen_sigma_gamma'],
        hydrogen_epsilon_gamma=config['hydrogen_epsilon_gamma'],
        radiative_transfer=True,
        radiative_transfer_method='long_characteristics',
        radiative_transfer_temporal_scheme=config.get(
            'radiative_transfer_temporal_scheme', 'instantaneous'
        ),
        radiative_transfer_c2ray_max_iterations=config.get(
            'radiative_transfer_c2ray_max_iterations', 32
        ),
        radiative_transfer_c2ray_tolerance=config.get(
            'radiative_transfer_c2ray_tolerance', 1.0e-6
        ),
        radiative_transfer_c2ray_relaxation=config.get(
            'radiative_transfer_c2ray_relaxation', 1.0
        ),
        radiative_transfer_c2ray_nonconvergence=config.get(
            'radiative_transfer_c2ray_nonconvergence', 'warn'
        ),
        radiative_transfer_boundary_flux=config['radiative_transfer_boundary_flux'],
        radiative_transfer_source_photon_rate=config['radiative_transfer_source_photon_rate'],
        radiative_transfer_direction=1,
        CodeUnits=code_units_obj,
        units=SimpleNamespace(CodeUnits=code_units_obj),
        unit_system=code_units_obj.unit_system,
    )
    par.simulation = SimpleNamespace(
        current_time=0.0 * unyt.Myr,
        box_size=config['boxsize'],
        coordinate_system=par.coordsys,
    )
    par.mesh = SimpleNamespace(
        grid_cells=par.nogrid,
        ghost_cells=par.noghost,
    )

    mesh = Mesh()
    mesh.boundary = np.linspace(
        0.0,
        config['boxsize'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm

    fluid = Fluid()
    fluid.eos = EOS(par.EOStype, par.gamma, code_units_obj)
    fluid.rho_code = (
        np.ones(par.nogrid)
        * config['hydrogen_number_density']
        * unyt.mp
    ).to(unyt.g / unyt.cm**3)
    fluid.vel_code = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp_code = np.ones(par.nogrid) * config['initial_temperature']
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.SetFluidTime(0.0 * unyt.Myr)

    solver = Solver()
    return par, mesh, fluid, solver


build_problem = build_static_problem


def write_initial_condition(config, runparams):
    """Build and write the initial-condition snapshot."""
    par, mesh, fluid, _ = build_static_problem(config)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    Path(runparams['ICfilename']).unlink(missing_ok=True)
    rio.writehdf5(sim, runparams['ICfilename'])


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_static_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    if getattr(par, 'noghost', 0) > 0:
        mesh.boundary = np.asarray(mesh.boundary[par.noghost : -par.noghost], dtype=float)
    mesh.SetUpMesh(par)
    fluid.SetPressure()
    return par, mesh, fluid


def output_files(outdir, outfileprefix):
    pattern = os.path.join(outdir, f'{outfileprefix}_*.hdf5')
    return sorted(glob.glob(pattern))


def interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def ionization_front_position(
    mesh,
    fluid,
    par,
    neutral_fraction=IONIZATION_FRONT_NEUTRAL_FRACTION,
):
    interior = interior_slice(par)
    radius = _to_kpc(mesh.coordinate[interior], par)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)

    if np.all(xHI > neutral_fraction):
        return 0.0
    if np.all(xHI <= neutral_fraction):
        return radius[-1]

    # The front is the first outward transition from ionized gas
    # (xHI <= 0.5) to neutral gas (xHI > 0.5). This remains well-defined
    # when the profile contains additional ionized pockets farther out.
    crossings = np.where(
        (xHI[:-1] <= neutral_fraction) & (xHI[1:] > neutral_fraction)
    )[0]
    if crossings.size == 0:
        return radius[np.where(xHI <= neutral_fraction)[0][-1]]

    left = int(crossings[0])
    right = left + 1
    weight = (neutral_fraction - xHI[left]) / (xHI[right] - xHI[left])
    return radius[left] + weight * (radius[right] - radius[left])


def mean_ionized_temperature(fluid, par):
    interior = interior_slice(par)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)
    temperature = _to_temperature(fluid.temp_code[interior], par)
    ionized_weight = 1.0 - xHI
    if np.sum(ionized_weight) <= 0.0:
        return 0.0
    return float(np.sum(ionized_weight * temperature) / np.sum(ionized_weight))


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(_to_myr(fluid.time, par))
    history['front_radius_kpc'].append(
        ionization_front_position(
            mesh,
            fluid,
            par,
            neutral_fraction=IONIZATION_FRONT_NEUTRAL_FRACTION,
        )
    )
    history['mean_ionized_temperature_K'].append(mean_ionized_temperature(fluid, par))


def load_history_from_outputs(outputfilenames, config):
    history = {
        'time_Myr': [],
        'front_radius_kpc': [],
        'mean_ionized_temperature_K': [],
    }
    for outputfilename in outputfilenames:
        par, mesh, fluid = load_output_state(outputfilename, config)
        append_history(history, mesh, fluid, par)
    return history


def stromgren_radius(config):
    radius = (
        3.0
        * config['source_photon_rate']
        / (
            4.0
            * np.pi
            * config['alpha_B_coefficient']
            * config['hydrogen_number_density']**2
        )
    ) ** (1.0 / 3.0)
    return radius.to(unyt.kpc)


def recombination_time(config):
    return (
        1.0
        / (config['hydrogen_number_density'] * config['alpha_B_coefficient'])
    ).to(unyt.Myr)


def ionized_sound_speed_from_history(history, gamma):
    """Return the Spitzer ionized-gas sound speed at 10^4 K.

    ``history`` is retained in the signature for compatibility with existing
    callers, but the analytic Spitzer comparison must not depend on the
    simulated final temperature.
    """
    del history
    temperature = 1.0e4 * unyt.K
    mu_ionized = 0.5
    return np.sqrt(gamma * unyt.kboltz * temperature / (mu_ionized * unyt.mp)).to(
        unyt.km / unyt.s
    )


def spitzer_radius(time, config, ci):
    radius_stromgren = stromgren_radius(config)
    factor = (
        1.0
        + 7.0
        * ci.to(unyt.cm / unyt.s)
        * time.to(unyt.s)
        / (4.0 * radius_stromgren.to(unyt.cm))
    )
    return (radius_stromgren * factor**(4.0 / 7.0)).to(unyt.kpc)


def shifted_spitzer_radius(time, config, ci):
    time_since_recombination = time - recombination_time(config)
    return spitzer_radius(time_since_recombination, config, ci)


def load_reference_profile(filename, radius_unit, log_value=False):
    if filename is None or not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    value = data[:, 1]
    if log_value:
        value = 10.0**value
    return {
        'radius_kpc': data[:, 0] * radius_unit.to_value(unyt.kpc),
        'value': value,
    }


def scatter_reference(ax, reference, label='ZEUS-MP'):
    if reference is None:
        return
    ax.scatter(
        reference['radius_kpc'],
        reference['value'],
        s=20,
        color='black',
        marker='o',
        facecolors='none',
        label=label,
    )


def save_front_plot(history, config, figure_filename):
    time = np.asarray(history['time_Myr']) * unyt.Myr
    front_radius = np.asarray(history['front_radius_kpc'])
    radius_stromgren = stromgren_radius(config)
    tau_recombination = recombination_time(config)
    ci = ionized_sound_speed_from_history(history, 5.0 / 3.0)
    spitzer_valid = time >= tau_recombination
    radius_spitzer = None
    if np.any(spitzer_valid):
        radius_spitzer = shifted_spitzer_radius(
            time[spitzer_valid],
            config,
            ci,
        ).to_value(unyt.kpc)
    plot_radius_max = config['plot_radius_max'].to_value(unyt.kpc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time.to_value(unyt.Myr),
        front_radius,
        color='tab:blue',
        lw=2.0,
        label=r'RadHydropy $x_{\rm HI}=0.5$',
    )
    if radius_spitzer is not None:
        ax.plot(
            time[spitzer_valid].to_value(unyt.Myr),
            radius_spitzer,
            color='black',
            lw=1.7,
            ls='--',
            label=(
                r'Spitzer after $\tau_{\rm rec}$, '
                r'$c_i=%.1f$ km s$^{-1}$'
                % ci.to_value(unyt.km / unyt.s)
            ),
        )
    ax.axvline(
        tau_recombination.to_value(unyt.Myr),
        color='0.45',
        lw=1.2,
        ls='-.',
        label=r'$\tau_{\rm rec}=%.1f$ Myr' % tau_recombination.to_value(unyt.Myr),
    )
    ax.axhline(
        radius_stromgren.to_value(unyt.kpc),
        color='0.3',
        lw=1.4,
        ls=':',
        label=r'$R_{\rm S}$',
    )
    ax.set_xlim(0.0, time[-1].to_value(unyt.Myr))
    if radius_spitzer is not None:
        ax.set_ylim(0.0, max(plot_radius_max, 1.05 * np.nanmax(radius_spitzer)))
    else:
        ax.set_ylim(0.0, plot_radius_max)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Ionization-front radius [kpc]')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_plot(mesh, fluid, par, config, figure_filename):
    interior = interior_slice(par)
    radius_pc = _to_kpc(mesh.coordinate[interior], par) * (1.0 * unyt.kpc).to_value(unyt.pc)
    number_density = _to_number_density(fluid.rho_code[interior], par)
    velocity = _to_km_s(fluid.vel_code[interior], par)
    neutral_fraction = np.asarray(fluid.xHI[interior], dtype=float)
    pressure = _to_pressure(fluid.pre_code[interior], par)
    temperature = _to_temperature(fluid.temp_code[interior], par)
    plot_radius_max = config['plot_radius_max'].to_value(unyt.pc)
    radius_unit = config.get('reference_radius_unit', 15.0 * unyt.kpc)
    density_reference = load_reference_profile(
        config.get('density_reference_filename', None),
        radius_unit,
        log_value=True,
    )
    velocity_reference = load_reference_profile(
        config.get('velocity_reference_filename', None),
        radius_unit,
        log_value=False,
    )
    pressure_reference = load_reference_profile(
        config.get('pressure_reference_filename', None),
        radius_unit,
        log_value=True,
    )
    neutral_fraction_reference = load_reference_profile(
        config.get('neutral_fraction_reference_filename', None),
        radius_unit,
        log_value=True,
    )

    reference_radius_scale = (1.0 * unyt.kpc).to_value(unyt.pc)
    for reference in (
        density_reference,
        velocity_reference,
        pressure_reference,
        neutral_fraction_reference,
    ):
        if reference is not None:
            reference['radius_kpc'] *= reference_radius_scale

    fig, axes = plt.subplots(5, 1, figsize=(7.4, 11.0), sharex=True)
    axes[0].plot(radius_pc, number_density, color='tab:blue', lw=1.8, label='RadHydropy')
    scatter_reference(axes[0], density_reference)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$n$ [cm$^{-3}$]')
    axes[0].legend(frameon=False, loc='best')

    positive_velocity = np.where(velocity > 0.0, velocity, np.nan)
    axes[1].plot(radius_pc, positive_velocity, color='tab:orange', lw=1.8, label='RadHydropy')
    scatter_reference(axes[1], velocity_reference)
    axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=0.5)
    axes[1].set_ylabel(r'$v_r$ [km s$^{-1}$]')
    axes[1].legend(frameon=False, loc='best')

    axes[2].plot(
        radius_pc,
        np.clip(neutral_fraction, 1.0e-8, 1.0),
        color='tab:green',
        lw=1.8,
        label='RadHydropy',
    )
    scatter_reference(axes[2], neutral_fraction_reference)
    axes[2].set_yscale('log')
    axes[2].set_ylabel(r'$x_{\rm HI}$')
    axes[2].legend(frameon=False, loc='best')

    axes[3].plot(radius_pc, pressure, color='tab:red', lw=1.8, label='RadHydropy')
    scatter_reference(axes[3], pressure_reference)
    axes[3].set_yscale('log')
    axes[3].set_ylabel(r'$P$ [g cm$^{-1}$ s$^{-2}$]')
    axes[3].legend(frameon=False, loc='best')

    axes[4].plot(radius_pc, temperature, color='tab:purple', lw=1.8, label='RadHydropy')
    axes[4].set_yscale('log')
    axes[4].set_ylabel(r'$T$ [K]')
    axes[4].set_xlabel('Radius [pc]')
    axes[4].legend(frameon=False, loc='best')

    for ax in axes:
        ax.set_xlim(0.0, plot_radius_max)
        ax.grid(True, which='both', alpha=0.25)
    final_time_myr = config['timesim'].to_value(unyt.Myr)
    fig.suptitle(
        'Dynamic photoheated Stromgren sphere at %.3g Myr' % final_time_myr
    )
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
