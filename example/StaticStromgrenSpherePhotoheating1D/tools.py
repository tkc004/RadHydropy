"""Helper utilities for the photoheated static Stromgren sphere example."""

import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.radiative_transfer as rrt
from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
import radhydropy.io as rio
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
from radhydropy.units import (
    CodeUnits,
    _as_cgs_float,
)

static_stromgren_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'StaticStromgrenSphere1D')
)
if static_stromgren_dir not in sys.path:
    sys.path.append(static_stromgren_dir)

import stromgren_analytic as sa


def build_static_problem(config):
    code_units = CodeUnits.from_mapping(config.get('CodeUnits'))
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=config['number_of_cells'],
        noghost=2,
        boxsize=config['boxsize'],
        verbose=config.get('verbose', 0),
        outdir=config.get('outdir', '.'),
        outfileprefix=config.get('outfileprefix', 'Output'),
        savedir=config.get('savedir', config.get('outdir', '.')),
        area=config['area'],
        EOStype='polytropic',
        gamma=5.0 / 3.0,
        dtmin=config['dtmin'],
        dtmax=config['dtmax'],
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_xHI_initial=1.0,
        hydrogen_xHI_inflow=1.0,
        hydrogen_xHI_outflow=1.0,
        hydrogen_source_CFL=config.get('evolution_timestep_cfl', 0.1),
        hydrogen_source_dtmin=config['hydrogen_source_dtmin'],
        hydrogen_update_mu=True,
        hydrogen_thermal_coupling=True,
        hydrogen_recombination=True,
        hydrogen_collisional_ionization=False,
        hydrogen_alpha_B=config['alpha_B_coefficient'],
        hydrogen_beta=config['hydrogen_beta'],
        hydrogen_radiation_field=False,
        hydrogen_radiation_evolution=False,
        hydrogen_ngamma_initial=config['hydrogen_ngamma_initial'],
        hydrogen_sigma_gamma=config['sigma_gamma'],
        hydrogen_epsilon_gamma=config['epsilon_gamma'],
        radiative_transfer=True,
        radiative_transfer_method='long_characteristics',
        radiative_transfer_boundary_flux=config['radiative_transfer_boundary_flux'],
        radiative_transfer_source_photon_rate=config['source_photon_rate'],
        radiative_transfer_direction=1,
        CodeUnits=code_units,
        unit_system=code_units.unit_system,
    )

    mesh = Mesh()
    mesh.boundary = np.linspace(
        0.0,
        config['boxsize'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm
    mesh.SetUpMesh(par)

    fluid = Fluid()
    fluid.eos = EOS(par.EOStype, par.gamma, code_units)
    fluid.rho = (
        np.ones(par.nogrid)
        * config['hydrogen_number_density']
        * unyt.mp
    ).to(unyt.g / unyt.cm**3)
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * config['initial_temperature']
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.SetUpFluid(par)
    fluid.SetFluidTime(0.0 * unyt.Myr)

    solver = Solver()
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid, verbose=getattr(par, 'verbose', 0))
    result = rrt.trace_long_characteristics(
        mesh,
        fluid.rho,
        fluid.xHI,
        hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
        sigma_gamma=_as_cgs_float(
            getattr(par, 'hydrogen_sigma_gamma', rrt.rh.DEFAULT_SIGMA_GAMMA),
            unyt.cm**2,
        ),
        boundary_flux=_as_cgs_float(
            getattr(par, 'radiative_transfer_boundary_flux', 0.0),
            1.0 / (unyt.cm**2 * unyt.s),
        ),
        source_photon_rate=_as_cgs_float(
            getattr(par, 'radiative_transfer_source_photon_rate', 0.0),
            1.0 / unyt.s,
        ),
        direction=getattr(par, 'radiative_transfer_direction', 1),
        coordsys=getattr(mesh, 'coordsys', 'cartesian'),
    )
    fluid.ngamma[:] = np.asarray(result.cell_photon_density, dtype=float)
    return par, mesh, fluid, solver


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_static_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    return par, mesh, fluid


def interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def ionization_front_position(mesh, fluid, par, neutral_fraction=0.5):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to(unyt.kpc)
    xHI = np.asarray(fluid.xHI[interior])

    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0 * unyt.kpc
    if np.all(ionized):
        return mesh.boundary[interior.stop].to(unyt.kpc)

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHI[left]
    x_right = xHI[right]
    if x_right == x_left:
        return radius[left]

    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return radius[left] + weight * (radius[right] - radius[left])


def mean_ionized_temperature(fluid, par):
    interior = interior_slice(par)
    xHI = np.asarray(fluid.xHI[interior])
    temperature = fluid.temp[interior].to_value(unyt.K)
    ionized = 1.0 - xHI
    if np.sum(ionized) <= 0.0:
        return 0.0
    return float(np.sum(ionized * temperature) / np.sum(ionized))


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(fluid.time.to_value(unyt.Myr))
    history['front_radius_kpc'].append(
        ionization_front_position(mesh, fluid, par).to_value(unyt.kpc)
    )
    history['mean_ionized_temp_K'].append(mean_ionized_temperature(fluid, par))


def load_log_reference_profile(filename, radius_unit):
    if filename is None or not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'radius_kpc': data[:, 0] * radius_unit.to_value(unyt.kpc),
        'value': 10.0**data[:, 1],
    }


def save_plot(mesh, fluid, par, history, config, figure_filename):
    interior = interior_slice(par)
    code_units = CodeUnits.from_mapping(config.get('CodeUnits'))
    if hasattr(mesh.coordinate[interior], 'to_value'):
        radius_kpc = mesh.coordinate[interior].to_value(unyt.kpc)
    else:
        radius_kpc = np.asarray(mesh.coordinate[interior], dtype=float)
    radius = radius_kpc * unyt.kpc
    snapshot = history.get('reference_snapshot', None)
    if snapshot is None:
        xHI = np.asarray(fluid.xHI[interior], dtype=float)
        if hasattr(fluid.temp[interior], 'to_value'):
            temperature_K = fluid.temp[interior].to_value(unyt.K)
        else:
            temperature_K = np.asarray(fluid.temp[interior], dtype=float)
        if hasattr(fluid.time, 'to_value'):
            profile_time_Myr = float(fluid.time.to_value(unyt.Myr))
        else:
            profile_time_Myr = float(np.asarray(fluid.time, dtype=float))
    else:
        radius_kpc = snapshot['radius_kpc']
        xHI = snapshot['xHI']
        temperature_K = snapshot['temperature_K']
        profile_time_Myr = snapshot['time_Myr']
    xHII = 1.0 - xHI
    plot_radius_max = config.get('plot_radius_max', config['boxsize']).to_value(unyt.kpc)
    reference_radius_unit = config.get('reference_radius_unit', 5.4 * unyt.kpc)
    temperature_reference = load_log_reference_profile(
        config.get('temperature_reference_filename', None),
        reference_radius_unit,
    )
    neutral_fraction_reference = load_log_reference_profile(
        config.get('neutral_fraction_reference_filename', None),
        reference_radius_unit,
    )
    xHI_analytic = sa.neutral_fraction_profile(
        radius,
        config['hydrogen_number_density'],
        config['sigma_gamma'],
        config['alpha_B_coefficient'],
        config['source_photon_rate'],
        inner_radius=config['analytic_inner_radius'],
    )
    xHII_analytic = 1.0 - xHI_analytic
    radius_stromgren = sa.stromgren_radius(
        config['source_photon_rate'],
        config['hydrogen_number_density'],
        config['alpha_B_coefficient'],
    ).to(unyt.kpc)
    analytic_front = sa.ionization_front_radius(
        np.asarray(history['time_Myr']) * unyt.Myr,
        config['source_photon_rate'],
        config['hydrogen_number_density'],
        config['alpha_B_coefficient'],
    ).to_value(unyt.kpc)

    fig, (ax_frac, ax_temp, ax_front) = plt.subplots(
        3,
        1,
        figsize=(7.4, 8.0),
        gridspec_kw={'height_ratios': [1.4, 1.2, 1.2], 'hspace': 0.34},
    )
    ax_frac.plot(radius_kpc, np.clip(xHI, 1.0e-6, 1.0), label=r'$x_{\rm HI}$')
    ax_frac.plot(radius_kpc, np.clip(xHII, 1.0e-6, 1.0), label=r'$x_{\rm HII}$')
    ax_frac.plot(
        radius_kpc,
        np.clip(xHI_analytic, 1.0e-6, 1.0),
        color='tab:blue',
        lw=1.4,
        ls='--',
        label=r'$x_{\rm HI}$ analytic',
    )
    ax_frac.plot(
        radius_kpc,
        np.clip(xHII_analytic, 1.0e-6, 1.0),
        color='tab:orange',
        lw=1.4,
        ls='--',
        label=r'$x_{\rm HII}$ analytic',
    )
    if neutral_fraction_reference is not None:
        ax_frac.scatter(
            neutral_fraction_reference['radius_kpc'],
            np.clip(neutral_fraction_reference['value'], 1.0e-6, 1.0),
            s=18,
            color='black',
            marker='o',
            facecolors='none',
            label=r'$x_{\rm HI}$ 100 Myr ref.',
        )
    ax_frac.axvline(
        radius_stromgren.to_value(unyt.kpc),
        color='black',
        lw=1.5,
        ls=':',
        label=r'$R_{\rm S}$',
    )
    ax_frac.set_xlim(0.0, plot_radius_max)
    ax_frac.set_ylim(1.0e-6, 1.2)
    ax_frac.set_yscale('log')
    ax_frac.set_ylabel('Hydrogen fraction')
    ax_frac.set_title('Radial profiles at %.0f Myr' % profile_time_Myr)
    ax_frac.grid(True, which='both', alpha=0.25)
    ax_frac.legend(frameon=False, loc='center right')

    ax_temp.plot(radius_kpc, temperature_K, color='tab:red', lw=1.8)
    if temperature_reference is not None:
        ax_temp.scatter(
            temperature_reference['radius_kpc'],
            temperature_reference['value'],
            s=18,
            color='black',
            marker='o',
            facecolors='none',
            label='100 Myr ref.',
        )
    ax_temp.axvline(radius_stromgren.to_value(unyt.kpc), color='black', lw=1.5, ls=':')
    ax_temp.set_xlim(0.0, plot_radius_max)
    ax_temp.set_yscale('log')
    ax_temp.set_ylabel('Temperature [K]')
    ax_temp.grid(True, which='both', alpha=0.25)
    if temperature_reference is not None:
        ax_temp.legend(frameon=False, loc='upper right')

    ax_front.plot(
        history['time_Myr'],
        history['front_radius_kpc'],
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}=0.5$',
    )
    ax_front.plot(
        history['time_Myr'],
        analytic_front,
        color='black',
        lw=1.6,
        ls='--',
        label=r'$R_I(t)$ fixed-$T$ reference',
    )
    ax_front.axhline(radius_stromgren.to_value(unyt.kpc), color='0.25', lw=1.2, ls=':')
    ax_front.set_xlim(0.0, history['time_Myr'][-1])
    ax_front.set_ylim(0.0, plot_radius_max)
    ax_front.set_xlabel('Time [Myr]')
    ax_front.set_ylabel('I-front radius [kpc]')
    ax_front.grid(True, alpha=0.25)
    ax_front.legend(frameon=False, loc='lower right')

    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
