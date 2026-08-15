"""Helper utilities for the spherical radiative-transfer example."""

from types import SimpleNamespace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unyt
import numpy as np

import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs
import radiative_transfer_analytic as rta


PC_IN_CM = unyt.unyt_quantity(1.0, unyt.pc).to_value(unyt.cm)


def build_static_problem(config):
    code_units_obj = CodeUnits.from_mapping(config.get('CodeUnits'))
    par = SimpleNamespace(
        coordsys=config.get('coordsys', 'spherical'),
        boundcond=config.get('boundcond', 'OpenSph'),
        nogrid=config['number_of_cells'],
        noghost=config.get('noghost', 2),
        boxsize=config['boxsize'],
        verbose=config.get('verbose', 0),
        outdir=config.get('outdir', '.'),
        outfileprefix=config.get('outfileprefix', 'Output'),
        savedir=config.get('savedir', config.get('outdir', '.')),
        area=config.get('area', 1.0 * unyt.cm**2),
        hydrogen_chemistry=config.get('hydrogen_chemistry', False),
        thermochemistry_network=config.get('thermochemistry_network', 'hydrogen'),
        hydrogen_mass_fraction=config.get('hydrogen_mass_fraction', 1.0),
        hydrogen_recombination=config.get('hydrogen_recombination', True),
        hydrogen_collisional_ionization=config.get('hydrogen_collisional_ionization', True),
        hydrogen_thermal_coupling=config.get('hydrogen_thermal_coupling', True),
        hydrogen_ngamma_initial=config.get('hydrogen_ngamma_initial', 0.0 / unyt.cm**3),
        hydrogen_sigma_gamma=config.get('hydrogen_sigma_gamma', 0.0 * unyt.cm**2),
        radiative_transfer=config.get('radiative_transfer', True),
        radiative_transfer_method=config.get('radiative_transfer_method', 'long_characteristics'),
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
        radiative_transfer_boundary_flux=config.get(
            'radiative_transfer_boundary_flux',
            0.0 / (unyt.cm**2 * unyt.s),
        ),
        radiative_transfer_source_photon_rate=config.get(
            'radiative_transfer_source_photon_rate',
            config.get('source_photon_rate', 0.0 / unyt.s),
        ),
        radiative_transfer_direction=config.get('radiative_transfer_direction', 1),
        CodeUnits=code_units_obj,
        unit_system=code_units_obj.unit_system,
    )

    mesh = SimpleNamespace()
    mesh.boundary = np.linspace(0.0, config['boxsize'].to_value(unyt.cm), par.nogrid + 1) * unyt.cm

    fluid = SimpleNamespace()
    fluid.rho = np.ones(par.nogrid) * unyt.mp / unyt.cm**3
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * unyt.K
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.time = 0.0 * unyt.s

    solver = SimpleNamespace()
    return par, mesh, fluid, solver


build_problem = build_static_problem


def _refresh_mesh_geometry(mesh, par):
    mesh.xdelta = mesh.boundary[1:] - mesh.boundary[:-1]
    mesh.oneoverdx = 1.0 / mesh.xdelta
    if par.coordsys == 'cartesian':
        mesh.coordinate = 0.5 * (mesh.boundary[1:] + mesh.boundary[:-1])
        if hasattr(par, 'area'):
            mesh.area = np.ones(len(mesh.xdelta)) * par.area
        else:
            mesh.area = np.ones(len(mesh.xdelta))
        mesh.vol = mesh.xdelta * mesh.area
    elif par.coordsys == 'spherical':
        mesh.area = (mesh.boundary[:-1] ** 2) * 4.0 * np.pi
        mesh.vol = np.absolute((mesh.boundary[1:] ** 3 - mesh.boundary[:-1] ** 3)) * 4.0 * np.pi / 3.0
        vol_denom = mesh.boundary[1:] ** 3 - mesh.boundary[:-1] ** 3
        mesh.coordinate = 0.5 * (mesh.boundary[1:] + mesh.boundary[:-1])
        nonzero_vol_denom = vol_denom != 0.0
        mesh.coordinate[nonzero_vol_denom] = 0.75 * (
            mesh.boundary[1:][nonzero_vol_denom] ** 4 - mesh.boundary[:-1][nonzero_vol_denom] ** 4
        ) / vol_denom[nonzero_vol_denom]
        for ig in range(len(mesh.vol)):
            if (mesh.boundary[ig] < 0.0) and (mesh.boundary[ig + 1] > 0.0):
                mesh.vol[ig] = (mesh.boundary[ig + 1] ** 3) * 4.0 * np.pi / 3.0
                mesh.coordinate[ig] = 0.75 * mesh.boundary[ig + 1]
                mesh.area[ig] = 0.0
    else:
        raise ValueError("coordsys unknown: %s" % par.coordsys)


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_static_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    _refresh_mesh_geometry(mesh, par)
    return par, mesh, fluid


def write_initial_condition(config, runparams):
    par, mesh, fluid, _ = build_static_problem(config)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    icfilename = Path(runparams['ICfilename'])
    icfilename.unlink(missing_ok=True)
    rio.writehdf5(sim, icfilename)


def save_plot(mesh, fluid, par, source_photon_rate, figure_filename, code_units=None):
    code_units_obj = CodeUnits.from_mapping(code_units)
    interior = slice(par.noghost, par.noghost + par.nogrid)
    radius_values = mesh.coordinate[interior]
    if hasattr(radius_values, 'to_value'):
        radius = radius_values.to_value(unyt.pc) * unyt.pc
    else:
        radius = (
            code_quantity_to_cgs(radius_values, code_units_obj, 'length_cm')
            / PC_IN_CM
        ) * unyt.pc
    simulated_values = fluid.ngamma[interior]
    if hasattr(simulated_values, 'to_value'):
        simulated = simulated_values.to_value(1.0 / unyt.cm**3) * (1.0 / unyt.cm**3)
    else:
        simulated = (
            code_quantity_to_cgs(simulated_values, code_units_obj, 'number_density_cm3')
            * (1.0 / unyt.cm**3)
        )
    analytic_fv = rta.finite_volume_density(
        mesh.boundary[interior.start : interior.stop + 1],
        mesh.vol[interior],
        source_photon_rate,
        code_units=code_units_obj,
    )

    r_min = mesh.boundary[interior.start + 1]
    r_max = mesh.boundary[interior.stop]
    radius_line = np.geomspace(
        float(
            np.asarray(
                code_quantity_to_cgs(r_min, code_units_obj, 'length_cm'),
                dtype=float,
            )
            / PC_IN_CM
        )
        if not hasattr(r_min, 'to_value')
        else float(np.asarray(r_min.to_value(unyt.pc), dtype=float)),
        float(
            np.asarray(
                code_quantity_to_cgs(r_max, code_units_obj, 'length_cm'),
                dtype=float,
            )
            / PC_IN_CM
        )
        if not hasattr(r_max, 'to_value')
        else float(np.asarray(r_max.to_value(unyt.pc), dtype=float)),
        512,
    ) * unyt.pc
    analytic_point = rta.point_density(
        radius_line,
        source_photon_rate,
        code_units=code_units_obj,
    )

    simulated_cgs = simulated.to_value(1.0 / unyt.cm**3)
    analytic_cgs = analytic_fv.to_value(1.0 / unyt.cm**3)
    relative_error = np.max(np.abs((simulated_cgs - analytic_cgs) / analytic_cgs))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(
        radius.to_value(unyt.pc),
        simulated.to_value(1.0 / unyt.cm**3),
        marker='o',
        ms=3.0,
        lw=0.0,
        label=(
            'RadHydropy C²-Ray'
            if getattr(par, 'radiative_transfer_temporal_scheme', 'instantaneous') == 'c2ray'
            else 'RadHydropy long characteristic'
        ),
    )
    ax.plot(
        radius.to_value(unyt.pc),
        analytic_fv.to_value(1.0 / unyt.cm**3),
        color='black',
        lw=2.0,
        label='Analytic finite-volume average',
    )
    ax.plot(
        radius_line.to_value(unyt.pc),
        analytic_point.to_value(1.0 / unyt.cm**3),
        color='tab:orange',
        ls='--',
        lw=1.5,
        label=r'$Q/(4\pi r^2 c)$',
    )
    ax.text(
        0.04,
        0.06,
        'max relative error = %.2e' % relative_error,
        transform=ax.transAxes,
    )
    ax.set_xlabel('Radius [pc]')
    ax.set_ylabel(r'Photon number density [cm$^{-3}$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
    return relative_error
