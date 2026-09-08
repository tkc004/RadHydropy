"""Helper utilities for the spherical radiative-transfer example."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unyt
import numpy as np

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import code_quantity_to_cgs, quantity_to_value
import radiative_transfer_analytic as rta


PC_IN_CM = unyt.unyt_quantity(1.0, unyt.pc).to_value(unyt.cm)


def build_static_problem(config):
    par_config = config['par']
    initial = config['initial_condition']
    grid_cells = int(par_config['mesh']['grid_cells'])
    sim = Rsim(par_config)
    code_units = sim.par.units.CodeUnits
    sim.par.simulation.box_size = quantity_to_value(
        initial['boxsize'], code_units.length_unit
    )
    sim.par.simulation.time_proper_code = 0.0
    sim.mesh.boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(0.0, initial['boxsize'].to_value(unyt.cm), grid_cells + 1) * unyt.cm,
        code_units.length_unit,
    ))
    boundary_proper_code = sim.mesh.boundary_proper_code
    width_proper_code = np.diff(boundary_proper_code)
    volume_proper_code = 4.0 * np.pi / 3.0 * (
        boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
    )
    coordinate_proper_code = 0.75 * (
        boundary_proper_code[1:] ** 4 - boundary_proper_code[:-1] ** 4
    ) / (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    area_proper_code = 4.0 * np.pi * boundary_proper_code[:-1] ** 2
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=coordinate_proper_code,
        boundary_proper_code=boundary_proper_code,
        width_proper_code=width_proper_code,
        area_proper_code=area_proper_code,
        volume_proper_code=volume_proper_code,
    )
    sim.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * unyt.mp / unyt.cm**3,
        code_units.density_unit,
    ))
    sim.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells, dtype=float))
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * unyt.K,
        code_units.temperature_unit,
    ))
    sim.fluid.mu = np.ones(grid_cells)
    sim.fluid.xHI = np.ones(grid_cells)
    sim.fluid.ngamma_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * sim.par.radiation.hydrogen_ngamma_initial,
        code_units.number_density_unit,
    ))
    sim.fluid.SetFluidTime(0.0)
    sim.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    sim.fluid.SetPressure()
    sim.fluid._refresh_runtime_state()
    return sim.par, sim.mesh, sim.fluid, sim.solver


build_problem = build_static_problem


def _refresh_mesh_geometry(mesh, par):
    mesh.width_proper_code = (
        mesh.boundary_proper_code[1:] - mesh.boundary_proper_code[:-1]
    )
    mesh.coordinate_inverse_proper_code = 1.0 / mesh.width_proper_code
    if par.simulation.coordinate_system == 'cartesian':
        mesh.x_proper_code = 0.5 * (mesh.boundary_proper_code[1:] + mesh.boundary_proper_code[:-1])
        if getattr(par.mesh, 'area', None) is not None:
            mesh.area_proper_code = np.ones(len(mesh.width_proper_code)) * quantity_to_value(par.mesh.area, par.units.CodeUnits.area_unit)
        else:
            mesh.area_proper_code = np.ones(len(mesh.width_proper_code))
        mesh.volume_proper_code = mesh.width_proper_code * mesh.area_proper_code
    elif par.simulation.coordinate_system == 'spherical':
        mesh.area_proper_code = (mesh.boundary_proper_code[:-1] ** 2) * 4.0 * np.pi
        mesh.volume_proper_code = np.absolute((mesh.boundary_proper_code[1:] ** 3 - mesh.boundary_proper_code[:-1] ** 3)) * 4.0 * np.pi / 3.0
        vol_denom = mesh.boundary_proper_code[1:] ** 3 - mesh.boundary_proper_code[:-1] ** 3
        mesh.x_proper_code = 0.5 * (mesh.boundary_proper_code[1:] + mesh.boundary_proper_code[:-1])
        nonzero_vol_denom = vol_denom != 0.0
        mesh.x_proper_code[nonzero_vol_denom] = 0.75 * (
            mesh.boundary_proper_code[1:][nonzero_vol_denom] ** 4 - mesh.boundary_proper_code[:-1][nonzero_vol_denom] ** 4
        ) / vol_denom[nonzero_vol_denom]
        for ig in range(len(mesh.volume_proper_code)):
            if (mesh.boundary_proper_code[ig] < 0.0) and (mesh.boundary_proper_code[ig + 1] > 0.0):
                mesh.volume_proper_code[ig] = (mesh.boundary_proper_code[ig + 1] ** 3) * 4.0 * np.pi / 3.0
                mesh.x_proper_code[ig] = 0.75 * mesh.boundary_proper_code[ig + 1]
                mesh.area_proper_code[ig] = 0.0
    else:
        raise ValueError("coordinate system unknown: %s" % par.simulation.coordinate_system)
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=mesh.x_proper_code,
        boundary_proper_code=mesh.boundary_proper_code,
        width_proper_code=mesh.width_proper_code,
        area_proper_code=mesh.area_proper_code,
        volume_proper_code=mesh.volume_proper_code,
    )


def load_output_state(outputfilename, config):
    par, mesh, fluid, _ = build_static_problem(config)
    rio.readhdf5(par, mesh, fluid, outputfilename)
    _refresh_mesh_geometry(mesh, par)
    return par, mesh, fluid


def write_initial_condition(config):
    par, mesh, fluid, solver = build_static_problem(config)
    icfilename = Path(config['par']['simulation']['initial_condition_filename'])
    rio.writehdf5(Rsim.FromComponents(par, mesh, fluid, solver), icfilename)


def save_plot(mesh, fluid, par, config, figure_filename):
    radiation = config['par']['radiation']
    source_photon_rate = radiation['radiative_transfer_source_photon_rate']
    code_units_obj = par.units.CodeUnits
    interior = slice(par.mesh.ghost_cells, par.mesh.ghost_cells + par.mesh.grid_cells)
    radius_values = mesh.x_proper_code[interior]
    if hasattr(radius_values, 'to_value'):
        radius = radius_values.to_value(unyt.pc) * unyt.pc
    else:
        radius = (
            code_quantity_to_cgs(radius_values, code_units_obj, 'length_cgs_cm')
            / PC_IN_CM
        ) * unyt.pc
    simulated_values = fluid.ngamma_code[interior]
    if hasattr(simulated_values, 'to_value'):
        simulated = simulated_values.to_value(1.0 / unyt.cm**3) * (1.0 / unyt.cm**3)
    else:
        simulated = (
            code_quantity_to_cgs(simulated_values, code_units_obj, 'number_density_cgs_cm3')
            * (1.0 / unyt.cm**3)
        )
    analytic_fv = rta.finite_volume_density(
        mesh.boundary_proper_code[interior.start : interior.stop + 1],
        mesh.volume_proper_code[interior],
        source_photon_rate,
        code_unit_system=code_units_obj,
    )

    r_min = mesh.boundary_proper_code[interior.start + 1]
    r_max = mesh.boundary_proper_code[interior.stop]
    radius_line = np.geomspace(
        float(
            np.asarray(
                code_quantity_to_cgs(r_min, code_units_obj, 'length_cgs_cm'),
                dtype=float,
            )
            / PC_IN_CM
        )
        if not hasattr(r_min, 'to_value')
        else float(np.asarray(r_min.to_value(unyt.pc), dtype=float)),
        float(
            np.asarray(
                code_quantity_to_cgs(r_max, code_units_obj, 'length_cgs_cm'),
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
        code_unit_system=code_units_obj,
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
        if par.radiation.radiative_transfer_temporal_scheme == 'c2ray'
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
