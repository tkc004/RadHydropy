"""Helper utilities for the spherical radiative-transfer example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unyt
import numpy as np

import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits
import radiative_transfer_analytic as rta


PC_IN_CM = unyt.unyt_quantity(1.0, unyt.pc).to_value(unyt.cm)


def build_static_problem(config):
    initial = config['initial_condition']
    grid_cells = int(config["par"]['mesh']['grid_cells'])
    code_units = config['_code_units']
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(
        np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * (unyt.mp / unyt.cm**3)
    )
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * unyt.K)
    writer.fluid.ngamma_radarray = writer.radarray(
        np.ones(grid_cells) * config['par']['thermochemistry']['hydrogen_ngamma_initial'],
    )
    writer.fluid.mu = np.ones(grid_cells)
    writer.simulation.fluid.xHI = np.ones(grid_cells)
    return writer


def load_output_state(output_filename, config):
    return rio.loadhdf5(config, str(output_filename))


def write_initial_condition(config):
    writer = build_static_problem(config)
    writer.write(config['par']['simulation']['initial_condition_filename'], validate=True)


def save_plot(snapshot, config, figure_filename):
    par = snapshot.par
    radiation = config['par']['radiation']
    source_photon_rate = radiation['source_photon_rate']
    boundary_proper_unyt = snapshot.mesh.boundary_radarray.to(unyt.cm)
    interior = slice(par.mesh.ghost_cells, par.mesh.ghost_cells + par.mesh.grid_cells)
    active_boundary_proper_unyt = boundary_proper_unyt[interior.start:interior.stop + 1]
    inner_radius_proper_unyt = active_boundary_proper_unyt[:-1]
    outer_radius_proper_unyt = active_boundary_proper_unyt[1:]
    volume_proper_unyt = 4.0 * np.pi / 3.0 * (
        outer_radius_proper_unyt**3 - inner_radius_proper_unyt**3
    )
    radius_proper_unyt = 0.75 * (
        outer_radius_proper_unyt**4 - inner_radius_proper_unyt**4
    ) / (outer_radius_proper_unyt**3 - inner_radius_proper_unyt**3)
    simulated = snapshot.fluid.ngamma_radarray[interior].to(1.0 / unyt.cm**3)
    analytic_fv = rta.finite_volume_density(
        active_boundary_proper_unyt,
        volume_proper_unyt,
        source_photon_rate,
        code_unit_system=config['_code_units'],
    )

    r_min = active_boundary_proper_unyt[1]
    r_max = active_boundary_proper_unyt[-1]
    radius_line = np.geomspace(
        float(r_min.to_value(unyt.pc)),
        float(r_max.to_value(unyt.pc)),
        512,
    ) * unyt.pc
    analytic_point = rta.point_photon_number_density(
        radius_line,
        source_photon_rate,
        code_unit_system=config['_code_units'],
    )

    simulated_cgs = simulated.to_value(1.0 / unyt.cm**3)
    analytic_cgs = analytic_fv.to_value(1.0 / unyt.cm**3)
    relative_error = np.max(np.abs((simulated_cgs - analytic_cgs) / analytic_cgs))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(
        radius_proper_unyt.to_value(unyt.pc),
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
        radius_proper_unyt.to_value(unyt.pc),
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
