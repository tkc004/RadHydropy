"""Helper utilities for the spherical inflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
import inflow_sph_analytic as ia


def _active_radarray(values, active_cells, ghost_cells, *, boundary=False):
    """Return the active portion of an IC or snapshot RadArray."""
    expected_active = active_cells + 1 if boundary else active_cells
    values_size = int(np.asarray(values).size)
    if values_size == expected_active:
        return values
    expected_ghosted = expected_active + 2 * ghost_cells
    if values_size == expected_ghosted:
        start = ghost_cells
        stop = start + expected_active
        return values[start:stop]
    raise ValueError(
        f"unexpected {'boundary' if boundary else 'fluid'} RadArray size "
        f"{values_size}; expected {expected_active} or {expected_ghosted}"
    )


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(initial['grid_cells'])
    box_size_proper_unyt = initial['box_size_proper']
    boundary_proper_unyt = (
        np.linspace(0.0, 1.0, grid_cells + 1) * box_size_proper_unyt
    )
    writer = InitialConditionWriter(
        ic_config=config["initial_condition"],
        par_config=config['par'],
        code_units=code_units,
    )
    writer.box_size = writer.radquantity(box_size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial['rho_proper']
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.ones(grid_cells) * initial['vel_proper']
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.fluid.mu = np.full(
        grid_cells, float(initial['mean_molecular_weight'])
    )
    return writer


def plot_snapshot(outfilename, config, **kwargs):
    initial = config['initial_condition']
    snapshot = rio.loadhdf5(config, outfilename)
    code_units = config['_code_units']
    active_cells = int(snapshot.par.mesh.grid_cells)
    ghost_cells = int(snapshot.par.mesh.ghost_cells)
    boundary_proper_code = _active_radarray(
        snapshot.mesh.boundary_radarray,
        active_cells,
        ghost_cells,
        boundary=True,
    ).to_value(code_units.length_unit)
    rho_proper_code = _active_radarray(
        snapshot.fluid.rho_radarray,
        active_cells,
        ghost_cells,
    ).to_value(code_units.density_unit)
    x_proper_code = 0.5 * (
        boundary_proper_code[:-1] + boundary_proper_code[1:]
    )
    time_proper_code = float(
        np.asarray(snapshot.fluid.runtime_state.time_proper_code).flat[0]
    )
    time_proper_unyt = time_proper_code * code_units.time_unit
    plt.plot(
        x_proper_code * code_units.length_unit,
        rho_proper_code * code_units.density_unit,
        **kwargs,
    )
    plt.ylim(ymax=10.1)
    plt.axvline(
        x=ia.front_position_proper_unyt(
            initial['box_size_proper'],
            time_proper_unyt,
            config["par"]['boundary']['vel_inflow_proper'],
        ),
        color=kwargs['color'],
        ls='dashed',
    )
    rho_analytic_proper_unyt = ia.density_profile_proper_unyt(
        x_proper_code * code_units.length_unit,
        config["par"]['boundary']['rho_inflow_proper'],
        initial['box_size_proper'],
    )
    plt.plot(
        x_proper_code * code_units.length_unit,
        rho_analytic_proper_unyt,
        ls='dashed',
        color='k',
    )
