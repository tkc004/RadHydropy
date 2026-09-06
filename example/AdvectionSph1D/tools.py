"""Helper utilities for the spherical advection example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

import radhydropy.io as rio
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)
from radhydropy.units import CodeUnits, quantity_to_value
import advection_sph_analytic as asa


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    icparams = config['initial_condition']
    code_units = config['_code_units']
    sim = SimpleNamespace()
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    if code_units is not None:
        sim.par.unit_system = code_units.unit_system

    grid_cells = icparams['grid_cells']
    box_size = np.asarray(
        [quantity_to_value(icparams['box_size'], code_units.length_unit)],
        dtype=float,
    )
    sim.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
    sim.par.simulation = SimpleNamespace(
        coordinate_system=icparams['coordinate_system'],
        time_proper_code=np.asarray(
            [quantity_to_value(icparams['current_time'], code_units.time_unit)]
        ),
        box_size=box_size,
    )

    dx = box_size[0] / grid_cells
    sim.mesh.boundary_proper_code = np.linspace(
        dx,
        box_size[0] + dx,
        grid_cells + 1,
    )
    coordinate = 0.5 * (
        sim.mesh.boundary_proper_code[1:]
        + sim.mesh.boundary_proper_code[:-1]
    )
    sim.mesh.x_proper_code = coordinate
    sim.mesh.width_proper_code = np.diff(sim.mesh.boundary_proper_code)
    sim.mesh.area_proper_code = 4.0 * np.pi * sim.mesh.boundary_proper_code[:-1] ** 2
    sim.mesh.volume_proper_code = np.absolute(
        sim.mesh.boundary_proper_code[1:] ** 3
        - sim.mesh.boundary_proper_code[:-1] ** 3
    ) * 4.0 * np.pi / 3.0

    sim.fluid.vel_proper_code = np.full(
        grid_cells,
        quantity_to_value(icparams['initial_velocity'], code_units.velocity_unit),
    )
    sim.fluid.temp_proper_code = np.full(
        grid_cells,
        quantity_to_value(icparams['initial_temperature'], code_units.temperature_unit),
    )
    rho = np.full(
        grid_cells,
        quantity_to_value(icparams['initial_density'], code_units.density_unit),
    )
    rho[
        np.logical_or(
            coordinate < 0.25 * box_size[0],
            coordinate > 0.75 * box_size[0],
        )
    ] *= 0.01
    sim.fluid.rho_proper_code = rho
    sim.fluid.mu = np.ones(grid_cells) * icparams['mean_molecular_weight']
    sim.par.time_proper_code = sim.par.simulation.time_proper_code
    sim.mesh.geometry_state = MeshGeometryState(
        x_proper_code=sim.mesh.x_proper_code,
        boundary_proper_code=sim.mesh.boundary_proper_code,
        width_proper_code=sim.mesh.width_proper_code,
        area_proper_code=sim.mesh.area_proper_code,
        volume_proper_code=sim.mesh.volume_proper_code,
    )
    sim.fluid.time_proper_code = float(sim.par.time_proper_code[0])
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        density=sim.fluid.rho_proper_code,
        velocity=sim.fluid.vel_proper_code,
        pressure=np.zeros_like(sim.fluid.rho_proper_code),
        temperature=sim.fluid.temp_proper_code,
        time=sim.fluid.time_proper_code,
        mu=sim.fluid.mu,
    )


    return sim

def ReadandPlot(outfilename, config, **kwargs):
    icparams = config['initial_condition']
    runparams = config['par']
    rout = build_initial_condition(config)
    code_units_obj = config['_code_units']
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    boundary_proper_code = np.asarray(rout.mesh.boundary_proper_code, dtype=float)
    volume_proper_code = np.absolute(
        boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
    ) * 4.0 * np.pi / 3.0
    x_proper_code = 0.5 * (
        boundary_proper_code[1:] + boundary_proper_code[:-1]
    )
    time = rout.par.simulation.time_proper_code * code_units_obj.time_unit
    radius = (
        x_proper_code[first:last] * code_units_obj.length_unit
    )
    plt.plot(
        radius,
        rout.fluid.rho_proper_code[first:last],
        ls=kwargs.get('ls', 'none'),
        marker=kwargs.get('marker', 'o'),
        mfc=kwargs.get('mfc', 'none'),
        markevery=kwargs.get('markevery', 10),
        color=kwargs.get('color'),
    )
    mtot = np.sum(
        rout.fluid.rho_proper_code[first:last]
        * volume_proper_code[first:last]
    )
    print('mtot', mtot)
    x = x_proper_code[first:last] * code_units_obj.length_unit
    rho = asa.top_hat_density_profile(
        radius,
        time,
        icparams['initial_velocity'],
        icparams['box_size'],
        icparams['initial_density'],
    )
    plt.plot(x, rho, color=kwargs['color'], ls='solid')
