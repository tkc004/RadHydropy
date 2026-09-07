"""Initial-condition builder for the passive gas angular-momentum example."""

import numpy as np

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)
from radhydropy.units import quantity_to_value


def build_initial_condition(config):
    """Build the active proper-code state from the complete nested config."""
    par_config = config['par']
    initial = config['initial_condition']
    sim = Rsim(par_config)
    code_units = sim.par.units.CodeUnits
    grid_cells = int(par_config['mesh']['grid_cells'])
    box_size_proper_code = quantity_to_value(
        initial['box_size'], code_units.length_unit
    )
    sim.par.simulation.box_size = box_size_proper_code
    sim.par.simulation.time_proper_code = quantity_to_value(
        initial['current_time'], code_units.time_unit
    )
    boundary_proper_code = as_named_array(
        np.linspace(0.0, box_size_proper_code, grid_cells + 1)
    )
    coordinate_proper_code = 0.5 * (
        boundary_proper_code[1:] + boundary_proper_code[:-1]
    )
    width_proper_code = np.diff(boundary_proper_code)
    sim.mesh.boundary_proper_code = boundary_proper_code
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=coordinate_proper_code,
        boundary=boundary_proper_code,
        width=width_proper_code,
        area=np.ones(grid_cells),
        volume=width_proper_code,
    )
    sim.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.full(grid_cells, initial['initial_density']),
        code_units.density_unit,
    ))
    sim.fluid.vel_proper_code = as_named_array(quantity_to_value(
        np.full(grid_cells, initial['velocity']),
        code_units.velocity_unit,
    ))
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.full(grid_cells, initial['temperature']),
        code_units.temperature_unit,
    ))
    sim.fluid.mu = as_named_array(
        np.full(grid_cells, float(initial['mean_molecular_weight']))
    )
    phase_dimensionless = (
        2.0 * np.pi * coordinate_proper_code / box_size_proper_code
    )
    angular_momentum_unit = code_units.length_unit * code_units.velocity_unit
    angular_momentum_offset_code = quantity_to_value(
        initial['angular_momentum_offset'], angular_momentum_unit
    )
    angular_momentum_amplitude_code = quantity_to_value(
        initial['angular_momentum_amplitude'], angular_momentum_unit
    )
    if initial.get('include_angular_momentum', True):
        sim.fluid.specific_angular_momentum_code = as_named_array(
            angular_momentum_offset_code
            + angular_momentum_amplitude_code * np.sin(phase_dimensionless)
        )
    sim.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    sim.fluid.SetFluidTime(sim.par.simulation.time_proper_code)
    sim.fluid.SetPressure()
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        density=sim.fluid.rho_proper_code,
        velocity=sim.fluid.vel_proper_code,
        pressure=sim.fluid.pre_proper_code,
        temperature=sim.fluid.temp_proper_code,
        time=sim.fluid.time_proper_code,
        mu=sim.fluid.mu,
    )
    return sim
