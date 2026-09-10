"""Dynamic Stromgren helper with the StellarWindBubble inner boundary."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import unyt

from radhydropy import io as rio
from radhydropy.units import quantity_to_value


TEMPLATE_TOOLS = (
    Path(__file__).resolve().parents[1]
    / 'DynamicStromgrenSpherePhotoheating20pc1D'
    / 'tools.py'
)
spec = importlib.util.spec_from_file_location(
    '_radhydropy_dynamic_stromgren_20pc_tools',
    TEMPLATE_TOOLS,
)
_template = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = _template
spec.loader.exec_module(_template)

for _name, _value in vars(_template).items():
    if not _name.startswith('__'):
        globals()[_name] = _value

_BASE_BUILD_STATIC_PROBLEM = _template.build_static_problem


def _wind_density(config):
    """Return the inner-boundary density implied by the requested mass loss."""
    example = config['example']
    mass_loss_rate = example['wind_mass_loss_rate_proper'].to(unyt.g / unyt.s)
    wind_velocity_proper_unyt = example['wind_velocity_proper'].to(unyt.cm / unyt.s)
    injection_radius_proper_cgs_cm_unyt = example['radius_injection_proper'].to(unyt.cm)
    return mass_loss_rate / (
        4.0 * np.pi * injection_radius_proper_cgs_cm_unyt**2 * wind_velocity_proper_unyt
    )


def build_static_problem(config):
    """Build the photoheated ambient cloud with a stellar-wind inner boundary."""
    sim = _BASE_BUILD_STATIC_PROBLEM(config)
    par, mesh, fluid, solver = sim.par, sim.mesh, sim.fluid, sim.solver
    initial = config['initial_condition']
    example = config['example']
    par.boundary.condition = 'OutflowSph'
    par.boundary.rho_outflow_proper = _wind_density(config)
    par.boundary.vel_outflow_proper = example['wind_velocity_proper']
    par.boundary.temperature_outflow_proper = example['wind_temperature_proper']
    par.boundary.outflow_mu = example['wind_mu']

    box_size_proper_cgs_cm = initial['box_size_proper'].to_value(unyt.cm)
    injection_radius_proper_cgs_cm = example['radius_injection_proper'].to_value(unyt.cm)
    boundary_proper_cgs_cm_unyt = np.linspace(
        injection_radius_proper_cgs_cm,
        injection_radius_proper_cgs_cm + box_size_proper_cgs_cm,
        config['par']['mesh']['grid_cells'] + 1,
    ) * unyt.cm
    mesh.boundary_proper_code = quantity_to_value(
        boundary_proper_cgs_cm_unyt, par.units.CodeUnits.length_unit
    )
    mesh.width_proper_code = np.diff(mesh.boundary_proper_code)
    mesh.area_proper_code = 4.0 * np.pi * mesh.boundary_proper_code[:-1] ** 2
    mesh.volume_proper_code = (
        4.0 * np.pi / 3.0
        * (
            mesh.boundary_proper_code[1:] ** 3
            - mesh.boundary_proper_code[:-1] ** 3
        )
    )
    mesh.x_proper_code = 0.75 * (
        mesh.boundary_proper_code[1:] ** 4
        - mesh.boundary_proper_code[:-1] ** 4
    ) / (
        mesh.boundary_proper_code[1:] ** 3
        - mesh.boundary_proper_code[:-1] ** 3
    )
    wind_cells = min(int(example.get('wind_injection_cells', 2)), int(par.mesh.grid_cells))
    if wind_cells > 0:
        first_active = int(par.mesh.ghost_cells)
        active_slice = slice(first_active, first_active + wind_cells)
        cell_center_proper_code = 0.5 * (
            mesh.boundary_proper_code[first_active:first_active + wind_cells]
            + mesh.boundary_proper_code[first_active + 1:first_active + wind_cells + 1]
        )
        injection_radius_proper_code = quantity_to_value(
            example['radius_injection_proper'], par.units.CodeUnits.length_unit
        )
        wind_density_proper_code = quantity_to_value(
            _wind_density(config), par.units.CodeUnits.density_unit
        )
        wind_velocity_proper_code = quantity_to_value(
            example['wind_velocity_proper'], par.units.CodeUnits.velocity_unit
        )
        wind_temperature_proper_code = quantity_to_value(
            example['wind_temperature_proper'], par.units.CodeUnits.temperature_unit
        )
        fluid.rho_proper_code[active_slice] = wind_density_proper_code * (
            injection_radius_proper_code / cell_center_proper_code
        ) ** 2
        fluid.vel_proper_code[active_slice] = wind_velocity_proper_code
        fluid.temp_proper_code[active_slice] = wind_temperature_proper_code
        fluid.mu[active_slice] = example['wind_mu']
    fluid.SetUpFluid(par, mesh)
    solver.SetConserved(mesh, fluid, verbose=0)
    _template._attach_proper_runtime_states(par, mesh, fluid)
    return sim


def write_initial_condition(config):
    """Write an IC file with the wind boundary parameters in its header."""
    sim = build_static_problem(config)
    filename = config['par']['simulation']['initial_condition_filename']
    Path(filename).unlink(missing_ok=True)
    rio.writehdf5(sim, filename)


_template.build_static_problem = build_static_problem
_template.write_initial_condition = write_initial_condition
