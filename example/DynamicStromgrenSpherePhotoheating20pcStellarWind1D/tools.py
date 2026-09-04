"""Dynamic Stromgren helper with the StellarWindBubble inner boundary."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import unyt

from radhydropy import io as rio


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
    mass_loss_rate = example['wind_mass_loss_rate'].to(unyt.g / unyt.s)
    wind_velocity = example['wind_velocity'].to(unyt.cm / unyt.s)
    injection_radius = example['rinj'].to(unyt.cm)
    return mass_loss_rate / (
        4.0 * np.pi * injection_radius**2 * wind_velocity
    )


def build_static_problem(config):
    """Build the photoheated ambient cloud with a stellar-wind inner boundary."""
    par, mesh, fluid, solver = _BASE_BUILD_STATIC_PROBLEM(config)
    initial = config['initial_condition']
    example = config['example']
    par.boundcond = 'OutflowSph'
    par.rho_outflow = _wind_density(config)
    par.vel_outflow = example['wind_velocity']
    par.temp_outflow = example['wind_temperature']
    par.mu_outflow = example['wind_mu']

    boxsize_cm = initial['box_size'].to_value(unyt.cm)
    rinj_cm = example['rinj'].to_value(unyt.cm)
    mesh.boundary = np.linspace(
        rinj_cm,
        rinj_cm + boxsize_cm,
        config['par']['mesh']['grid_cells'] + 1,
    ) * unyt.cm
    return par, mesh, fluid, solver


def write_initial_condition(config):
    """Write an IC file with the wind boundary parameters in its header."""
    par, mesh, fluid, _ = build_static_problem(config)
    sim = _template.SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    filename = config['par']['simulation']['initial_condition_filename']
    Path(filename).unlink(missing_ok=True)
    rio.writehdf5(sim, filename)


_template.build_static_problem = build_static_problem
_template.build_problem = build_static_problem
_template.write_initial_condition = write_initial_condition
