"""Focused checks for the boundary-driven NFW virial-shock example."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import unyt

EXAMPLE = Path(__file__).parents[1] / 'example' / 'NFWBoundaryDrivenVirialShock1D'
if str(EXAMPLE.parent) not in sys.path:
    sys.path.insert(0, str(EXAMPLE.parent))
import example_utils
SPEC = importlib.util.spec_from_file_location('nfw_boundary_shock_tools', EXAMPLE / 'tools.py')
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOLS)


def test_inflow_density_conserves_prescribed_mass_flux():
    mdot = 30.0 * unyt.Msun / unyt.yr
    radius = 400.0 * unyt.kpc
    velocity = -200.0 * unyt.km / unyt.s
    density = TOOLS.inflow_density(mdot, radius, velocity)
    recovered = 4.0 * np.pi * radius**2 * density * abs(velocity)
    assert np.isclose((recovered / mdot).to_value('dimensionless'), 1.0)


def test_shock_locator_selects_hot_compressed_inner_state():
    radius = np.geomspace(5.0, 500.0, 128)
    density = np.ones(128) * 1.0e-28
    temperature = np.ones(128) * 1.0e4
    density[radius < 180.0] *= 4.0
    temperature[radius < 180.0] *= 40.0
    snapshot = {
        'radius_kpc': radius,
        'density_cgs_g_cm3': density,
        'temperature_cgs_K': temperature,
    }
    index = TOOLS.locate_shock(snapshot, 200.0)
    assert index is not None
    assert np.isclose(radius[index], 180.0, rtol=0.06)


def test_shock_locator_rejects_cold_inner_cooling_front():
    radius = np.geomspace(5.0, 500.0, 128)
    density = np.ones(128) * 1.0e-28
    temperature = np.ones(128) * 2.0e5
    density[radius < 180.0] *= 4.0
    temperature[radius < 180.0] = 1.0e4
    snapshot = {
        'radius_kpc': radius,
        'density_cgs_g_cm3': density,
        'temperature_cgs_K': temperature,
        'velocity_km_s': np.ones(128) * -100.0,
    }
    assert TOOLS.locate_shock(snapshot, 200.0) is None


def test_lower_mass_configs_use_isolated_output_directories():
    masses = []
    output_directories = []
    for name in (
        'nfw_boundary_driven_virial_shock1d.yaml',
        'nfw_boundary_driven_virial_shock_3e11.yaml',
        'nfw_boundary_driven_virial_shock_1e11.yaml',
    ):
        config = example_utils.load_nested_example_config(EXAMPLE / name)
        masses.append(config['initial_condition']['halo_mass'].to_value(unyt.Msun))
        output_directories.append(config['par']['output']['directory'])
    assert np.allclose(masses, [1.0e12, 3.0e11, 1.0e11])
    assert len(set(output_directories)) == 3


def test_long_case_requests_high_cadence_and_longer_pie_stage():
    config = example_utils.load_nested_example_config(
        EXAMPLE / 'nfw_boundary_driven_virial_shock_1e11_long.yaml'
    )
    assert np.isclose(config['example']['pie_final_time'].to_value(unyt.Myr), 3200.0)
    schedule = EXAMPLE / config['example']['pie_outputtimefilename']
    with schedule.open(encoding='utf-8') as stream:
        values = [float(line) for line in stream if line.strip() and line.strip() != 'Myr']
    assert len(values) == 65
    assert np.isclose(np.diff(values).min(), 50.0)


def test_massive_long_case_targets_1e13_msun():
    config = example_utils.load_nested_example_config(
        EXAMPLE / 'nfw_boundary_driven_virial_shock_1e13_long.yaml'
    )
    assert np.isclose(
        config['initial_condition']['halo_mass'].to_value(unyt.Msun), 1.0e13
    )
