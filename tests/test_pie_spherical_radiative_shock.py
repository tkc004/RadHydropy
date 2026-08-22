"""Focused checks for the gravity-free spherical PIE shock example."""

import importlib.util
from pathlib import Path

import numpy as np


EXAMPLE = Path(__file__).parents[1] / 'example' / 'PIESphericalRadiativeShock1D'
SPEC = importlib.util.spec_from_file_location(
    'pie_spherical_radiative_shock_tools', EXAMPLE / 'tools.py'
)
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOLS)


def test_shock_radius_finds_density_jump():
    radius = np.linspace(1.0, 10.0, 12) * TOOLS.KPC_CM
    density = np.ones(11) * 1.0e-27
    density[5:] = 4.0e-27
    snapshot = {
        'boundary_cm': radius,
        'density_g_cm3': density,
    }
    assert np.isclose(TOOLS.shock_radius(snapshot), 4.68, atol=0.05)


def test_spherical_ic_density_is_mass_density():
    import unyt

    icparams = {
        'nogrid': 4,
        'boxsize': 19.0 * unyt.kpc,
        'time': 0.0 * unyt.Myr,
        'rmin': 1.0 * unyt.kpc,
        'rmax': 20.0 * unyt.kpc,
        'hydrogen_density': 1.0e-3 * unyt.cm**-3,
        'inflow_velocity': -100.0 * unyt.km / unyt.s,
        'outflow_velocity': 100.0 * unyt.km / unyt.s,
        'inflow_temperature': 1.0e4 * unyt.K,
        'muini': 0.62,
    }
    from radhydropy.units import CodeUnits

    code_units = CodeUnits.from_mapping({
        'name': 'galactic_unit_system',
        'InternalUnitSystem': {
            'UnitMass_in_cgs': 1.0e10,
            'UnitLength_in_cgs': 3.08567758e21,
            'UnitVelocity_in_cgs': 1.0e5,
            'UnitCurrent_in_cgs': 1.0,
            'UnitTemp_in_cgs': 1.0,
        },
    })
    initial = TOOLS.Simwrap(icparams, code_units, 0.76)
    assert np.isclose(
        initial.fluid.rho[0].to_value('g/cm**3'),
        1.0e-3 * TOOLS.PROTON_MASS_G / 0.76,
    )
