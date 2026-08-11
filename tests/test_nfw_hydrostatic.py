import importlib.util
from pathlib import Path

import numpy as np
import unyt


TOOLS_PATH = (
    Path(__file__).resolve().parents[1]
    / 'example'
    / 'NFWHydrostaticEquilibrium1D'
    / 'tools.py'
)
SPEC = importlib.util.spec_from_file_location('nfw_hydrostatic_tools_test', TOOLS_PATH)
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOLS)


def test_nfw_parameters_recover_halo_mass():
    halo = TOOLS.nfw_halo_parameters(1.0e8 * unyt.Msun)
    enclosed = TOOLS.nfw_enclosed_mass(halo['virial_radius'], halo)

    assert np.isclose(enclosed.to_value(unyt.Msun), 1.0e8, rtol=1.0e-12)
    assert np.isclose(halo['scale_radius'].to_value(unyt.kpc), 0.1 * halo['virial_radius'].to_value(unyt.kpc))


def test_hydrostatic_profile_has_the_expected_pressure_gradient():
    halo = TOOLS.nfw_halo_parameters(1.0e8 * unyt.Msun)
    boundaries = np.linspace(0.1, 9.5, 128) * unyt.kpc
    radius = TOOLS.spherical_cell_centers(boundaries)
    temperature = TOOLS.virial_temperature(halo)
    density = TOOLS.hydrostatic_density_profile(
        radius,
        boundaries,
        halo,
        temperature,
        0.59,
    )
    pressure = density * (unyt.kb * temperature / (0.59 * unyt.mp))
    dpressure = np.diff(pressure.to_value(unyt.erg / unyt.cm**3))
    midpoint = 0.5 * (radius[1:] + radius[:-1])
    acceleration = (
        unyt.physical_constants.gravitational_constant
        * TOOLS.nfw_enclosed_mass(midpoint, halo)
        / midpoint**2
    ).to_value(unyt.cm / unyt.s**2)
    expected_gradient = -0.5 * (density[1:] + density[:-1]).to_value(unyt.g / unyt.cm**3) * acceleration * np.diff(radius.to_value(unyt.cm))

    np.testing.assert_allclose(dpressure, expected_gradient, rtol=0.03)
