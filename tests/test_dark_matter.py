import numpy as np

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


def code_units():
    return CodeUnits.from_mapping(
        {
            "UnitMass_in_cgs": 1.0e33,
            "UnitLength_in_cgs": 1.0e18,
            "UnitVelocity_in_cgs": 1.0e5,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        }
    )


def test_shells_sort_all_state_arrays_together():
    shells = DarkMatterShells(
        radius=[3.0, 1.0, 2.0],
        velocity=[30.0, 10.0, 20.0],
        mass=[3.0, 1.0, 2.0],
        angular_momentum=[0.3, 0.1, 0.2],
        code_units=code_units(),
    )
    assert np.allclose(shells.radius, [1.0, 2.0, 3.0])
    assert np.allclose(shells.velocity, [10.0, 20.0, 30.0])
    assert np.allclose(shells.mass, [1.0, 2.0, 3.0])
    assert np.allclose(shells.angular_momentum, [0.1, 0.2, 0.3])


def test_enclosed_mass_uses_half_shell_at_equal_radius():
    shells = DarkMatterShells(
        radius=[1.0, 2.0, 3.0],
        velocity=[0.0, 0.0, 0.0],
        mass=[1.0, 2.0, 3.0],
        code_units=code_units(),
    )
    assert np.allclose(shells.enclosed_mass([0.5, 1.0, 2.0, 2.5, 4.0]), [0.0, 0.5, 2.0, 3.0, 6.0])


def test_acceleration_includes_gravity_and_angular_momentum():
    units = code_units()
    shells = DarkMatterShells(
        radius=[2.0],
        velocity=[0.0],
        mass=[3.0],
        angular_momentum=[4.0],
        code_units=units,
    )
    g_code = GRAVITATIONAL_CONSTANT_CGS * units.mass_in_cgs / (
        units.length_in_cgs * units.velocity_in_cgs**2
    )
    expected = -0.5 * g_code * 3.0 / 2.0**2 + 4.0**2 / 2.0**3
    assert np.allclose(shells.acceleration(), expected)


def test_crossing_timestep_is_finite_for_closing_shells():
    shells = DarkMatterShells(
        radius=[1.0, 2.0],
        velocity=[1.0, -1.0],
        mass=[1.0, 1.0],
        code_units=code_units(),
    )
    assert np.isclose(shells.crossing_timestep(safety_factor=0.1), 0.05)


def test_step_preserves_mass_and_sorting_through_crossing():
    shells = DarkMatterShells(
        radius=[1.0, 2.0],
        velocity=[1.0, -1.0],
        mass=[1.0, 1.0],
        code_units=code_units(),
    )
    total_mass = shells.total_mass
    shells.step(0.6)
    assert np.isclose(shells.total_mass, total_mass)
    assert np.all(np.diff(shells.radius) >= 0.0)


def test_fixed_enclosed_mass_ignores_test_shell_mass():
    units = code_units()
    shells = DarkMatterShells(
        radius=[2.0],
        velocity=[0.0],
        mass=[1.0e-12],
        angular_momentum=[0.0],
        fixed_enclosed_mass=3.0,
        code_units=units,
    )
    g_code = GRAVITATIONAL_CONSTANT_CGS * units.mass_in_cgs / (
        units.length_in_cgs * units.velocity_in_cgs**2
    )
    assert np.allclose(shells.acceleration(), -g_code * 3.0 / 2.0**2)
