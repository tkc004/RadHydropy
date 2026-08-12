import numpy as np
import h5py
import tempfile
from pathlib import Path

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.dark_matter import enclosed_gas_mass
from radhydropy.gravity import Gravity
import radhydropy.io as rio
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


def test_callable_fixed_enclosed_mass_supports_analytic_backgrounds():
    units = code_units()
    shells = DarkMatterShells(
        radius=[2.0],
        velocity=[0.0],
        mass=[1.0e-12],
        angular_momentum=[0.0],
        fixed_enclosed_mass=lambda radius: 3.0 + np.asarray(radius) ** 3,
        code_units=units,
    )
    g_code = GRAVITATIONAL_CONSTANT_CGS * units.mass_in_cgs / (
        units.length_in_cgs * units.velocity_in_cgs**2
    )
    assert np.allclose(shells.acceleration(), -g_code * 11.0 / 2.0**2)


class Mesh:
    coordsys = "spherical"
    boundary = np.array([0.0, 1.0, 2.0, 3.0])
    coordinate = np.array([0.75, 1.5, 2.5])
    vol = 4.0 * np.pi / 3.0 * np.diff(boundary**3)


class Par:
    CodeUnits = code_units()
    noghost = 0
    nogrid = 3


def test_enclosed_gas_mass_handles_partial_cells():
    mesh = Mesh()
    rho = np.ones(3)
    expected = 4.0 * np.pi / 3.0 * np.array([0.5**3, 1.5**3, 3.0**3])
    assert np.allclose(enclosed_gas_mass(mesh, rho, [0.5, 1.5, 4.0], Par), expected)


def test_dark_matter_field_is_added_to_gas_gravity():
    units = code_units()
    dm = DarkMatterShells(
        radius=[1.0], velocity=[0.0], mass=[2.0], code_units=units
    )
    gravity = Gravity(dark_matter=dm, code_units=units)
    acceleration = gravity.acceleration_on_mesh(Mesh(), rho=np.ones(3), par=Par)
    g_code = GRAVITATIONAL_CONSTANT_CGS * units.mass_in_cgs / (
        units.length_in_cgs * units.velocity_in_cgs**2
    )
    assert np.allclose(acceleration, [-0.0, -g_code * 2.0 / 1.5**2, -g_code * 2.0 / 2.5**2])


def test_dark_matter_shell_force_includes_enclosed_gas_mass():
    units = code_units()
    dm = DarkMatterShells(
        radius=[2.0], velocity=[0.0], mass=[1.0], code_units=units
    )
    gravity = Gravity(dark_matter=dm, code_units=units)
    gas_mass = 4.0 * np.pi / 3.0 * 2.0**3
    acceleration = gravity.dark_matter.acceleration(
        gas_enclosed_mass=np.array([gas_mass])
    )
    g_code = GRAVITATIONAL_CONSTANT_CGS * units.mass_in_cgs / (
        units.length_in_cgs * units.velocity_in_cgs**2
    )
    expected = -g_code * (0.5 + gas_mass) / 2.0**2
    assert np.allclose(acceleration, expected)


def test_dark_matter_snapshot_group_is_written():
    units = code_units()
    dm = DarkMatterShells(
        radius=[1.0, 2.0], velocity=[0.0, 0.0], mass=[1.0, 1.0],
        angular_momentum=[0.1, 0.2], code_units=units,
    )
    class Fluid:
        rho = np.ones(2)
        vel = np.zeros(2)
        temp = np.ones(2)
        mu = np.ones(2)
    class MeshForIO:
        boundary = np.array([0.0, 1.0, 2.0])
    class ParForIO:
        def __init__(self):
            self.CodeUnits = units
            self.time = np.array([0.0])
            self.boxsize = np.array([2.0])
            self.dark_matter = dm
    class State:
        par = ParForIO()
        mesh = MeshForIO()
        fluid = Fluid()
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / "snapshot.hdf5"
        rio.writehdf5(State(), filename)
        with h5py.File(filename, "r") as handle:
            assert "DarkMatter" in handle
            assert set(handle["DarkMatter"]) == {
                "Radius", "RadialVelocity", "Mass", "SpecificAngularMomentum"
            }


def test_dark_matter_snapshot_reconstructs_live_shells():
    units = code_units()
    dm = DarkMatterShells(
        radius=[1.0, 2.0], velocity=[0.3, -0.2], mass=[1.0, 2.0],
        angular_momentum=[0.1, 0.2], softening=0.05, code_units=units,
    )

    class Fluid:
        rho = np.ones(2)
        vel = np.zeros(2)
        temp = np.ones(2)
        mu = np.ones(2)

    class MeshForIO:
        boundary = np.array([0.0, 1.0, 2.0])

    class ParForIO:
        def __init__(self):
            self.CodeUnits = units
            self.time = np.array([0.0])
            self.boxsize = np.array([2.0])
            self.dark_matter = dm

    class State:
        par = ParForIO()
        mesh = MeshForIO()
        fluid = Fluid()

    class LoadedPar:
        coordsys = None
        nogrid = None

    class LoadedMesh:
        pass

    class LoadedFluid:
        pass

    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / "snapshot.hdf5"
        rio.writehdf5(State(), filename)
        loaded_par = LoadedPar()
        rio.readhdf5(loaded_par, LoadedMesh(), LoadedFluid(), filename)

    reconstructed = loaded_par.dark_matter
    assert isinstance(reconstructed, DarkMatterShells)
    assert np.allclose(reconstructed.radius, dm.radius)
    assert np.allclose(reconstructed.velocity, dm.velocity)
    assert np.allclose(reconstructed.mass, dm.mass)
    assert np.allclose(reconstructed.angular_momentum, dm.angular_momentum)
    assert np.isclose(reconstructed.softening, dm.softening)
    assert loaded_par.dark_matter_snapshot["mass"].shape == dm.mass.shape
