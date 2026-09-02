import numpy as np
import h5py
import tempfile
from pathlib import Path

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.dark_matter import enclosed_gas_mass
from radhydropy.dark_matter import prepare_enclosed_gas_mass
from radhydropy.gravity import Gravity
import radhydropy.io as rio
from radhydropy.units import CodeUnits
from radhydropy.cosmology import EinsteinDeSitter
from types import SimpleNamespace


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


def test_sort_skips_already_sorted_shells():
    shells = DarkMatterShells(
        radius=[1.0, 2.0, 2.0, 4.0],
        velocity=[10.0, 20.0, 30.0, 40.0],
        mass=[1.0, 2.0, 3.0, 4.0],
        angular_momentum=[0.1, 0.2, 0.3, 0.4],
        code_units=code_units(),
    )
    radius_before = shells.radius.copy()

    order = shells.sort_by_radius()

    assert np.array_equal(order, np.arange(4))
    assert np.array_equal(shells.radius, radius_before)
    assert np.array_equal(shells.velocity, [10.0, 20.0, 30.0, 40.0])


def test_sort_is_stable_for_equal_radii():
    shells = DarkMatterShells(
        radius=[2.0, 1.0, 2.0],
        velocity=[20.0, 10.0, 21.0],
        mass=[2.0, 1.0, 3.0],
        code_units=code_units(),
    )

    assert np.array_equal(shells.radius, [1.0, 2.0, 2.0])
    assert np.array_equal(shells.velocity, [10.0, 20.0, 21.0])
    assert np.array_equal(shells.mass, [1.0, 2.0, 3.0])


def test_sort_invalidates_mass_prefix_cache_after_reordering():
    shells = DarkMatterShells(
        radius=[1.0, 2.0],
        velocity=[0.0, 0.0],
        mass=[1.0, 2.0],
        code_units=code_units(),
    )
    shells.enclosed_mass()
    assert shells._mass_prefix_cache is not None

    shells.radius[:] = [2.0, 1.0]
    shells.sort_by_radius()

    assert shells._mass_prefix_cache is None
    assert np.allclose(shells.enclosed_mass(), [1.0, 2.5])


def test_enclosed_mass_uses_half_shell_at_equal_radius():
    shells = DarkMatterShells(
        radius=[1.0, 2.0, 3.0],
        velocity=[0.0, 0.0, 0.0],
        mass=[1.0, 2.0, 3.0],
        code_units=code_units(),
    )
    assert np.allclose(shells.enclosed_mass([0.5, 1.0, 2.0, 2.5, 4.0]), [0.0, 0.5, 2.0, 3.0, 6.0])


def test_enclosed_mass_at_shell_radii_matches_generic_lookup():
    shells = DarkMatterShells(
        radius=[1.0, 1.0, 2.0, 4.0],
        velocity=[0.0, 0.0, 0.0, 0.0],
        mass=[1.0, 2.0, 3.0, 4.0],
        code_units=code_units(),
    )
    expected = shells.enclosed_mass(shells.radius)
    assert np.allclose(shells.enclosed_mass(), expected)


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
    advanced = shells.step(1.0)
    assert np.isclose(advanced, 1.0)
    assert np.isclose(shells.total_mass, total_mass)
    assert np.all(np.diff(shells.radius) >= 0.0)


def test_coincident_crossing_does_not_return_zero_timestep():
    shells = DarkMatterShells(
        radius=[1.0, 1.0],
        velocity=[1.0, -1.0],
        mass=[1.0, 2.0],
        angular_momentum=[0.1, 0.2],
        code_units=code_units(),
    )
    assert np.isinf(shells.crossing_timestep(safety_factor=1.0))
    shells.step(0.1, crossing_safety_factor=0.5)
    assert np.all(np.isfinite(shells.radius))
    assert np.all(np.diff(shells.radius) >= 0.0)
    assert np.isclose(shells.total_mass, 3.0)


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


def test_inward_shell_is_absorbed_into_softened_core():
    shells = DarkMatterShells(
        radius=[0.8, 2.0],
        velocity=[-2.0, 0.0],
        mass=[3.0, 5.0],
        softening=0.5,
        fixed_enclosed_mass=1000.0,
        central_core_radius=0.5,
        core_absorption_velocity=0.0,
        code_units=code_units(),
    )
    shells.step(0.2)
    assert shells.number_of_shells == 1
    assert np.isclose(shells.central_core_mass, 1003.0)
    assert np.isclose(shells.fixed_enclosed_mass, 1003.0)
    assert np.isclose(shells.total_mass, 5.0)


def test_unbound_outward_shell_is_not_absorbed_at_core_boundary():
    shells = DarkMatterShells(
        radius=[0.5],
        velocity=[1.0],
        mass=[3.0],
        fixed_enclosed_mass=1.0,
        central_core_radius=0.5,
        code_units=code_units(),
    )
    shells.step(0.0)
    assert shells.number_of_shells == 1
    assert np.isclose(shells.central_core_mass, 1.0)


def test_bound_outward_shell_is_absorbed_by_energy_criterion():
    shells = DarkMatterShells(
        radius=[0.5],
        velocity=[0.01],
        mass=[3.0],
        fixed_enclosed_mass=10.0,
        central_core_radius=0.6,
        code_units=code_units(),
    )
    shells.step(1.0e-8)
    assert shells.number_of_shells == 0
    assert np.isclose(shells.central_core_mass, 13.0)


class Mesh:
    coordsys = "spherical"
    boundary = np.array([0.0, 1.0, 2.0, 3.0])
    coordinate = np.array([0.75, 1.5, 2.5])
    vol = 4.0 * np.pi / 3.0 * np.diff(boundary**3)


class Par:
    CodeUnits = code_units()
    noghost = 0
    nogrid = 3
    mesh = SimpleNamespace(ghost_cells=0, grid_cells=3)


def test_enclosed_gas_mass_handles_partial_cells():
    mesh = Mesh()
    rho = np.ones(3)
    expected = 4.0 * np.pi / 3.0 * np.array([0.5**3, 1.5**3, 3.0**3])
    assert np.allclose(enclosed_gas_mass(mesh, rho, [0.5, 1.5, 4.0], Par), expected)


def test_prepared_enclosed_gas_mass_matches_direct_evaluation():
    mesh = Mesh()
    rho = np.array([1.0, 2.0, 3.0])
    radius = np.array([0.25, 0.5, 1.0, 1.75, 2.5, 4.0])
    profile = prepare_enclosed_gas_mass(mesh, rho, Par)
    assert np.allclose(profile(radius), enclosed_gas_mass(mesh, rho, radius, Par))


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


def test_cosmological_shell_force_subtracts_background_and_scales_with_a():
    units = code_units()
    cosmology = EinsteinDeSitter.from_code_units(units)
    shells = DarkMatterShells(
        radius=[2.0], velocity=[0.0], mass=[5.0], code_units=units,
    )
    gas_mass = 7.0
    background_mass = 3.0
    g_code = GRAVITATIONAL_CONSTANT_CGS * units.mass_in_cgs / (
        units.length_in_cgs * units.velocity_in_cgs**2
    )
    acceleration = shells.acceleration(
        gas_enclosed_mass=np.array([gas_mass]),
        background_enclosed_mass=np.array([background_mass]),
        scale_factor=2.5,
        cosmological=True,
    )
    expected = -g_code * 2.5 * (0.5 * 5.0 + gas_mass - background_mass) / 2.0**2
    assert np.allclose(acceleration, expected)


def test_dark_matter_snapshot_group_is_written():
    units = code_units()
    dm = DarkMatterShells(
        radius=[1.0, 2.0], velocity=[0.0, 0.0], mass=[1.0, 1.0],
        angular_momentum=[0.1, 0.2], code_units=units,
    )
    class Fluid:
        rho_code = np.ones(2)
        vel_code = np.zeros(2)
        temp_code = np.ones(2)
        mu = np.ones(2)
    class MeshForIO:
        boundary = np.array([0.0, 1.0, 2.0])
    class ParForIO:
        def __init__(self):
            self.CodeUnits = units
            self.time = np.array([0.0])
            self.boxsize = np.array([2.0])
            self.dark_matter = dm
            self.mesh = SimpleNamespace(ghost_cells=0, grid_cells=2)
            self.simulation = SimpleNamespace(box_size=self.boxsize, current_time=self.time)
            self.units = SimpleNamespace(CodeUnits=units)
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
        rho_code = np.ones(2)
        vel_code = np.zeros(2)
        temp_code = np.ones(2)
        mu = np.ones(2)

    class MeshForIO:
        boundary = np.array([0.0, 1.0, 2.0])

    class ParForIO:
        def __init__(self):
            self.CodeUnits = units
            self.time = np.array([0.0])
            self.boxsize = np.array([2.0])
            self.dark_matter = dm
            self.mesh = SimpleNamespace(ghost_cells=0, grid_cells=2)
            self.simulation = SimpleNamespace(box_size=self.boxsize, current_time=self.time)
            self.units = SimpleNamespace(CodeUnits=units)

    class State:
        par = ParForIO()
        mesh = MeshForIO()
        fluid = Fluid()

    class LoadedPar:
        coordsys = None
        nogrid = None
        mesh = SimpleNamespace(ghost_cells=0, grid_cells=None)
        simulation = SimpleNamespace(coordinate_system=None)

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
