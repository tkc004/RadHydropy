import numpy as np
import pytest
import unyt
from types import SimpleNamespace

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
from radhydropy.gravity import (
    Gravity,
    nfw_potential,
    point_mass_potential,
    singular_isothermal_potential,
)
from radhydropy.units import CodeUnits
from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM


class DummyMesh:
    def __init__(self):
        self.coordsys = "cartesian"
        self.coordinate = np.array([0.5, 1.5, 2.5], dtype=float)
        self.vol = np.ones(3, dtype=float)


class DummySphericalMesh:
    def __init__(self):
        self.coordsys = "spherical"
        self.boundary = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        self.coordinate = np.array([0.75, 1.5, 2.5], dtype=float)
        self.xdelta = np.ones(3, dtype=float)
        self.vol = 4.0 * np.pi / 3.0 * np.diff(self.boundary**3)


class DummyFluid:
    def __init__(self):
        self.rho_code = np.ones(3, dtype=float)
        self.vel_code = np.array([0.0, 1.0, -2.0], dtype=float)
        self.Mom_code = self.rho_code * self.vel_code
        self.Energy_code = np.zeros(3, dtype=float)


class DummyPar:
    externalgravity = True
    selfgravity = False
    noghost = 0
    nogrid = 3
    mesh = SimpleNamespace(ghost_cells=0, grid_cells=3)


def _code_units():
    return CodeUnits.from_mapping(
        {
            "UnitMass_in_cgs": 2.0,
            "UnitLength_in_cgs": 4.0,
            "UnitVelocity_in_cgs": 8.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        }
    )


def test_gravity_acceleration_from_potential():
    code_units = _code_units()
    coordinate = np.array([0.0, 1.0, 2.0, 3.0], dtype=float) * code_units.length_unit
    potential = 3.0 * np.array([0.0, 1.0, 2.0, 3.0], dtype=float) * (
        code_units.velocity_unit**2
    )
    gravity = Gravity(
        externalgravity=True,
        potential=potential,
        coordinate=coordinate,
        code_units=code_units,
    )

    acceleration = gravity.acceleration_on(coordinate)

    assert np.allclose(acceleration, -3.0)


def test_point_mass_potential():
    radius = np.array([1.0, 2.0], dtype=float) * unyt.cm
    mass = 2.0 * unyt.g
    potential = point_mass_potential(radius, mass)
    expected = -GRAVITATIONAL_CONSTANT_CGS * 2.0 / np.array([1.0, 2.0], dtype=float)

    assert np.allclose(potential.to_value(unyt.cm**2 / unyt.s**2), expected)


def test_point_mass_potential_with_code_units():
    code_units = _code_units()
    radius = np.array([1.0, 2.0], dtype=float) * code_units.length_unit
    mass = 2.0 * code_units.mass_unit
    potential = point_mass_potential(radius, mass, code_units=code_units)
    g_code = (
        GRAVITATIONAL_CONSTANT_CGS
        * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )
    expected = -g_code * np.array([2.0, 1.0], dtype=float)

    assert np.allclose(potential.to_value(code_units.velocity_unit**2), expected)


def test_singular_isothermal_potential():
    radius = np.array([1.0, np.e], dtype=float) * unyt.cm
    sigma = 10.0 * unyt.cm / unyt.s
    reference_radius = 1.0 * unyt.cm
    potential = singular_isothermal_potential(
        radius,
        sigma,
        reference_radius=reference_radius,
    )

    expected = 2.0 * 10.0**2 * np.log(np.array([1.0, np.e], dtype=float))

    assert np.allclose(potential.to_value(unyt.cm**2 / unyt.s**2), expected)


def test_nfw_potential():
    radius = np.array([0.0, 1.0, 2.0], dtype=float) * unyt.cm
    rho_s = 5.0 * unyt.g / unyt.cm**3
    r_s = 2.0 * unyt.cm
    potential = nfw_potential(radius, rho_s, r_s)

    x_value = np.array([0.0, 0.5, 1.0], dtype=float)
    log_over_x = np.ones_like(x_value)
    nonzero = x_value != 0.0
    log_over_x[nonzero] = np.log1p(x_value[nonzero]) / x_value[nonzero]
    expected = -4.0 * np.pi * GRAVITATIONAL_CONSTANT_CGS * 5.0 * 2.0**2 * log_over_x

    assert np.allclose(potential.to_value(unyt.cm**2 / unyt.s**2), expected)


def test_spherical_self_gravity_and_external_gravity_are_combined():
    code_units = _code_units()
    mesh = DummySphericalMesh()
    par = DummyPar()
    par.selfgravity = True
    par.externalgravity = True
    par.gravity = Gravity(
        selfgravity=True,
        externalgravity=True,
        acceleration=lambda coordinate: -np.ones_like(coordinate),
        code_units=code_units,
    )

    rho = np.ones(3, dtype=float)
    total = par.gravity.acceleration_on_mesh(mesh, rho=rho, par=par)
    g_code = (
        GRAVITATIONAL_CONSTANT_CGS
        * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )
    enclosed_mass = rho * (4.0 * np.pi / 3.0) * mesh.coordinate**3
    expected_self = -g_code * enclosed_mass / mesh.coordinate**2
    expected_self[0] = 0.0

    assert np.allclose(total + 1.0, expected_self, rtol=1.0e-12, atol=1.0e-12)


def test_self_gravity_requires_parameters():
    code_units = _code_units()
    gravity = Gravity(selfgravity=True, code_units=code_units)
    with np.testing.assert_raises(ValueError):
        gravity.acceleration_on_mesh(DummyMesh())


@pytest.mark.parametrize("cosmology_type", ["einstein_de_sitter", "lambda_cdm"])
def test_cosmological_gravity_cancels_homogeneous_background(cosmology_type):
    units = _code_units()
    mesh = DummySphericalMesh()
    if cosmology_type == "lambda_cdm":
        cosmology = LambdaCDM.from_code_units(
            units, t_ref=2.0, omega_m=0.3, omega_lambda=0.7, hubble_ref=0.4
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(units)
    cosmic_time = 2.0
    tau = cosmology.supercomoving_time(cosmic_time)

    class Par:
        noghost = 0
        nogrid = 3
        time = tau
        supercomoving_coordinates = True
        mesh = SimpleNamespace(ghost_cells=0, grid_cells=3)
        simulation = SimpleNamespace(current_time=tau)
    par = Par()
    par.cosmology = cosmology

    scale_factor = cosmology.scale_factor(cosmic_time)
    background = cosmology.background_density(cosmic_time) * scale_factor**3
    gravity = Gravity(
        selfgravity=True,
        cosmological=True,
        cosmology=cosmology,
        code_units=units,
    )
    acceleration = gravity.acceleration_on_mesh(
        mesh, rho=np.full(3, background), par=par
    )
    assert np.allclose(acceleration, 0.0)


@pytest.mark.parametrize("cosmology_type", ["einstein_de_sitter", "lambda_cdm"])
def test_cosmological_gravity_scales_excess_mass_with_scale_factor(cosmology_type):
    units = _code_units()
    mesh = DummySphericalMesh()
    if cosmology_type == "lambda_cdm":
        cosmology = LambdaCDM.from_code_units(
            units, t_ref=2.0, omega_m=0.3, omega_lambda=0.7, hubble_ref=0.4
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(units)
    cosmic_time = 2.0
    tau = cosmology.supercomoving_time(cosmic_time)

    class Par:
        noghost = 0
        nogrid = 3
        time = tau
        supercomoving_coordinates = True
        mesh = SimpleNamespace(ghost_cells=0, grid_cells=3)
        simulation = SimpleNamespace(current_time=tau)
    par = Par()
    par.cosmology = cosmology

    scale_factor = cosmology.scale_factor(cosmic_time)
    background = cosmology.background_density(cosmic_time) * scale_factor**3
    density = np.full(3, background)
    density[0] += 1.0
    gravity = Gravity(
        selfgravity=True,
        cosmological=True,
        cosmology=cosmology,
        code_units=units,
    )
    acceleration = gravity.acceleration_on_mesh(mesh, rho=density, par=par)
    g_code = (
        GRAVITATIONAL_CONSTANT_CGS
        * units.mass_in_cgs
        / (units.length_in_cgs * units.velocity_in_cgs**2)
    )
    expected = np.zeros(3)
    expected[1:] = -g_code * scale_factor * (4.0 * np.pi / 3.0) / mesh.coordinate[1:]**2
    assert np.allclose(acceleration, expected)
