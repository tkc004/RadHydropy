import numpy as np
import unyt

from radhydropy.gravity import (
    Gravity,
    point_mass_potential,
    nfw_potential,
    singular_isothermal_potential,
)
from radhydropy.solver import Solver


class DummyMesh:
    def __init__(self):
        self.coordsys = "cartesian"
        self.coordinate = np.array([0.5, 1.5, 2.5]) * unyt.cm
        self.vol = np.ones(3) * unyt.cm**3


class DummyFluid:
    def __init__(self):
        self.rho = np.ones(3) * unyt.g / unyt.cm**3
        self.vel = np.array([0.0, 1.0, -2.0]) * unyt.cm / unyt.s
        self.Mom = self.rho * self.vel * (1.0 * unyt.cm**3)
        self.Energy = np.zeros(3) * unyt.erg


class DummyPar:
    externalgravity = True
    selfgravity = False


def test_gravity_acceleration_from_potential():
    coordinate = np.array([0.0, 1.0, 2.0, 3.0]) * unyt.cm
    potential = 3.0 * coordinate * unyt.cm / unyt.s**2
    gravity = Gravity(externalgravity=True, potential=potential, coordinate=coordinate)

    acceleration = gravity.acceleration_on(coordinate)

    assert np.allclose(
        acceleration.to_value(unyt.cm / unyt.s**2),
        -3.0,
    )


def test_point_mass_potential():
    radius = np.array([1.0, 2.0]) * unyt.cm
    mass = 2.0 * unyt.g
    potential = point_mass_potential(radius, mass)
    expected = (
        -unyt.physical_constants.gravitational_constant * mass / radius
    ).to(unyt.cm**2 / unyt.s**2)

    assert np.allclose(
        potential.to_value(unyt.cm**2 / unyt.s**2),
        expected.to_value(unyt.cm**2 / unyt.s**2),
    )


def test_singular_isothermal_potential():
    radius = np.array([1.0, np.e]) * unyt.cm
    sigma = 10.0 * unyt.cm / unyt.s
    reference_radius = 1.0 * unyt.cm
    potential = singular_isothermal_potential(
        radius,
        sigma,
        reference_radius=reference_radius,
    )

    expected = (
        2.0 * sigma**2 * np.log(radius / reference_radius)
    ).to(unyt.cm**2 / unyt.s**2)

    assert np.allclose(
        potential.to_value(unyt.cm**2 / unyt.s**2),
        expected.to_value(unyt.cm**2 / unyt.s**2),
    )


def test_nfw_potential():
    radius = np.array([0.0, 1.0, 2.0]) * unyt.cm
    rho_s = 5.0 * unyt.g / unyt.cm**3
    r_s = 2.0 * unyt.cm
    potential = nfw_potential(radius, rho_s, r_s)

    x = radius / r_s
    x_value = x.to_value(unyt.dimensionless)
    log_over_x = np.ones_like(x_value)
    nonzero = x_value != 0.0
    log_over_x[nonzero] = np.log1p(x_value[nonzero]) / x_value[nonzero]
    expected = (
        -4.0
        * np.pi
        * unyt.physical_constants.gravitational_constant
        * rho_s
        * r_s**2
        * log_over_x
    ).to(unyt.cm**2 / unyt.s**2)

    assert np.allclose(
        potential.to_value(unyt.cm**2 / unyt.s**2),
        expected.to_value(unyt.cm**2 / unyt.s**2),
    )


def test_apply_external_gravity_updates_momentum_and_energy():
    mesh = DummyMesh()
    fluid = DummyFluid()
    par = DummyPar()
    initial_momentum = fluid.Mom.copy()
    initial_energy = fluid.Energy.copy()
    par.gravity = Gravity(
        externalgravity=True,
        potential=2.0 * mesh.coordinate * unyt.cm / unyt.s**2,
        coordinate=mesh.coordinate,
    )

    solver = Solver()
    solver.ApplyExternalGravity(1.0 * unyt.s, mesh, fluid, par)

    expected_acceleration = -2.0 * unyt.cm / unyt.s**2
    expected_momentum = initial_momentum + fluid.rho * expected_acceleration * mesh.vol * (1.0 * unyt.s)
    expected_energy = initial_energy + fluid.rho * fluid.vel * expected_acceleration * mesh.vol * (1.0 * unyt.s)

    assert np.allclose(fluid.Mom.to_value(expected_momentum.units), expected_momentum.to_value(expected_momentum.units))
    assert np.allclose(fluid.Energy.to_value(expected_energy.units), expected_energy.to_value(expected_energy.units))
