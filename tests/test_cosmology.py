import numpy as np
import pytest
import h5py
import tempfile
from pathlib import Path
from types import SimpleNamespace

from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from radhydropy.cosmological_variables import (
    physical_density,
    physical_temperature,
    physical_velocity,
    supercomoving_scale,
    to_supercomoving_density,
    to_supercomoving_temperature,
    to_supercomoving_velocity,
)
from radhydropy.units import CodeUnits
from radhydropy.params import Par
import radhydropy.io as rio


def code_units():
    return CodeUnits.from_mapping({
        "UnitMass_in_cgs": 1.0e33,
        "UnitLength_in_cgs": 1.0e18,
        "UnitVelocity_in_cgs": 1.0e5,
        "UnitCurrent_in_cgs": 1.0,
        "UnitTemp_in_cgs": 1.0,
    })


def test_einstein_de_sitter_background_relations():
    cosmology = EinsteinDeSitter.from_code_units(code_units())
    assert np.isclose(cosmology.scale_factor(8.0), 4.0)
    assert np.isclose(cosmology.hubble(2.0), 1.0 / 3.0)
    assert np.isclose(
        cosmology.background_density(2.0),
        1.0 / (6.0 * np.pi * cosmology.gravitational_constant * 4.0),
    )


def test_einstein_de_sitter_rejects_zero_time():
    with pytest.raises(ValueError):
        EinsteinDeSitter().scale_factor(0.0)


def test_lambda_cdm_reference_normalization_and_round_trip():
    cosmology = LambdaCDM.from_code_units(
        code_units(), t_ref=2.0, a_ref=1.5, omega_m=0.3, omega_lambda=0.7,
    )
    assert np.isclose(cosmology.scale_factor(2.0), 1.5)
    assert np.isclose(
        cosmology.background_density(2.0),
        3.0 * cosmology._hubble_ref**2 * 0.3
        / (8.0 * np.pi * cosmology.gravitational_constant),
    )
    for time in (0.5, 2.0, 8.0):
        tau = cosmology.supercomoving_time(time)
        assert np.isclose(cosmology.cosmic_time_from_supercomoving(tau), time)
    assert np.isclose(cosmology.cosmic_time_from_scale_factor(1.5), 2.0)


def test_supercomoving_time_round_trip():
    cosmology = EinsteinDeSitter()
    for time in (1.0, 2.0, 8.0):
        tau = cosmology.supercomoving_time(time)
        assert np.isclose(cosmology.cosmic_time_from_supercomoving(tau), time)


def test_supercomoving_physical_conversions():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)
    assert np.isclose(cosmology.physical_density(4.0, tau), 1.0)
    assert np.isclose(
        cosmology.physical_pressure(2.0 ** (10.0 / 3.0), tau, 5.0 / 3.0),
        1.0,
    )


def test_supercomoving_variable_round_trip():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)
    class Par:
        time = tau
    par = Par()
    par.cosmology = cosmology
    a, hubble = supercomoving_scale(par)
    gamma = 5.0 / 3.0
    radius = np.array([2.0])
    physical_rho = np.array([4.0])
    physical_temp = np.array([10.0])
    physical_vel = np.array([hubble * a * radius[0] + 3.0 / a])
    rho = to_supercomoving_density(physical_rho, a)
    temp = to_supercomoving_temperature(physical_temp, a, gamma)
    velocity = to_supercomoving_velocity(physical_vel, radius, a, hubble)
    assert np.allclose(physical_density(rho, a), physical_rho)
    assert np.allclose(physical_temperature(temp, a, gamma), physical_temp)
    assert np.allclose(physical_velocity(velocity, radius, a, hubble), physical_vel)


def test_supercomoving_specific_angular_momentum_is_scale_factor_invariant():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)
    class Par:
        time = tau
    par = Par()
    par.cosmology = cosmology
    a, _ = supercomoving_scale(par)

    comoving_radius = np.array([2.0, 4.0])
    physical_tangential_velocity = np.array([3.0, -1.5])
    physical_radius = a * comoving_radius
    supercomoving_tangential_velocity = a * physical_tangential_velocity

    physical_j = physical_radius * physical_tangential_velocity
    supercomoving_j = comoving_radius * supercomoving_tangential_velocity
    assert np.allclose(supercomoving_j, physical_j)


def test_cosmology_header_round_trip_and_supercomoving_input_output():
    units = code_units()
    cosmology = EinsteinDeSitter.from_code_units(units)
    tau = cosmology.supercomoving_time(2.0)
    par = SimpleNamespace(
        coordsys='cartesian', nogrid=2, noghost=0,
        CodeUnits=units, time=tau, boxsize=2.0,
        cosmological_expansion=True, supercomoving_coordinates=True,
        cosmology=cosmology, cosmology_type='einstein_de_sitter',
        cosmology_t_ref=1.0, cosmology_a_ref=1.0,
        coordinate_frame='comoving', time_coordinate='supercomoving',
        velocity_representation='supercomoving_peculiar',
        density_representation='comoving',
        pressure_representation='supercomoving',
        temperature_representation='supercomoving', gamma=5.0 / 3.0,
    )
    mesh = SimpleNamespace(boundary=np.array([0.0, 1.0, 2.0]))
    fluid = SimpleNamespace(
        rho=np.ones(2) * 4.0, vel=np.ones(2) * 2.0,
        temp=np.ones(2) * 3.0, mu=np.ones(2), time=tau,
    )
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / 'supercomoving.hdf5'
        rio.writehdf5(sim, filename)
        with h5py.File(filename, 'r') as handle:
            header = handle['Header']
            assert header.attrs['CosmologyType'] == 'einstein_de_sitter'
            assert header.attrs['TimeCoordinate'] == 'supercomoving'
            assert header.attrs['ScaleFactor'] == pytest.approx(2.0 ** (2.0 / 3.0))
        loaded = SimpleNamespace()
        rio.readhdf5(loaded, SimpleNamespace(), SimpleNamespace(), filename)
        assert loaded.cosmological_expansion
        assert loaded.supercomoving_coordinates
        assert loaded.cosmology.type_name == 'einstein_de_sitter'


def test_lambda_cdm_header_round_trip():
    units = code_units()
    cosmology = LambdaCDM.from_code_units(
        units, t_ref=2.0, a_ref=1.0, omega_m=0.3, omega_lambda=0.7,
        hubble_ref=0.4,
    )
    tau = cosmology.supercomoving_time(2.0)
    par = SimpleNamespace(
        coordsys='cartesian', nogrid=1, noghost=0,
        CodeUnits=units, time=tau, boxsize=1.0,
        cosmological_expansion=True, supercomoving_coordinates=True,
        cosmology=cosmology, cosmology_type='lambda_cdm',
        cosmology_t_ref=2.0, cosmology_a_ref=1.0,
        coordinate_frame='comoving', time_coordinate='supercomoving',
        velocity_representation='supercomoving_peculiar',
        density_representation='comoving', pressure_representation='supercomoving',
        temperature_representation='supercomoving', gamma=5.0 / 3.0,
    )
    mesh = SimpleNamespace(boundary=np.array([0.0, 1.0]))
    fluid = SimpleNamespace(rho=np.ones(1), vel=np.zeros(1), temp=np.ones(1), mu=np.ones(1), time=tau)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / 'lambda_cdm.hdf5'
        rio.writehdf5(sim, filename)
        loaded = SimpleNamespace()
        rio.readhdf5(loaded, SimpleNamespace(), SimpleNamespace(), filename)
        assert loaded.cosmology.type_name == 'lambda_cdm'
        assert loaded.cosmology.omega_m == pytest.approx(0.3)
        assert loaded.cosmology.omega_lambda == pytest.approx(0.7)
        assert loaded.cosmology._hubble_ref == pytest.approx(0.4)


def test_par_constructs_lambda_cdm_from_parameters():
    units = {
        "name": "test",
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0e33,
            "UnitLength_in_cgs": 1.0e18,
            "UnitVelocity_in_cgs": 1.0e5,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    }
    par = Par({
        "CodeUnits": units,
        "cosmological_expansion": True,
        "cosmology_type": "lambda_cdm",
        "cosmology_omega_m": 0.3,
        "cosmology_omega_lambda": 0.7,
        "cosmology_hubble_ref": 0.4,
    })
    assert par.cosmology.type_name == "lambda_cdm"
    assert par.cosmology._hubble_ref == pytest.approx(0.4)
