"""Cosmological background models and runtime state conversions.

The package façade keeps the public ``radhydropy.cosmology`` import focused
on the background models while exposing the representation-aware runtime
types and conversion helpers from their responsibility-specific modules.
Persisted HDF5 schema remains under :mod:`radhydropy.io`.
"""

from radhydropy.cosmology.background import EinsteinDeSitter, LambdaCDM
from radhydropy.cosmology.context import CosmologyContext
from radhydropy.cosmology.state import (
    ProperCgsState,
    SupercomovingHdf5State,
    SupercomovingMeshState,
    SupercomovingState,
    proper_to_supercomoving_time,
    supercomoving_to_cosmic_time,
    to_proper_state,
    to_supercomoving_state,
    validate_supercomoving_contract,
)
from radhydropy.cosmology.variables import (
    physical_density,
    physical_fields,
    physical_pressure,
    physical_radius,
    physical_temperature,
    physical_velocity,
    supercomoving_fields,
    supercomoving_scale,
    to_supercomoving_density,
    to_supercomoving_temperature,
    to_supercomoving_velocity,
)

__all__ = [
    "EinsteinDeSitter",
    "LambdaCDM",
    "CosmologyContext",
    "ProperCgsState",
    "SupercomovingHdf5State",
    "SupercomovingMeshState",
    "SupercomovingState",
    "physical_density",
    "physical_fields",
    "physical_pressure",
    "physical_radius",
    "physical_temperature",
    "physical_velocity",
    "proper_to_supercomoving_time",
    "supercomoving_fields",
    "supercomoving_scale",
    "supercomoving_to_cosmic_time",
    "to_proper_state",
    "to_supercomoving_density",
    "to_supercomoving_state",
    "to_supercomoving_temperature",
    "to_supercomoving_velocity",
    "validate_supercomoving_contract",
]
