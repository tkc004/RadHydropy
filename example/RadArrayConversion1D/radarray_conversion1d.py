"""Executable conversion checks for RadHydropy's representation-aware arrays."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example"))

from radhydropy.cosmology_context import CosmologyContext
from radhydropy.field_metadata import field_spec
from radhydropy.radarray import RadArray, RepresentationMismatchError
from radhydropy.units import CodeUnits
import example_utils as eu


CONFIG_FILE = Path(__file__).with_name("radarray_conversion1d.yaml")


def _rad_array(values, field_name, code_units, cosmology):
    hubble_parameter_km_s_Mpc = (
        cosmology.hubble_parameter_km_s_Mpc
        if field_name == "vel_supercomoving_code"
        else None
    )
    return RadArray(
        values,
        code_units=code_units,
        field_spec=field_spec(
            field_name,
            code_units,
            scale_factor=cosmology.scale_factor,
            cosmology=cosmology.cosmology,
            hubble_parameter_km_s_Mpc=hubble_parameter_km_s_Mpc,
        ),
        cosmology=cosmology,
    )


def run(config_file=CONFIG_FILE):
    """Run the standalone RadArray conversion and arithmetic checks."""
    config = eu.load_nested_example_config(config_file)
    code_units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    gamma = float(config["par"]["hydrodynamics"]["gamma"])
    scale_factor = float(config["example"]["scale_factor"])
    hubble_parameter_km_s_Mpc = float(
        config["example"]["hubble_parameter"].to_value("km/(s*Mpc)")
    )
    cosmology = CosmologyContext(
        gamma=gamma,
        scale_factor=scale_factor,
        hubble_parameter_km_s_Mpc=hubble_parameter_km_s_Mpc,
    )

    radius_comoving_code_radarray = _rad_array(
        np.array([1.0, 2.0]), "boundary_comoving_code", code_units, cosmology
    )
    radius_proper_code_radarray = radius_comoving_code_radarray.to_proper()
    np.testing.assert_allclose(radius_proper_code_radarray.value, [0.5, 1.0])
    np.testing.assert_allclose(
        radius_proper_code_radarray.to_comoving().value,
        radius_comoving_code_radarray.value,
    )

    rho_comoving_code_radarray = _rad_array(
        np.array([8.0, 16.0]), "rho_comoving_code", code_units, cosmology
    )
    rho_proper_code_radarray = rho_comoving_code_radarray.to_proper()
    np.testing.assert_allclose(rho_proper_code_radarray.value, [64.0, 128.0])
    np.testing.assert_allclose(
        rho_proper_code_radarray.to_comoving().value,
        rho_comoving_code_radarray.value,
    )

    temp_supercomoving_code_radarray = _rad_array(
        np.array([8.0]), "temp_supercomoving_code", code_units, cosmology
    )
    temp_proper_code_radarray = temp_supercomoving_code_radarray.to_proper()
    np.testing.assert_allclose(temp_proper_code_radarray.value, [32.0])
    np.testing.assert_allclose(
        temp_proper_code_radarray.to_comoving().value,
        temp_supercomoving_code_radarray.value,
    )

    pre_supercomoving_code_radarray = _rad_array(
        np.array([64.0]), "pre_supercomoving_code", code_units, cosmology
    )
    pre_proper_code_radarray = pre_supercomoving_code_radarray.to_proper()
    np.testing.assert_allclose(pre_proper_code_radarray.value, [2048.0])
    np.testing.assert_allclose(
        pre_proper_code_radarray.to_comoving().value,
        pre_supercomoving_code_radarray.value,
    )

    x_comoving_code = np.array([4.0])
    vel_supercomoving_code_radarray = _rad_array(
        np.array([2.0]), "vel_supercomoving_code", code_units, cosmology
    )
    vel_proper_code_radarray = vel_supercomoving_code_radarray.to_proper(
        x_comoving_code=x_comoving_code
    )
    np.testing.assert_allclose(vel_proper_code_radarray.value, [4.14])
    np.testing.assert_allclose(
        vel_proper_code_radarray.to_comoving(
            x_comoving_code=x_comoving_code
        ).value,
        vel_supercomoving_code_radarray.value,
    )

    try:
        rho_proper_code_radarray + rho_comoving_code_radarray
    except RepresentationMismatchError:
        pass
    else:
        raise AssertionError(
            "proper and comoving density arrays must not be addable"
        )

    return {
        "radius_proper_code": radius_proper_code_radarray.value,
        "rho_proper_code": rho_proper_code_radarray.value,
        "temp_proper_code": temp_proper_code_radarray.value,
        "pre_proper_code": pre_proper_code_radarray.value,
        "vel_proper_code": vel_proper_code_radarray.value,
    }


if __name__ == "__main__":
    run()
    print("RadArrayConversion1D: all conversion and representation checks passed")
