from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import pytest

from example.example_utils import load_nested_example_config
from radhydropy.cosmology_context import CosmologyContext
from radhydropy.field_metadata import field_spec
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.radarray import RadArray, RadQuantity
from radhydropy.units import CodeUnits
import radhydropy.io as rio


CONFIG_FILE = (
    Path(__file__).parents[1]
    / "example"
    / "HDF5UnitCosmologyRoundtrip1D"
    / "hdf5_unit_cosmology_roundtrip1d.yaml"
)


def _code_units(config):
    return CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])


def _radarray(values, field_name, code_units, context):
    return RadArray(
        values,
        code_units=code_units,
        field_spec=field_spec(
            field_name,
            code_units,
            cosmology=context.cosmology,
            scale_factor=context.scale_factor,
            hubble_parameter_km_s_Mpc=(
                context.hubble_parameter_km_s_Mpc
                if field_name == "vel_supercomoving_code"
                else None
            ),
        ),
        cosmology=context,
    )


def _box_size(value, code_units, context):
    return RadQuantity(
        value,
        code_units=code_units,
        field_spec=field_spec(
            "boundary_proper_code",
            code_units,
            cosmology=context.cosmology,
            scale_factor=context.scale_factor,
        ),
        cosmology=context,
    )


def test_writer_compact_assignments_roundtrip_proper_ic():
    config = load_nested_example_config(CONFIG_FILE)
    code_units = _code_units(config)
    context = CosmologyContext(gamma=5.0 / 3.0, cosmology="proper")
    writer = InitialConditionWriter(
        par_config=deepcopy(config["par"]),
        code_units=code_units,
    )
    writer.box_size = _box_size(2.0, code_units, context)
    writer.mesh.boundary_radarray = _radarray(
        np.array([0.0, 1.0, 2.0]),
        "boundary_proper_code",
        code_units,
        context,
    )
    writer.fluid.rho_radarray = _radarray(
        np.array([2.0, 3.0]), "rho_proper_code", code_units, context
    )
    writer.fluid.vel_radarray = _radarray(
        np.array([4.0, 5.0]), "vel_proper_code", code_units, context
    )
    writer.fluid.temp_radarray = _radarray(
        np.array([6.0, 7.0]), "temp_proper_code", code_units, context
    )
    writer.fluid.pre_radarray = _radarray(
        np.array([8.0, 9.0]), "pre_proper_code", code_units, context
    )

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as output:
        writer.write(output.name)
        np.testing.assert_allclose(writer.simulation.fluid.Mass_code, [2.0, 3.0])
        np.testing.assert_allclose(writer.simulation.fluid.Mom_code, [8.0, 15.0])
        assert np.all(np.asarray(writer.simulation.fluid.Energy_code) > 0.0)
        restored = rio.loadhdf5(config, output.name)

    np.testing.assert_allclose(restored.mesh.boundary_proper_code, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(restored.fluid.rho_proper_code, [2.0, 3.0])
    np.testing.assert_allclose(restored.fluid.vel_proper_code, [4.0, 5.0])
    np.testing.assert_allclose(restored.fluid.temp_proper_code, [6.0, 7.0])


def test_writer_radarray_factory_uses_default_proper_context():
    config = load_nested_example_config(CONFIG_FILE)
    code_units = _code_units(config)
    writer = InitialConditionWriter(
        par_config=deepcopy(config["par"]),
        code_units=code_units,
    )

    density_radarray = writer.radarray(
        np.array([1.0, 2.0]) * code_units.density_unit,
    )

    assert isinstance(density_radarray, RadArray)
    assert density_radarray.field_spec.representation == "proper"
    assert density_radarray.cosmology.cosmology == "proper"
    assert density_radarray.cosmology.gamma == pytest.approx(5.0 / 3.0)
    np.testing.assert_allclose(density_radarray.value, [1.0, 2.0])

    box_size_radquantity = writer.radquantity(2.0 * code_units.length_unit)
    assert isinstance(box_size_radquantity, RadQuantity)
    assert box_size_radquantity.field_spec.quantity == "radius"
    assert box_size_radquantity.cosmology.cosmology == "proper"
    assert box_size_radquantity.value == pytest.approx(2.0)

    with pytest.raises(ValueError, match="cannot infer"):
        writer.radarray(np.ones(2) * code_units.time_unit)

    with pytest.raises(TypeError, match="unit-bearing"):
        writer.radarray(np.ones(2))


def test_writer_converts_radarrays_to_supercomoving_velocity_with_position():
    config = load_nested_example_config(CONFIG_FILE)
    config = deepcopy(config)
    config["par"]["cosmology"]["cosmological_expansion"] = True
    config["par"]["cosmology"]["supercomoving_coordinates"] = True
    code_units = _code_units(config)
    writer = InitialConditionWriter(
        par_config=config["par"],
        code_units=code_units,
    )
    simulation = writer.simulation
    scale_factor = 0.5
    time_cosmic_code = float(
        simulation.par.cosmology.model.cosmic_time_from_scale_factor(scale_factor)
    )
    hubble_code = float(simulation.par.cosmology.model.hubble(time_cosmic_code))
    hubble_unit_km_s_Mpc = (
        code_units.velocity_unit.to_value("km/s")
        / code_units.length_unit.to_value("Mpc")
    )
    context = CosmologyContext(
        gamma=5.0 / 3.0,
        cosmology=simulation.par.cosmology.model.type_name,
        scale_factor=scale_factor,
        hubble_parameter_km_s_Mpc=hubble_code * hubble_unit_km_s_Mpc,
    )
    simulation.par.cosmology_context = context
    tau_supercomoving_code = float(
        simulation.par.cosmology.model.supercomoving_time(time_cosmic_code)
    )
    simulation.par.tau_supercomoving_code = np.array([tau_supercomoving_code])
    simulation.par.simulation.tau_supercomoving_code = (
        simulation.par.tau_supercomoving_code.copy()
    )
    simulation.par.coordinate_frame = "comoving"
    simulation.par.time_coordinate = "supercomoving"
    simulation.par.velocity_representation = "supercomoving_peculiar"
    simulation.par.density_representation = "comoving"
    simulation.par.pressure_representation = "supercomoving"
    simulation.par.temperature_representation = "supercomoving"

    velocity_radquantity = writer.radquantity(5.0 * code_units.velocity_unit)
    assert velocity_radquantity.field_spec.quantity == "velocity"
    assert velocity_radquantity.field_spec.representation == "supercomoving"
    proper_velocity = velocity_radquantity.to_proper(x_comoving_code=1.0)
    assert proper_velocity.value == pytest.approx(
        _radarray(np.array([5.0]), "vel_supercomoving_code", code_units, context)
        .to_proper(x_comoving_code=np.array([1.0]))[0]
    )

    writer.box_size = _box_size(2.0, code_units, context)
    writer.mesh.boundary_radarray = _radarray(
        np.array([0.0, 1.0, 2.0]),
        "boundary_proper_code",
        code_units,
        context,
    )
    writer.fluid.rho_radarray = _radarray(
        np.ones(2), "rho_proper_code", code_units, context
    )
    writer.fluid.vel_radarray = _radarray(
        np.array([5.0, 5.0]), "vel_proper_code", code_units, context
    )
    writer.fluid.temp_radarray = _radarray(
        np.ones(2), "temp_proper_code", code_units, context
    )
    writer.fluid.pre_radarray = _radarray(
        np.ones(2), "pre_proper_code", code_units, context
    )

    writer.prepare()
    expected = _radarray(
        np.array([5.0, 5.0]), "vel_proper_code", code_units, context
    ).to_comoving(x_comoving_code=np.array([1.0, 3.0]))
    np.testing.assert_allclose(
        writer.simulation.fluid.vel_supercomoving_code,
        expected.value,
    )


def test_writer_cosmological_prepare_and_hdf5_roundtrip():
    config = deepcopy(load_nested_example_config(CONFIG_FILE))
    config["par"]["cosmology"].update({
        "cosmological_expansion": True,
        "supercomoving_coordinates": True,
    })
    code_units = _code_units(config)
    writer = InitialConditionWriter(
        par_config=config["par"],
        code_units=code_units,
    )
    simulation = writer.simulation
    scale_factor = 0.5
    time_cosmic_code = float(
        simulation.par.cosmology.model.cosmic_time_from_scale_factor(scale_factor)
    )
    hubble_code = float(simulation.par.cosmology.model.hubble(time_cosmic_code))
    hubble_unit_km_s_Mpc = (
        code_units.velocity_unit.to_value("km/s")
        / code_units.length_unit.to_value("Mpc")
    )
    context = CosmologyContext(
        gamma=5.0 / 3.0,
        cosmology=simulation.par.cosmology.model.type_name,
        scale_factor=scale_factor,
        hubble_parameter_km_s_Mpc=hubble_code * hubble_unit_km_s_Mpc,
    )
    simulation.par.cosmology_context = context
    tau_supercomoving_code = float(
        simulation.par.cosmology.model.supercomoving_time(time_cosmic_code)
    )
    simulation.par.tau_supercomoving_code = np.array([tau_supercomoving_code])
    simulation.par.simulation.tau_supercomoving_code = (
        simulation.par.tau_supercomoving_code.copy()
    )
    simulation.par.coordinate_frame = "comoving"
    simulation.par.time_coordinate = "supercomoving"
    simulation.par.velocity_representation = "supercomoving_peculiar"
    simulation.par.density_representation = "comoving"
    simulation.par.pressure_representation = "supercomoving"
    simulation.par.temperature_representation = "supercomoving"
    simulation.par.gas_angular_momentum = True
    writer.box_size = _box_size(2.0, code_units, context)
    writer.mesh.boundary_radarray = _radarray(
        np.array([0.0, 1.0, 2.0]),
        "boundary_proper_code",
        code_units,
        context,
    )
    writer.fluid.rho_radarray = _radarray(
        np.array([2.0, 3.0]), "rho_proper_code", code_units, context
    )
    writer.fluid.vel_radarray = _radarray(
        np.array([5.0, 5.0]), "vel_proper_code", code_units, context
    )
    writer.fluid.temp_radarray = _radarray(
        np.array([8.0, 8.0]), "temp_proper_code", code_units, context
    )
    writer.fluid.pre_radarray = _radarray(
        np.array([1.0, 1.0]), "pre_proper_code", code_units, context
    )
    simulation.fluid.mu = np.ones(2)
    simulation.fluid.specific_angular_momentum_code = np.full(2, 0.25)

    expected_boundary = _radarray(
        np.array([0.0, 1.0, 2.0]), "boundary_proper_code", code_units, context
    ).to_comoving().value
    expected_density = _radarray(
        np.array([2.0, 3.0]), "rho_proper_code", code_units, context
    ).to_comoving().value
    expected_temperature = _radarray(
        np.array([8.0, 8.0]), "temp_proper_code", code_units, context
    ).to_comoving().value
    expected_velocity = _radarray(
        np.array([5.0, 5.0]), "vel_proper_code", code_units, context
    ).to_comoving(x_comoving_code=np.array([1.0, 3.0])).value

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as output:
        writer.write(output.name)
        prepared = writer.simulation
        np.testing.assert_allclose(
            prepared.mesh.boundary_comoving_code, expected_boundary
        )
        np.testing.assert_allclose(
            prepared.fluid.rho_comoving_code, expected_density
        )
        np.testing.assert_allclose(
            prepared.fluid.temp_supercomoving_code, expected_temperature
        )
        np.testing.assert_allclose(
            prepared.fluid.vel_supercomoving_code, expected_velocity
        )
        np.testing.assert_allclose(
            prepared.fluid.specific_angular_momentum_code, 0.25
        )
        assert np.all(np.isfinite(prepared.fluid.Energy_code))
        restored = rio.loadhdf5(config, output.name)

    np.testing.assert_allclose(
        restored.mesh.boundary_comoving_code, expected_boundary
    )
    np.testing.assert_allclose(restored.fluid.rho_comoving_code, expected_density)
    np.testing.assert_allclose(
        restored.fluid.temp_supercomoving_code, expected_temperature
    )
    np.testing.assert_allclose(restored.fluid.vel_supercomoving_code, expected_velocity)
    np.testing.assert_allclose(
        restored.par.tau_supercomoving_code, [tau_supercomoving_code]
    )
    np.testing.assert_allclose(
        restored.fluid.specific_angular_momentum_code, 0.25
    )


def test_writer_rejects_code_unit_mismatch():
    config = load_nested_example_config(CONFIG_FILE)
    wrong_units = CodeUnits.from_mapping(
        {
            "name": "wrong",
            "InternalUnitSystem": {
                "UnitMass_in_cgs": 2.0,
                "UnitLength_in_cgs": 1.0,
                "UnitVelocity_in_cgs": 1.0,
                "UnitCurrent_in_cgs": 1.0,
                "UnitTemp_in_cgs": 1.0,
            },
        }
    )
    with pytest.raises(ValueError, match="code-unit mass_in_cgs"):
        InitialConditionWriter(
            par_config=deepcopy(config["par"]),
            code_units=wrong_units,
        )


def test_writer_rejects_box_size_mismatch():
    config = load_nested_example_config(CONFIG_FILE)
    code_units = _code_units(config)
    context = CosmologyContext(gamma=5.0 / 3.0, cosmology="proper")
    writer = InitialConditionWriter(
        par_config=deepcopy(config["par"]),
        code_units=code_units,
        box_size=_box_size(3.0, code_units, context),
    )
    writer.mesh.boundary_radarray = _radarray(
        np.array([0.0, 1.0, 2.0]),
        "boundary_proper_code",
        code_units,
        context,
    )
    writer.fluid.rho_radarray = _radarray(
        np.ones(2), "rho_proper_code", code_units, context
    )
    writer.fluid.vel_radarray = _radarray(
        np.zeros(2), "vel_proper_code", code_units, context
    )
    writer.fluid.temp_radarray = _radarray(
        np.ones(2), "temp_proper_code", code_units, context
    )
    writer.fluid.pre_radarray = _radarray(
        np.ones(2), "pre_proper_code", code_units, context
    )

    with pytest.raises(ValueError, match="box_size"):
        writer.prepare()


def test_writer_validates_active_state_after_solver_setup():
    config = load_nested_example_config(CONFIG_FILE)
    code_units = _code_units(config)
    context = CosmologyContext(gamma=5.0 / 3.0, cosmology="proper")
    writer = InitialConditionWriter(
        par_config=deepcopy(config["par"]),
        code_units=code_units,
    )
    writer.mesh.boundary_radarray = _radarray(
        np.array([0.0, 1.0, 2.0]),
        "boundary_proper_code",
        code_units,
        context,
    )
    writer.fluid.rho_radarray = _radarray(
        np.array([1.0, -1.0]), "rho_proper_code", code_units, context
    )
    writer.fluid.vel_radarray = _radarray(
        np.zeros(2), "vel_proper_code", code_units, context
    )
    writer.fluid.temp_radarray = _radarray(
        np.ones(2), "temp_proper_code", code_units, context
    )
    writer.fluid.pre_radarray = _radarray(
        np.ones(2), "pre_proper_code", code_units, context
    )

    writer.prepare()
    with pytest.raises(ValueError, match="active rho_proper_code"):
        writer.prepare(validate=True)
