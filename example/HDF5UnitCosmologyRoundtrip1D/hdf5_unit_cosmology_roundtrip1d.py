"""Write and read only initial-condition HDF5 files under several contracts."""

from copy import deepcopy
from pathlib import Path
import tempfile
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.cosmology_context import CosmologyContext
from radhydropy.field_metadata import field_spec
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.radarray import RadArray, RadQuantity
from radhydropy.units import CodeUnits
import example_utils as eu


CONFIG_FILE = Path(__file__).with_name("hdf5_unit_cosmology_roundtrip1d.yaml")


def _radarray(code_values, field_name, code_units, cosmology_context):
    hubble_parameter_km_s_Mpc = (
        cosmology_context.hubble_parameter_km_s_Mpc
        if field_name == "vel_supercomoving_code"
        else None
    )
    return RadArray(
        code_values,
        code_units=code_units,
        field_spec=field_spec(
            field_name,
            code_units,
            cosmology=cosmology_context.cosmology,
            scale_factor=cosmology_context.scale_factor,
            hubble_parameter_km_s_Mpc=hubble_parameter_km_s_Mpc,
        ),
        cosmology=cosmology_context,
    )


def _case_config(config, case_name, case):
    case_config = deepcopy(config)
    case_par = case_config["par"]
    case_par["simulation"]["name"] = case_name
    case_par["units"]["CodeUnits"] = deepcopy(case["CodeUnits"])
    case_par["cosmology"]["cosmological_expansion"] = bool(
        case["cosmological_expansion"]
    )
    case_par["cosmology"]["supercomoving_coordinates"] = bool(
        case["supercomoving_coordinates"]
    )
    case_par["cosmology"]["cosmology_type"] = case["cosmology_type"]
    return case_config


def _build_initial_condition(case_config):
    code_units = CodeUnits.from_mapping(
        case_config["par"]["units"]["CodeUnits"]
    )
    writer = InitialConditionWriter(
        ic_config=case_config["initial_condition"],
        par_config=case_config["par"],
        code_units=code_units,
    )
    sim = writer.simulation
    initial_condition = case_config["initial_condition"]
    code_units = sim.par.units.CodeUnits
    gamma = float(sim.par.hydrodynamics.gamma)
    grid_cells = int(sim.par.mesh.grid_cells)
    box_size_proper_code = float(
        initial_condition["box_size_proper"].to_value(code_units.length_unit)
    )
    density_proper_code = float(
        initial_condition["density_proper"].to_value(code_units.density_unit)
    )
    temperature_proper_code = float(
        initial_condition["temperature_proper"].to_value(code_units.temperature_unit)
    )
    velocity_proper_code = float(
        initial_condition["velocity_proper"].to_value(code_units.velocity_unit)
    )
    mu_dimensionless = float(initial_condition["mean_molecular_weight"])
    boundary_proper_code = np.linspace(
        0.0, box_size_proper_code, grid_cells + 1
    )
    rho_proper_code = np.full(grid_cells, density_proper_code)
    vel_proper_code = np.full(grid_cells, velocity_proper_code)
    temp_proper_code = np.full(grid_cells, temperature_proper_code)
    mu_dimensionless_array = np.full(grid_cells, mu_dimensionless)
    sim.fluid.mu = mu_dimensionless_array

    if case_config["par"]["cosmology"]["cosmological_expansion"]:
        time_cosmic_code = float(
            initial_condition["time_cosmic"].to_value(code_units.time_unit)
        )
        cosmology = sim.par.cosmology
        scale_factor = float(cosmology.scale_factor(time_cosmic_code))
        hubble_parameter_code = float(cosmology.hubble(time_cosmic_code))
        hubble_unit_km_s_Mpc = (
            code_units.velocity_unit.to_value("km/s")
            / code_units.length_unit.to_value("Mpc")
        )
        cosmology_context = CosmologyContext(
            gamma=gamma,
            cosmology=cosmology.type_name,
            scale_factor=scale_factor,
            hubble_parameter_km_s_Mpc=(
                hubble_parameter_code * hubble_unit_km_s_Mpc
            ),
        )
    else:
        cosmology_context = CosmologyContext(
            gamma=gamma,
            cosmology="proper",
            scale_factor=1.0,
            hubble_parameter_km_s_Mpc=0.0,
        )

    boundary_proper_code_radarray = _radarray(
        boundary_proper_code,
        "boundary_proper_code",
        code_units,
        cosmology_context,
    )
    rho_proper_code_radarray = _radarray(
        rho_proper_code,
        "rho_proper_code",
        code_units,
        cosmology_context,
    )
    temp_proper_code_radarray = _radarray(
        temp_proper_code,
        "temp_proper_code",
        code_units,
        cosmology_context,
    )
    vel_proper_code_radarray = _radarray(
        vel_proper_code,
        "vel_proper_code",
        code_units,
        cosmology_context,
    )
    pressure_proper_code = sim.fluid.eos.pressure(
        rho_proper_code, temp_proper_code, mu_dimensionless_array
    )
    pressure_proper_code_radarray = _radarray(
        pressure_proper_code,
        "pre_proper_code",
        code_units,
        cosmology_context,
    )
    sim.par.cosmology_context = cosmology_context

    if case_config["par"]["cosmology"]["cosmological_expansion"]:
        tau_supercomoving_code = float(
            cosmology.supercomoving_time(time_cosmic_code)
        )
        sim.par.tau_supercomoving_code = np.asarray([tau_supercomoving_code])
        sim.par.simulation.tau_supercomoving_code = (
            sim.par.tau_supercomoving_code.copy()
        )
        sim.par.coordinate_frame = "comoving"
        sim.par.time_coordinate = "supercomoving"
        sim.par.velocity_representation = "supercomoving_peculiar"
        sim.par.density_representation = "comoving"
        sim.par.pressure_representation = "supercomoving"
        sim.par.temperature_representation = "supercomoving"
    else:
        sim.par.simulation.time_proper_code = 0.0

    writer.box_size = RadQuantity(
        box_size_proper_code,
        code_units=code_units,
        field_spec=field_spec(
            "boundary_proper_code",
            code_units,
            cosmology=cosmology_context.cosmology,
            scale_factor=cosmology_context.scale_factor,
        ),
        cosmology=cosmology_context,
    )
    writer.mesh.boundary_radarray = boundary_proper_code_radarray
    writer.fluid.rho_radarray = rho_proper_code_radarray
    writer.fluid.vel_radarray = vel_proper_code_radarray
    writer.fluid.temp_radarray = temp_proper_code_radarray
    writer.fluid.pre_radarray = pressure_proper_code_radarray
    return writer


def _assert_roundtrip(case_config, output_filename):
    initial_writer = _build_initial_condition(case_config)
    initial_writer.write(output_filename)
    target_fields = (
        "boundary_comoving_code",
        "rho_comoving_code",
        "vel_supercomoving_code",
        "temp_supercomoving_code",
        ) if case_config["par"]["cosmology"]["cosmological_expansion"] else (
        "boundary_proper_code",
        "rho_proper_code",
        "vel_proper_code",
        "temp_proper_code",
    )
    expected = {
        field_name: np.asarray(
            getattr(
                initial_writer.simulation.mesh
                if field_name.startswith("boundary_")
                else initial_writer.simulation.fluid,
                field_name,
            ),
            dtype=float,
        ).copy()
        for field_name in target_fields
    }
    restored = rio.loadhdf5(case_config, output_filename)
    code_units = restored.par.units.CodeUnits
    assert code_units.name == case_config["example"]["_active_case_name"]
    for field_name, expected_values in expected.items():
        if field_name.startswith("boundary_"):
            actual_values = getattr(restored.mesh, field_name)
        else:
            actual_values = getattr(restored.fluid, field_name)
        np.testing.assert_allclose(actual_values, expected_values)
    assert isinstance(restored.mesh.boundary_radarray, RadArray)
    assert isinstance(restored.fluid.rho_radarray, RadArray)
    assert isinstance(restored.fluid.vel_radarray, RadArray)
    assert isinstance(restored.fluid.temp_radarray, RadArray)
    np.testing.assert_allclose(
        restored.fluid.rho_radarray.value,
        expected["rho_comoving_code" if "rho_comoving_code" in expected else "rho_proper_code"],
    )
    if case_config["par"]["cosmology"]["cosmological_expansion"]:
        assert restored.par.cosmology_context is not None
        assert restored.par.coordinate_frame == "comoving"
        assert restored.par.velocity_representation == "supercomoving_peculiar"
    else:
        assert restored.par.cosmology_context is not None
        assert restored.par.cosmology_context.scale_factor == 1.0
        assert restored.par.coordinate_frame == "physical"


def run(config_file=CONFIG_FILE, output_directory=None):
    """Run proper and cosmological initial-condition HDF5 round trips."""
    config = eu.load_nested_example_config(config_file)
    if output_directory is None:
        output_directory = Path(tempfile.mkdtemp(prefix="radhydropy-hdf5-"))
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for case_name, case in config["example"]["cases"].items():
        case_config = _case_config(config, case_name, case)
        case_config["example"]["_active_case_name"] = case["CodeUnits"]["name"]
        output_filename = output_directory / f"{case_name}.hdf5"
        _assert_roundtrip(case_config, str(output_filename))
        results[case_name] = output_filename
    return results


if __name__ == "__main__":
    run()
    print("HDF5UnitCosmologyRoundtrip1D: all initial-condition round trips passed")
