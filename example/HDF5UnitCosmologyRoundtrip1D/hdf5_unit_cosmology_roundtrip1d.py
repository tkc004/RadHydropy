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
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
)
from radhydropy.units import CodeUnits
import example_utils as eu


CONFIG_FILE = Path(__file__).with_name("hdf5_unit_cosmology_roundtrip1d.yaml")


def _case_config(config, case_name, case):
    case_config = deepcopy(config)
    case_par = case_config["par"]
    case_par["simulation"]["name"] = case_name
    case_par["units"]["CodeUnits"] = deepcopy(case["CodeUnits"])
    case_par["gravity"]["cosmological_expansion"] = bool(
        case["cosmological_expansion"]
    )
    case_par["gravity"]["supercomoving_coordinates"] = bool(
        case["supercomoving_coordinates"]
    )
    case_par["gravity"]["cosmology_type"] = case["cosmology_type"]
    return case_config


def _build_initial_condition(case_config):
    sim = Rsim(case_config["par"])
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
    x_proper_code = 0.5 * (
        boundary_proper_code[:-1] + boundary_proper_code[1:]
    )
    width_proper_code = np.diff(boundary_proper_code)
    area_proper_code = np.ones(grid_cells)
    volume_proper_code = width_proper_code.copy()
    rho_proper_code = np.full(grid_cells, density_proper_code)
    vel_proper_code = np.full(grid_cells, velocity_proper_code)
    temp_proper_code = np.full(grid_cells, temperature_proper_code)
    mu_dimensionless_array = np.full(grid_cells, mu_dimensionless)
    sim.fluid.mu = mu_dimensionless_array

    if case_config["par"]["gravity"]["cosmological_expansion"]:
        cosmology = sim.par.cosmology
        time_cosmic_code = float(
            initial_condition["time_cosmic"].to_value(code_units.time_unit)
        )
        tau_supercomoving_code = float(
            cosmology.supercomoving_time(time_cosmic_code)
        )
        scale_factor = float(cosmology.scale_factor(time_cosmic_code))
        boundary_comoving_code = boundary_proper_code / scale_factor
        x_comoving_code = x_proper_code / scale_factor
        width_comoving_code = width_proper_code / scale_factor
        area_comoving_code = area_proper_code / scale_factor**2
        volume_comoving_code = volume_proper_code / scale_factor**3
        rho_comoving_code = rho_proper_code.copy()
        vel_supercomoving_code = vel_proper_code / scale_factor
        temp_supercomoving_code = temp_proper_code * scale_factor ** (
            3.0 * (gamma - 1.0)
        )
        pressure_supercomoving_code = sim.fluid.eos.pressure(
            rho_comoving_code,
            temp_supercomoving_code,
            mu_dimensionless_array,
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
        sim.mesh.x_comoving_code = x_comoving_code
        sim.mesh.boundary_comoving_code = boundary_comoving_code
        sim.mesh.width_comoving_code = width_comoving_code
        sim.mesh.area_comoving_code = area_comoving_code
        sim.mesh.volume_comoving_code = volume_comoving_code
        sim.fluid.rho_comoving_code = rho_comoving_code
        sim.fluid.vel_supercomoving_code = vel_supercomoving_code
        sim.fluid.pre_supercomoving_code = pressure_supercomoving_code
        sim.fluid.temp_supercomoving_code = temp_supercomoving_code
        sim.fluid.tau_supercomoving_code = tau_supercomoving_code
        sim.mesh.geometry_state = MeshGeometryState.from_arrays(
            SUPERCOMOVING_RUNTIME_FIELDS,
            x_comoving_code=x_comoving_code,
            boundary_comoving_code=boundary_comoving_code,
            width_comoving_code=width_comoving_code,
            area_comoving_code=area_comoving_code,
            volume_comoving_code=volume_comoving_code,
        )
        sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
            SUPERCOMOVING_RUNTIME_FIELDS,
            rho_comoving_code=rho_comoving_code,
            vel_supercomoving_code=vel_supercomoving_code,
            pre_supercomoving_code=pressure_supercomoving_code,
            temp_supercomoving_code=temp_supercomoving_code,
            tau_supercomoving_code=tau_supercomoving_code,
            mu_dimensionless=mu_dimensionless_array,
        )
        expected = {
            "boundary_comoving_code": boundary_comoving_code,
            "rho_comoving_code": rho_comoving_code,
            "vel_supercomoving_code": vel_supercomoving_code,
            "temp_supercomoving_code": temp_supercomoving_code,
        }
    else:
        pressure_proper_code = sim.fluid.eos.pressure(
            rho_proper_code, temp_proper_code, mu_dimensionless_array
        )
        sim.mesh.x_proper_code = x_proper_code
        sim.mesh.boundary_proper_code = boundary_proper_code
        sim.mesh.width_proper_code = width_proper_code
        sim.mesh.area_proper_code = area_proper_code
        sim.mesh.volume_proper_code = volume_proper_code
        sim.fluid.rho_proper_code = rho_proper_code
        sim.fluid.vel_proper_code = vel_proper_code
        sim.fluid.pre_proper_code = pressure_proper_code
        sim.fluid.temp_proper_code = temp_proper_code
        sim.fluid.time_proper_code = 0.0
        sim.mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            x_proper_code=x_proper_code,
            boundary_proper_code=boundary_proper_code,
            width_proper_code=width_proper_code,
            area_proper_code=area_proper_code,
            volume_proper_code=volume_proper_code,
        )
        sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=rho_proper_code,
            vel_proper_code=vel_proper_code,
            pre_proper_code=pressure_proper_code,
            temp_proper_code=temp_proper_code,
            time_proper_code=0.0,
            mu_dimensionless=mu_dimensionless_array,
        )
        expected = {
            "boundary_proper_code": boundary_proper_code,
            "rho_proper_code": rho_proper_code,
            "vel_proper_code": vel_proper_code,
            "temp_proper_code": temp_proper_code,
        }
    return sim, expected


def _assert_roundtrip(case_config, output_filename):
    initial, expected = _build_initial_condition(case_config)
    rio.writehdf5(initial, output_filename)
    restored = Rsim(case_config["par"])
    rio.readhdf5(
        restored.par,
        restored.mesh,
        restored.fluid,
        output_filename,
    )
    code_units = restored.par.units.CodeUnits
    assert code_units.name == case_config["example"]["_active_case_name"]
    for field_name, expected_values in expected.items():
        if field_name.startswith("boundary_"):
            actual_values = getattr(restored.mesh, field_name)
        else:
            actual_values = getattr(restored.fluid, field_name)
        np.testing.assert_allclose(actual_values, expected_values)
    if case_config["par"]["gravity"]["cosmological_expansion"]:
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
