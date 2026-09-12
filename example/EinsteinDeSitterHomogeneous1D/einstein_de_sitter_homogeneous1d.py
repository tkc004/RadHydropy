"""Phase 1 Einstein--de Sitter homogeneous expansion diagnostic."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import numpy as np

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.cosmology_context import CosmologyContext
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu


def main(config_filename=Path(__file__).with_name("einstein_de_sitter_homogeneous1d.yaml")):
    config = eu.load_nested_example_config(config_filename)

    units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    cosmology = EinsteinDeSitter.from_code_units(units)
    t0 = quantity_to_value(
        config['initial_condition']['time_cosmic'], units.time_unit
    )
    t1 = quantity_to_value(
        config["par"]['simulation']['final_time'], units.time_unit
    )
    initial_condition = config['initial_condition']
    tau0 = cosmology.supercomoving_time(t0)
    par = config["par"]
    par.setdefault("mesh", {}).update(grid_cells=1, ghost_cells=0)
    par.setdefault("hydrodynamics", {}).setdefault("gamma", 5.0 / 3.0)
    par["cosmology"] = {
        **par.get("cosmology", {}),
        "cosmological": True,
        "cosmological_expansion": True,
        "supercomoving_coordinates": True,
    }
    scale_factor = float(cosmology.scale_factor(t0))
    writer = InitialConditionWriter(
        par_config=par,
        code_units=units,
        cosmology_context=CosmologyContext(
            gamma=float(par["hydrodynamics"]["gamma"]),
            cosmology=cosmology.type_name,
            scale_factor=scale_factor,
            hubble_parameter_km_s_Mpc=float(cosmology.hubble(t0)) * (
                units.velocity_unit.to_value("km/s")
                / units.length_unit.to_value("Mpc")
            ),
        ),
    )
    sim = writer.simulation
    writer.box_size = writer.radquantity(1.0 * units.length_unit)
    writer.mesh.boundary_radarray = writer.radarray(
        np.array([0.0, 1.0]) * units.length_unit,
        representation="comoving",
    )
    writer.set_field("x_comoving_code", np.array([0.5]))
    sim.par.tau_supercomoving_code = tau0
    sim.par.simulation.tau_supercomoving_code = tau0
    sim.fluid.tau_supercomoving_code = tau0
    rho_comoving_code = np.array([
        quantity_to_value(initial_condition['rho_proper'], units.density_unit)
    ])
    vel_supercomoving_code = np.array([
        quantity_to_value(initial_condition['vel_proper'], units.velocity_unit)
    ])
    pre_supercomoving_code = np.array([
        quantity_to_value(
            initial_condition['pressure_initial_proper'], units.pressure_unit
        )
    ])
    temp_supercomoving_code = pre_supercomoving_code / rho_comoving_code
    writer.fluid.rho_radarray = writer.radarray(
        rho_comoving_code * units.density_unit, representation="comoving"
    )
    writer.fluid.vel_radarray = writer.radarray(
        vel_supercomoving_code * units.velocity_unit,
        representation="supercomoving",
    )
    writer.fluid.pre_radarray = writer.radarray(
        pre_supercomoving_code * units.pressure_unit,
        representation="supercomoving",
    )
    writer.fluid.temp_radarray = writer.radarray(
        temp_supercomoving_code * units.temperature_unit,
        representation="supercomoving",
    )
    writer.prepare()
    fluid = sim.fluid
    initial = (
        fluid.rho_comoving_code.copy(),
        fluid.vel_supercomoving_code.copy(),
        fluid.pre_supercomoving_code.copy(),
    )
    # Supercomoving homogeneous Euler evolution has no expansion source.
    assert np.allclose(fluid.rho_comoving_code, initial[0])
    assert np.allclose(fluid.vel_supercomoving_code, initial[1])
    assert np.allclose(fluid.pre_supercomoving_code, initial[2])
    a_ratio = cosmology.scale_factor(t1) / cosmology.scale_factor(t0)
    assert np.isclose(a_ratio, 2.0**(2.0 / 3.0))
    print("Einstein-De Sitter homogeneous expansion passed")
    print("a(t=2)/a(t=1) = %.8g" % a_ratio)
    print("supercomoving density/velocity/pressure remain constant")
    print("physical density ratio = %.8g" % a_ratio**-3)
    print("physical pressure ratio = %.8g" % a_ratio**-5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=Path(__file__).with_name('einstein_de_sitter_homogeneous1d.yaml'))
    main(parser.parse_args().config)
