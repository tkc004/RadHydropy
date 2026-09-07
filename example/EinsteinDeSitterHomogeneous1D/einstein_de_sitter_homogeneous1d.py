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
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu


def main(config_filename=Path(__file__).with_name("einstein_de_sitter_homogeneous1d.yaml")):
    config = eu.load_nested_example_config(config_filename)
    par_config = config['par']
    units = CodeUnits.from_mapping(par_config['units']['CodeUnits'])
    cosmology = EinsteinDeSitter.from_code_units(units)
    t0 = quantity_to_value(
        par_config['simulation']['initial_time'], units.time_unit
    )
    t1 = quantity_to_value(
        par_config['simulation']['final_time'], units.time_unit
    )
    initial_condition = config['initial_condition']
    tau0 = cosmology.supercomoving_time(t0)
    sim = Rsim(par_config)
    sim.par.tau_supercomoving_code = tau0
    sim.par.simulation.tau_supercomoving_code = tau0
    sim.fluid.tau_supercomoving_code = tau0
    sim.fluid.rho_comoving_code = np.array([
        quantity_to_value(initial_condition['density'], units.density_unit)
    ])
    sim.fluid.vel_supercomoving_code = np.array([
        quantity_to_value(initial_condition['velocity'], units.velocity_unit)
    ])
    sim.fluid.pre_supercomoving_code = np.array([
        quantity_to_value(initial_condition['pressure'], units.pressure_unit)
    ])
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
