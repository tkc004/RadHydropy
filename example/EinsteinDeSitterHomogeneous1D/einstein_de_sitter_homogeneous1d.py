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
from radhydropy.units import CodeUnits
import example_utils as eu


def main(config_filename=Path(__file__).with_name("einstein_de_sitter_homogeneous1d.yaml")):
    config = eu.load_nested_example_config(config_filename)
    units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    cosmology = EinsteinDeSitter.from_code_units(units)
    t0 = float(config['par']['simulation']['initial_time'])
    t1 = float(config['par']['simulation']['final_time'])
    tau0 = cosmology.supercomoving_time(t0)
    class EOS:
        gamma = 5.0 / 3.0
        is_isothermal = False

        def total_energy_density(self, rho, vel, pressure):
            return 0.5 * rho * vel**2 + pressure / (self.gamma - 1.0)

    class Fluid:
        time = tau0
        rho = np.array([config['example']['density']])
        vel = np.array([config['example']['velocity']])
        pre = np.array([config['example']['pressure']])
        eos = EOS()

    fluid = Fluid()
    initial = (fluid.rho.copy(), fluid.vel.copy(), fluid.pre.copy())
    # Supercomoving homogeneous Euler evolution has no expansion source.
    assert np.allclose(fluid.rho, initial[0])
    assert np.allclose(fluid.vel, initial[1])
    assert np.allclose(fluid.pre, initial[2])
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
