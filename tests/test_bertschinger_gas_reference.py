"""Focused checks for the standalone Bertschinger gas reference."""

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (Path(__file__).parents[1] / 'example' /
               'BertschingerGasReference' / 'bertschinger_gas.py')
SPEC = importlib.util.spec_from_file_location('bertschinger_gas_reference', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_transonic_shock_location():
    shock = MODULE.shoot_shock_lambda()
    assert abs(shock - 0.33897694) < 3.0e-7


def test_shock_jump_and_standalone_solution():
    shock = MODULE.shoot_shock_lambda()
    exterior = MODULE.exterior_solution([shock])
    postshock = MODULE.shock_jump(
        (exterior[0][0], exterior[1][0], exterior[2][0]), shock)
    assert abs(postshock[0] / exterior[0][0] - 4.0) < 1.0e-10
    solution = MODULE.solve_bertschinger_gas(points=128)
    assert solution.shock_lambda == shock
    assert solution.lambda_in[0] < shock
    assert solution.velocity_in[0] < 1.0e-8
