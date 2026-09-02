import numpy as np
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

from radhydropy.eos import EOS
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits


CODE_UNITS = CodeUnits.from_mapping({
    "name": "hydrostatic_core_test_units",
    "InternalUnitSystem": {
        "UnitMass_in_cgs": 1.0,
        "UnitLength_in_cgs": 1.0,
        "UnitVelocity_in_cgs": 1.0,
        "UnitCurrent_in_cgs": 1.0,
        "UnitTemp_in_cgs": 1.0,
    },
})


def _core_problem(model="hydrostatic_fixed"):
    mesh = SimpleNamespace(
        coordsys="spherical",
        coordinate=np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
    )
    fluid = SimpleNamespace(
        rho_code=np.ones(6),
        vel_code=np.zeros(6),
        temp_code=np.ones(6),
        mu=np.ones(6),
        pre_code=np.ones(6),
    )
    par = parameter_namespace(
        gas_core_model=model,
        gas_core_radius=10.0,
        noghost=1,
        nogrid=4,
    )
    return mesh, fluid, par


def test_hydrostatic_core_is_opt_in_and_masks_only_inner_cells():
    solver = Solver()
    mesh, fluid, par = _core_problem()
    solver.InitializeHydrostaticCore(mesh, fluid, par)

    assert par._hydrostatic_core_face == 4
    np.testing.assert_array_equal(
        par._hydrostatic_core_mask,
        [False, True, True, True, False, False],
    )

    fluid.rho_code[1:4] = 7.0
    fluid.vel_code[1:4] = 3.0
    solver.ApplyHydrostaticCore(mesh, fluid, par)
    np.testing.assert_array_equal(fluid.rho_code[1:4], 1.0)
    np.testing.assert_array_equal(fluid.vel_code[1:4], 0.0)


def test_default_core_model_does_not_create_core_state():
    solver = Solver()
    mesh, fluid, par = _core_problem(model="none")
    solver.InitializeHydrostaticCore(mesh, fluid, par)
    assert not hasattr(fluid, "_hydrostatic_core")
