# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
from types import SimpleNamespace  # noqa: CPY001
from unittest.mock import Mock, patch

import numpy as np

from radhydropy.solver.boundaries import set_boundary
from radhydropy.solver.radiation import apply_radiation_pressure


def test_set_boundary_dispatches_periodic_conditions_at_module_boundary():
    solver = Mock()
    solver.apply_periodic_boundary = Mock()
    mesh = SimpleNamespace()
    fluid = SimpleNamespace(runtime_fields=object())
    par = SimpleNamespace(
        boundary=SimpleNamespace(condition="Periodic"),
        CodeUnits=None,
        mesh=SimpleNamespace(ghost_cells=2, grid_cells=4),
    )

    set_boundary(solver, mesh, fluid, par)

    solver.ApplyHydrostaticCore.assert_called_once_with(mesh, fluid, par)
    solver.apply_periodic_boundary.assert_called_once()
    args = solver.apply_periodic_boundary.call_args.args
    assert args[1:3] == (slice(2, 6), slice(0, 2))
    assert args[4] == 2  # noqa: PLR2004


def test_apply_radiation_pressure_updates_conserved_arrays_at_module_boundary():
    solver = Mock()
    solver.interior_slice.return_value = slice(0, 1)
    solver.active_primitive_arrays.return_value = (
        np.ones(1),
        np.zeros(1),
        np.ones(1),
        np.ones(1),
    )
    solver.geometry_state.return_value = SimpleNamespace(
        volume_runtime_code=np.ones(1),
    )
    fluid = SimpleNamespace(Mom_code=np.zeros(1), Energy_code=np.zeros(1))
    par = SimpleNamespace(
        radiation_pressure=True,
        radiation_pressure_efficiency=1.0,
        mesh=SimpleNamespace(grid_cells=1),
        units=SimpleNamespace(CodeUnits=None),
    )
    # Unit scales of one make the expected update directly interpretable.
    par.CodeUnits = SimpleNamespace(
        unit_conversion={
            "density_cgs_g_cm3": 1.0,
            "acceleration_cgs_cm_s2": 1.0,
        },
    )
    source_result = {
        "absorbed_photon_rate": np.array([2.0]),
        "photon_energy_cgs_erg": np.array([3.0]),
        "direction": 1,
    }

    with patch(
        "radhydropy.solver.radiation.code_unit_scales",
        return_value={
            "density_cgs_g_cm3": 1.0,
            "acceleration_cgs_cm_s2": 1.0,
        },
    ):
        applied = apply_radiation_pressure(
            solver,
            4.0,
            SimpleNamespace(),
            fluid,
            par,
            source_result,
        )

    assert applied == 1
    assert fluid.Mom_code[0] > 0.0
    np.testing.assert_allclose(fluid.Energy_code, 0.0)
