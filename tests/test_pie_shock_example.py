# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0

import unyt

from example.PIERadiativeShockTube1D.pie_radiative_shock_tube_1d import (
    _select_upstream_properties,
)

EXPECTED_DENSITY = 1.0e-27
EXPECTED_VELOCITY = 1.0e7


def test_pie_shock_diagnostic_uses_initial_upstream_fallback():
    density, velocity, source = _select_upstream_properties(
        {
            "rho_proper_cgs_g_cm3": EXPECTED_DENSITY,
            "upstream_velocity_cgs_cm_s": 100.0 * unyt.km / unyt.s,
        },
        density_proper_cgs_g_cm3=[1.0, 2.0],
        velocity_proper_cgs_cm_s=[-3.0, 4.0],
        upstream_slice=[False, False],
    )

    assert density == EXPECTED_DENSITY
    assert velocity == EXPECTED_VELOCITY
    assert source == "initial_condition_fallback"
