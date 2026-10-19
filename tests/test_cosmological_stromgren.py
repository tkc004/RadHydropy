# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Focused tests for the cosmological Strömgren reference solution."""

import numpy as np
import unyt

from example.CosmologicalStromgrenSphere1D.cosmological_stromgren_analytic import (
    front_radius_comoving,
)


def test_cosmological_front_starts_at_zero_and_grows():
    radius = front_radius_comoving(
        np.array([0.1, 0.2, 0.5]),
        0.1,
        0.30116930091678135 * unyt.Gyr,
        5.0e48 / unyt.s,
        5.5e-4 / unyt.cm**3,
        2.59e-13 * unyt.cm**3 / unyt.s,
    )
    values = radius.to_value(unyt.kpc)
    assert values[0] == 0.0
    assert np.all(np.diff(values) > 0.0)
