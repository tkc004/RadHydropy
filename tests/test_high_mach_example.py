# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0

import warnings
from types import SimpleNamespace

import numpy as np

from example.HighMachAdvection1D.tools import entropy_profile


def test_entropy_profile_masks_vacuum_cells_without_warning():
    state = SimpleNamespace(
        par=SimpleNamespace(
            mesh=SimpleNamespace(ghost_cells=1, grid_cells=3),
            hydrodynamics=SimpleNamespace(gamma=5.0 / 3.0),
        ),
        mesh=SimpleNamespace(
            boundary_proper_code=np.arange(6, dtype=float),
        ),
        fluid=SimpleNamespace(
            rho_proper_code=np.array([1.0, 2.0, 0.0, 4.0, 1.0]),
            temp_proper_code=np.ones(5),
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, entropy = entropy_profile(state)

    assert np.isfinite(entropy[[0, 2]]).all()
    assert np.isnan(entropy[1])
