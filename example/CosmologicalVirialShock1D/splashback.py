# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Shared splashback-radius diagnostic for cosmological virial-shock examples."""

import numpy as np


def _prepare_splashback_profile(
    dm_radius_proper_code,
    dm_mass_comoving_code,
    rvir_proper_code,
    bin_count,
):
    if not np.isfinite(rvir_proper_code) or float(rvir_proper_code) <= 0.0:
        return None
    radius_proper_code = np.asarray(dm_radius_proper_code, dtype=float)
    mass_comoving_code = np.asarray(dm_mass_comoving_code, dtype=float)
    valid = (
        np.isfinite(radius_proper_code)
        & np.isfinite(mass_comoving_code)
        & (radius_proper_code > 0.0)
        & (mass_comoving_code > 0.0)
    )
    radius_proper_code = radius_proper_code[valid]
    mass_comoving_code = mass_comoving_code[valid]
    if radius_proper_code.size < 16:  # noqa: PLR2004
        return None
    order = np.argsort(radius_proper_code)
    radius_proper_code = radius_proper_code[order]
    mass_comoving_code = mass_comoving_code[order]
    edges = np.geomspace(
        max(radius_proper_code[0] * 0.9, 1.0e-12),
        radius_proper_code[-1] * 1.1,
        int(max(32, bin_count)) + 1,
    )
    shell_mass_comoving_code, _ = np.histogram(
        radius_proper_code,
        bins=edges,
        weights=mass_comoving_code,
    )
    shell_volume = 4.0 * np.pi / 3.0 * np.diff(edges**3)
    rho_comoving_code = shell_mass_comoving_code / np.maximum(shell_volume, 1.0e-300)
    occupied = rho_comoving_code > 0.0
    if np.count_nonzero(occupied) < 12:  # noqa: PLR2004
        return None
    radii = np.sqrt(edges[:-1] * edges[1:])[occupied]
    return radii, rho_comoving_code[occupied]


def splashback_radius(
    dm_radius_proper_code,
    dm_mass_comoving_code,
    rvir_proper_code=np.nan,
    bin_count=128,
):
    """Estimate splashback from the steepest outer DM density slope."""
    profile = _prepare_splashback_profile(
        dm_radius_proper_code,
        dm_mass_comoving_code,
        rvir_proper_code,
        bin_count,
    )
    if profile is None:
        return float("nan")
    radii, rho_comoving_code = profile
    log_radius = np.log(radii)
    log_density = np.log(rho_comoving_code)
    window = min(7, log_density.size if log_density.size % 2 else log_density.size - 1)
    if window >= 3:  # noqa: PLR2004
        padded = np.pad(log_density, (window // 2,), mode="edge")
        log_density = np.convolve(
            padded,
            np.ones(window) / float(window),
            mode="valid",
        )
    slope = np.gradient(log_density, log_radius)
    lower = max(float(rvir_proper_code), radii[0])
    upper = min(3.0 * float(rvir_proper_code), 0.95 * radii[-1])
    if upper > lower:
        candidates = np.flatnonzero((radii >= lower) & (radii <= upper))
        if candidates.size >= 3:  # noqa: PLR2004
            local = candidates[np.argmin(slope[candidates])]
            if np.isfinite(slope[local]) and slope[local] <= -1.0:
                return float(radii[local])
    return float("nan")
