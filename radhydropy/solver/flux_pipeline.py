# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Interface-flux orchestration for the finite-volume solver."""

import logging
from typing import Any

import numpy as np

import radhydropy.utils as ru
from radhydropy.diagnostic_logging import log_diagnostic


def set_interface_flux(
    solver: Any,
    mesh: Any,
    fluid: Any,
    boundcond: Any,
    method: str = "Rusanov",
    verbose: Any = None,
    order: int = 0,
) -> None:
    """Construct primary and optional interface fluxes."""
    if verbose is None:
        verbose = 0
    geometry = solver.geometry_state(
        mesh,
        getattr(mesh, "par", getattr(mesh, "_par", None)),
    )
    if method in ("GLF", "Rusanov", "HLLC"):
        if method == "GLF":
            # Global Lax Friedrich scheme.
            fluid.cmax = geometry.width_runtime_code / np.amin(solver.dt)
        elif method == "Rusanov":
            # Local Lax Friedrich scheme.
            fluid.cmax = np.maximum(
                fluid.vsignal_code,
                ru.periodic_roll(fluid.vsignal_code, 1),
            )
        else:  # HLLC uses Rusanov speeds for CFL and vacuum fallback.
            fluid.cmax = np.maximum(
                fluid.vsignal_code,
                ru.periodic_roll(fluid.vsignal_code, 1),
            )

        solver.SetFaceLR(mesh, fluid, boundcond, order=order)
        solver.SetFluxOnFace(
            fluid,
            boundcond,
            order=order,
            par=getattr(mesh, "par", getattr(mesh, "_par", None)),
            method=method,
        )
        solver.apply_low_density_flux_mask(
            fluid,
            getattr(mesh, "par", getattr(mesh, "_par", None)),
        )
        solver.apply_hydrostatic_core_flux(
            fluid,
            getattr(mesh, "par", getattr(mesh, "_par", None)),
        )
        solver.zero_spherical_origin_flux(mesh, fluid)
        solver.apply_local_angular_energy_fallback(
            mesh,
            fluid,
            getattr(mesh, "par", getattr(mesh, "_par", None)),
        )
        angular_momentum_face = solver.set_angular_momentum_flux(
            fluid,
            order=order,
        )
        solver.set_rotational_energy_flux(
            mesh,
            fluid,
            getattr(mesh, "par", getattr(mesh, "_par", None)),
            j_face=angular_momentum_face,
        )
        # Optional fluxes are constructed after the primary hydro fluxes;
        # enforce the exact-origin condition once more at the end so a
        # later reconstruction cannot repopulate that face.
        solver.zero_spherical_origin_flux(mesh, fluid)
    else:
        raise ValueError(f"Interface flux method unknown: {method}")
    if verbose >= 2:  # noqa: PLR2004
        log_diagnostic(
            logging.DEBUG,
            "interface_fluxes_constructed",
            mass_flux=fluid.Mass_code.flux,
            momentum_flux=fluid.Mom_code.flux,
            energy_flux=fluid.Energy_code.flux,
        )
