# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Small, explicit representation-specific parameter fixtures."""

from types import SimpleNamespace


def parameter_namespace(**values):
    """Build a lightweight parameter object with real nested groups."""
    par = SimpleNamespace(**values)
    par.mesh = SimpleNamespace(
        ghost_cells=values.get("noghost", 0),
        grid_cells=values.get("nogrid"),
        area_proper=values.get("area_proper"),
    )
    par.simulation = SimpleNamespace(
        coordinate_system=values.get("coordsys"),
        final_time=values.get("timesim"),
        initial_condition_filename=values.get("ICfilename"),
        time_proper_code=values.get("time_proper_code"),
        tau_supercomoving_code=values.get("tau_supercomoving_code"),
        box_size_proper_code=values.get("box_size_proper"),
        box_size_comoving_code=values.get("box_size_proper"),
    )
    par.hydrodynamics = SimpleNamespace(
        eos_type=values.get("EOStype"),
        gamma=values.get("gamma", 5.0 / 3.0),
        CFL=values.get("CFL"),
        order=values.get("order"),
        riemann_solver=values.get("riemann_solver"),
    )
    par.units = SimpleNamespace(CodeUnits=values.get("CodeUnits"))
    par.boundary = SimpleNamespace(
        condition=values.get("boundcond"),
        rho_inflow_proper=values.get("rho_inflow_proper"),
        vel_inflow_proper=values.get("vel_inflow_proper"),
        temperature_inflow_proper=values.get("temperature_inflow_proper"),
        inflow_mu=values.get("mu_inflow"),
        rho_outflow_proper=values.get("rho_outflow_proper"),
        vel_outflow_proper=values.get("vel_outflow_proper"),
        temperature_outflow_proper=values.get("temperature_outflow_proper"),
        outflow_mu=values.get("mu_outflow"),
    )
    par.timestep = SimpleNamespace(
        dtmin=values.get("dtmin"),
        dtmax=values.get("dtmax"),
    )
    par.output = SimpleNamespace(
        directory=values.get("outdir"),
        filename_prefix=values.get("outfileprefix"),
        cadence=values.get("outdeltatime"),
        time_list_filename=values.get("outputtimefilename"),
    )
    radiation = values.get("radiation")
    radiation_boundary_flux = values.get("radiative_transfer_boundary_flux")
    if radiation_boundary_flux is None and radiation is not None:
        radiation_boundary_flux = getattr(radiation, "boundary_flux", None)
    source_photon_rate = values.get("source_photon_rate")
    if source_photon_rate is None and radiation is not None:
        source_photon_rate = getattr(radiation, "source_photon_rate", None)
    par.radiation = SimpleNamespace(
        radiative_transfer=values.get("radiative_transfer"),
        method=values.get("radiative_transfer_method"),
        temporal_scheme=values.get("radiative_transfer_temporal_scheme"),
        direction=values.get("radiative_transfer_direction", 1),
        boundary_flux=radiation_boundary_flux,
        source_photon_rate=source_photon_rate,
        boundary_flux_groups=values.get("radiative_transfer_boundary_flux_groups"),
        source_photon_rate_groups=values.get("source_photon_rate_groups"),
    )
    return par
