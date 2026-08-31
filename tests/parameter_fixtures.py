"""Small, explicit nested parameter fixtures for focused unit tests.

The production code consumes nested parameter groups.  These fixtures retain
the flat names only where a test is exercising file-format metadata or legacy
input construction; the groups themselves are ordinary namespaces, not
attribute-forwarding compatibility proxies.
"""

from types import SimpleNamespace


def parameter_namespace(**values):
    """Build a lightweight parameter object with real nested groups."""
    par = SimpleNamespace(**values)
    par.mesh = SimpleNamespace(
        ghost_cells=values.get("noghost", 0),
        grid_cells=values.get("nogrid"),
        area=values.get("area"),
    )
    par.simulation = SimpleNamespace(
        coordinate_system=values.get("coordsys"),
        final_time=values.get("timesim"),
        initial_condition_filename=values.get("ICfilename"),
        current_time=values.get("time", values.get("fluid_time")),
        box_size=values.get("boxsize"),
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
        inflow_density=values.get("rho_inflow"),
        inflow_velocity=values.get("vel_inflow"),
        inflow_temperature=values.get("temp_inflow"),
        inflow_mu=values.get("mu_inflow"),
        outflow_density=values.get("rho_outflow"),
        outflow_velocity=values.get("vel_outflow"),
        outflow_temperature=values.get("temp_outflow"),
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
    par.radiation = SimpleNamespace(
        radiative_transfer=values.get("radiative_transfer"),
        method=values.get("radiative_transfer_method"),
        temporal_scheme=values.get("radiative_transfer_temporal_scheme"),
        direction=values.get("radiative_transfer_direction", 1),
        boundary_flux=values.get("radiative_transfer_boundary_flux"),
        source_photon_rate=values.get("radiative_transfer_source_photon_rate"),
        boundary_flux_groups=values.get("radiative_transfer_boundary_flux_groups"),
        source_photon_rate_groups=values.get("radiative_transfer_source_photon_rate_groups"),
    )
    return par
