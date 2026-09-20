"""Boundary-condition orchestration for the hydrodynamics solver."""

from radhydropy.units import code_unit_scales


def set_boundary(solver, mesh, fluid, par):
    """Fill ghost cells according to the configured boundary condition."""
    if getattr(fluid, "runtime_fields", None) is None:
        raise ValueError("boundary update requires configured fluid runtime state")
    solver.ApplyHydrostaticCore(mesh, fluid, par)
    boundary_type = par.boundary.condition
    code_units = getattr(par, "CodeUnits", None)
    scales = code_unit_scales(code_units)
    ghost_cells = int(par.mesh.ghost_cells)
    grid_cells = int(par.mesh.grid_cells)
    last_cell = ghost_cells + grid_cells - 1
    first_cell = ghost_cells
    right_start = ghost_cells + grid_cells
    interior = slice(first_cell, right_start)
    left_ghost = slice(0, ghost_cells)
    right_ghost = slice(right_start, right_start + ghost_cells)

    if boundary_type == "Periodic":
        solver._apply_periodic_boundary(
            fluid, interior, left_ghost, right_ghost, ghost_cells,
        )
    elif boundary_type == "Open":
        solver._apply_open_boundary(
            fluid, first_cell, last_cell, left_ghost, right_ghost,
        )
    elif boundary_type == "Reflecting":
        solver._apply_reflecting_boundary(
            fluid, interior, left_ghost, right_ghost, ghost_cells,
        )
    elif boundary_type == "OpenSph":
        solver._apply_open_spherical_boundary(
            mesh, fluid, par, scales, first_cell, last_cell,
            left_ghost, right_ghost, ghost_cells,
        )
    elif boundary_type == "InflowSph":
        solver._apply_inflow_spherical_boundary(
            mesh, fluid, par, scales, first_cell, last_cell,
            left_ghost, right_ghost, ghost_cells,
        )
    elif boundary_type == "OutflowSph":
        solver._apply_outflow_spherical_boundary(
            mesh, fluid, par, scales, first_cell, last_cell,
            left_ghost, right_ghost, ghost_cells,
        )
    elif boundary_type == "WindSph":
        solver._apply_wind_spherical_boundary(
            mesh, fluid, par, scales, first_cell, last_cell,
            left_ghost, right_ghost, ghost_cells,
        )
    else:
        raise ValueError("Boundary condition unknown: %s" % boundary_type)
