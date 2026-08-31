"""Rsim execution subsystem helpers."""

import numpy as np


def _initialize_runtime_state(sim):
    """Initialize counters and per-cell diagnostic state for this run."""
    sim.energy_diagnostics_enabled = bool(
        getattr(sim.par, "energy_diagnostics", False)
    )
    sim.cumulative_hydro_boundary_energy = 0.0
    sim.cumulative_gravity_work = 0.0
    sim.cumulative_gravity_potential_change = 0.0
    sim.cumulative_gravity_potential_flux = 0.0
    diagnostic_count = int(sim.par.mesh.grid_cells or len(getattr(sim.fluid, 'Energy', [])))
    for name in (
        "cumulative_gravity_work_by_cell",
        "cumulative_hydro_energy_change_by_cell",
        "cumulative_thermochemistry_energy_change_by_cell",
        "cumulative_compression_work_by_cell",
        "cumulative_shock_work_by_cell",
    ):
        setattr(sim, name, np.zeros(diagnostic_count, dtype=float))
    sim.last_dark_matter_substeps = 0
    sim.cumulative_dark_matter_substeps = 0
    sim.dark_matter_substep_history = []
