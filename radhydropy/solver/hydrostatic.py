"""Fixed pressure-supported central-core operations."""

import numpy as np

from radhydropy.arrays import as_named_array
from radhydropy.runtime_fields import runtime_fields


def hydrostatic_core_enabled(par):
    return str(getattr(par, "gas_core_model", "none")).lower() in (
        "hydrostatic",
        "hydrostatic_fixed",
        "fixed_hydrostatic",
    )


def initialize_hydrostatic_core(solver, mesh, fluid, par):
    """Initialize an optional fixed, pressure-supported central core."""
    if not hydrostatic_core_enabled(par):
        return
    if getattr(mesh, "coordsys", None) != "spherical":
        raise ValueError("gas_core_model requires a spherical mesh")
    radius = getattr(par, "radius_core_proper", None)
    if radius is None or float(radius) <= 0.0:
        raise ValueError("radius_core_proper must be positive for gas_core_model")
    first = int(par.mesh.ghost_cells)
    last = first + int(par.mesh.grid_cells)
    geometry = solver._geometry_state(mesh, par)
    coordinate = np.asarray(
        geometry.coordinate_runtime_code[first:last],
        dtype=float,
    )
    core_local = coordinate < float(radius)
    if not np.any(core_local) or np.all(core_local):
        raise ValueError(
            "radius_core_proper must contain at least one, but not all, resolved cells",
        )
    core = np.zeros(len(geometry.coordinate_runtime_code), dtype=bool)
    core[first:last] = core_local
    core_indices = np.flatnonzero(core)
    state = {
        "core_mask": core,
        "core_indices": core_indices,
        "core_last": int(core_indices[-1]),
    }
    fields = runtime_fields(par)
    primitive_names = (
        fields.density,
        fields.velocity,
        fields.temperature,
        fields.pressure,
    )
    for name in (
        *primitive_names,
        "mu",
        "xHI",
        "xHeI",
        "xHeII",
        "xHeIII",
        "specific_angular_momentum_code",
    ):
        if hasattr(fluid, name):
            state[name] = np.asarray(getattr(fluid, name)[core], dtype=float).copy()
    fluid._hydrostatic_core = state
    par._hydrostatic_core_mask = core
    par._hydrostatic_core_face = int(core_indices[-1] + 1)


def apply_hydrostatic_core(solver, mesh, fluid, par):
    """Restore the fixed core state before a resolved-halo update."""
    state = getattr(fluid, "_hydrostatic_core", None)
    if state is None:
        return
    core = state["core_mask"]
    fields = runtime_fields(par)
    primitive_names = (
        fields.density,
        fields.velocity,
        fields.temperature,
        fields.pressure,
    )
    for name in (
        *primitive_names,
        "mu",
        "xHI",
        "xHeI",
        "xHeII",
        "xHeIII",
        "specific_angular_momentum_code",
    ):
        if name in state and hasattr(fluid, name):
            values = np.asarray(getattr(fluid, name), dtype=float).copy()
            values[core] = state[name]
            setattr(fluid, name, as_named_array(values))
    # A fixed core is hydrostatic and has no resolved radial motion.
    solver._active_primitive_arrays(fluid, par)[1][core] = 0.0


def apply_hydrostatic_core_flux(solver, fluid, par):
    """Close the resolved halo with a pressure-bearing, no-mass-flux core."""
    face = getattr(par, "_hydrostatic_core_face", None)
    state = getattr(fluid, "_hydrostatic_core", None)
    if face is None or state is None:
        return
    core_last = state["core_last"]
    # The core is fixed-mass: pressure acts on the halo, but gas, energy,
    # and radial momentum do not cross the core/halo interface.
    fluid.Mass_code.flux[face] = 0.0
    fluid.Energy_code.flux[face] = 0.0
    _, _, pressure_runtime_code, _ = solver._active_primitive_arrays(
        fluid,
        par,
    )
    fluid.Mom_code.flux[face] = pressure_runtime_code[core_last]
