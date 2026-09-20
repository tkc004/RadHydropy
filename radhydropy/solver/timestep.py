# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""CFL timestep calculation for the finite-volume solver."""  # noqa: CPY001

import logging

import numpy as np

from radhydropy.diagnostic_logging import log_diagnostic
from radhydropy.runtime_fields import (
    select_fluid_primitive_arrays,
    select_mesh_geometry_arrays,
)


def get_time_step(solver, mesh, fluid, par, CFL=None):
    """Return a CFL-limited timestep in the active time coordinate."""
    if CFL is None:
        CFL = par.hydrodynamics.CFL  # noqa: N806
    geometry = getattr(mesh, "geometry_state", None)
    if geometry is not None:
        mesh_coordinate, _, width_runtime_code, area_runtime_code, volume_runtime_code = (
            select_mesh_geometry_arrays(geometry, par)
        )
    else:
        raise ValueError("timestep calculation requires typed mesh geometry state")
    fluid.SetSoundSpeed()
    runtime_state = getattr(fluid, "runtime_state", None)
    runtime_state = fluid if runtime_state is None else runtime_state
    density_field, velocity, pressure_field, _, time_runtime_code = select_fluid_primitive_arrays(
        runtime_state,
        par,
    )
    vsignal = np.absolute(velocity) + fluid.cs_code
    density = np.asarray(density_field, dtype=float)
    if width_runtime_code.shape != vsignal.shape:
        interior = solver.interior_slice(par)
        if width_runtime_code[interior].shape == vsignal.shape:
            width_runtime_code = width_runtime_code[interior]
            density = density[interior]
        elif vsignal[interior].shape == width_runtime_code.shape:
            vsignal = vsignal[interior]
            density = density[interior]

    # Ghost zones are needed by the Riemann solve but must not determine
    # the CFL step.  In particular, reflecting/outflow boundary updates
    # can leave a ghost velocity temporarily very large while the active
    # solution remains valid.  Keep the full signal-speed array for later
    # flux work, and reduce only over the active cells here.
    active_width_runtime_code = width_runtime_code
    active_density = density
    active_vsignal = vsignal
    first = int(par.mesh.ghost_cells)
    active_count = int(par.mesh.grid_cells)
    active_slice = slice(first, first + active_count)
    if (
        width_runtime_code.ndim == 1
        and vsignal.ndim == 1
        and len(vsignal) >= first + active_count + first
    ):
        active_width_runtime_code = width_runtime_code[active_slice]
        active_density = density[active_slice]
        active_vsignal = vsignal[active_slice]

    # A prescribed boundary state participates in the Riemann problem at the
    # active-domain edge.  Its signal speed must therefore constrain the CFL
    # step even though the rest of the ghost zone is excluded.  This matters
    # for imposed spherical winds, where the first ghost cell can be much
    # faster than every active cell.  Include only the two interface-adjacent
    # ghost cells; farther ghost cells cannot directly affect this update.
    cfl_width_runtime_code = active_width_runtime_code
    cfl_density = active_density
    cfl_vsignal = active_vsignal
    cfl_indices = np.arange(first, first + active_count)
    if (
        width_runtime_code.ndim == 1
        and vsignal.ndim == 1
        and len(width_runtime_code) == len(vsignal)
        and first > 0
        and first + active_count < len(vsignal)
        and getattr(getattr(par, "boundary", None), "condition", None)
        in ("InflowSph", "OutflowSph", "WindSph")
    ):
        interface_indices = np.array([first - 1, first + active_count])
        cfl_width_runtime_code = np.concatenate(
            (
                np.asarray(active_width_runtime_code, dtype=float),
                np.asarray(width_runtime_code[interface_indices], dtype=float),
            ),
        )
        cfl_density = np.concatenate(
            (
                np.asarray(active_density, dtype=float),
                np.asarray(density[interface_indices], dtype=float),
            ),
        )
        cfl_vsignal = np.concatenate(
            (
                np.asarray(active_vsignal, dtype=float),
                np.asarray(vsignal[interface_indices], dtype=float),
            ),
        )
        cfl_indices = np.concatenate((cfl_indices, interface_indices))

    core_mask = getattr(par, "hydrostatic_core_mask", None)
    if core_mask is not None:
        core_active = np.asarray(core_mask[active_slice], dtype=bool)
        active_vsignal = np.asarray(active_vsignal, dtype=float).copy()
        active_vsignal[core_active] = 0.0

    # A vacuum cell has no characteristic speed for the CFL constraint.
    # EOS sound-speed evaluation can produce ``inf`` for rho == 0 because
    # pressure/rho is undefined; exclude such cells from the minimum and
    # keep their interface signal speed neutral for the next flux update.
    density_floor = max(
        0.0,
        float(np.asarray(getattr(par, "cfl_density_floor", 0.0))),
    )
    zero_density = active_density <= density_floor
    if np.any(zero_density):
        active_vsignal = np.asarray(active_vsignal, dtype=float).copy()
        active_vsignal[zero_density] = 0.0
    cfl_density_zero = cfl_density <= density_floor
    if np.any(cfl_density_zero):
        cfl_vsignal = np.asarray(cfl_vsignal, dtype=float).copy()
        cfl_vsignal[cfl_density_zero] = 0.0
    dt_array = solver.safe_divide(CFL * cfl_width_runtime_code, cfl_vsignal)

    # A prescribed spherical wind/inflow can have a density very different
    # from the first active cell.  The wave-speed CFL condition alone then
    # permits a single update to replace many cell masses at once; the
    # positivity limiter would consequently suppress most of the boundary
    # flux.  Bound the step by the mass-loading time of the boundary-adjacent
    # active cell so the imposed flux is evolved conservatively.
    boundary = getattr(par, "boundary", None)
    boundary_condition = getattr(boundary, "condition", None)
    if (
        getattr(getattr(par, "hydrodynamics", None), "boundary_mass_loading_timestep", False)
        and boundary_condition in ("InflowSph", "OutflowSph")
        and hasattr(fluid, "Mass_code")
        and first + 1 < len(fluid.Mass_code)
        and first < len(area_runtime_code)
    ):
        if boundary_condition == "InflowSph":
            boundary_density = getattr(boundary, "rho_inflow_proper", 0.0)
            boundary_velocity = getattr(boundary, "vel_inflow_proper", 0.0)
        else:
            boundary_density = getattr(boundary, "rho_outflow_proper", 0.0)
            boundary_velocity = getattr(boundary, "vel_outflow_proper", 0.0)
        mass_flux = (
            abs(float(np.asarray(boundary_density)))
            * abs(
                float(np.asarray(boundary_velocity)),
            )
            * abs(float(np.asarray(area_runtime_code[first])))
        )
        # The reconstructed boundary/front stencil can deliver the imposed
        # flux into the next active cell as the wind front advances.  Use the
        # lower mass of the two receiving cells so the constraint follows a
        # newly formed low-density cavity instead of assuming that the first
        # cell remains the receiver.
        receiving_mass = np.asarray(fluid.Mass_code[first : first + 2], dtype=float)
        cell_mass = (
            float(np.min(receiving_mass[receiving_mass > 0.0]))
            if np.any(
                receiving_mass > 0.0,
            )
            else 0.0
        )
        if mass_flux > 0.0 and cell_mass > 0.0:
            # Keep the injected mass below the receiving-cell mass.  Using
            # the same safety fraction as the wave-speed CFL preserves the
            # normal solver accuracy while the two-cell receiving stencil
            # accounts for cold, low-density cells entering the front.
            mass_loading_fraction = float(CFL)
            dt_mass_loading = mass_loading_fraction * cell_mass / mass_flux
            dt_array = np.minimum(dt_array, dt_mass_loading)

    dtmax_value = par.timestep.dtmax
    dtmax = float(np.asarray(dtmax_value, dtype=float))
    dt_array = np.where(cfl_vsignal != 0.0, dt_array, dtmax)
    dt = np.amin(dt_array)
    fluid.vsignal_code = np.asarray(vsignal, dtype=float)
    if len(fluid.vsignal_code) == len(active_vsignal):
        fluid.vsignal_code[zero_density] = 0.0
    else:
        fluid.vsignal_code[active_slice] = active_vsignal
    solver.dt = dt
    if np.isnan(np.asarray(dt)):
        log_diagnostic(
            logging.ERROR,
            "hydro_timestep_nan",
            vsignal=vsignal,
            velocity=velocity,
            sound_speed=fluid.cs_code,
        )
        raise Exception("time step is nan")
    dtmin_value = par.timestep.dtmin
    if dt < float(np.asarray(dtmin_value, dtype=float)):
        active_index = int(np.argmin(dt_array))
        min_index = int(cfl_indices[active_index])
        diagnostic_index = min_index
        raise ValueError(
            " time step %.2e smaller than the minimum time step %.2e "
            "at cell %d (rho=%.2e, vel=%.2e, cs=%.2e, dx=%.2e)"
            % (
                dt,
                dtmin_value,
                min_index,
                cfl_density[active_index],
                velocity[diagnostic_index],
                fluid.cs_code[diagnostic_index],
                cfl_width_runtime_code[active_index],
            ),
        )
    dt = min(dt, dtmax)
    if (
        getattr(par, "verbose", 0) >= 1
        # Keep routine CFL reductions quiet; report only a timestep that
        # has fallen at least four decades below the configured maximum.
        and dt <= 1.0e-4 * dtmax
    ):
        min_index = int(np.argmin(dt_array))
        if len(np.asarray(velocity)) == len(active_vsignal):
            diagnostic_index = min_index
        else:
            diagnostic_index = min_index + first
        log_diagnostic(
            logging.WARNING,
            "hydro_timestep_reduced",
            time=time_runtime_code,
            dt=dt,
            cell=diagnostic_index,
            radius=np.asarray(mesh_coordinate)[diagnostic_index],
            density=np.asarray(density_field)[diagnostic_index],
            velocity=np.asarray(velocity)[diagnostic_index],
            sound_speed=np.asarray(fluid.cs_code)[diagnostic_index],
            signal_speed=np.asarray(vsignal)[diagnostic_index],
            width=np.asarray(width_runtime_code)[diagnostic_index],
            pressure=np.asarray(pressure_field)[diagnostic_index],
            dtmin=dtmin_value,
            dtmax=dtmax_value,
        )
        cell_volume = np.asarray(volume_runtime_code)[diagnostic_index]
        cell_rho_code = np.asarray(density_field)[diagnostic_index]
        cell_vel_code = np.asarray(velocity)[diagnostic_index]
        cell_energy_density = np.asarray(fluid.Energy_code)[diagnostic_index] / cell_volume
        cell_kinetic_density = 0.5 * cell_rho_code * cell_vel_code**2
        cell_thermal_density = cell_energy_density - cell_kinetic_density
        cell_specific_thermal = cell_thermal_density / cell_rho_code if cell_rho_code > 0.0 else 0.0
        log_diagnostic(
            logging.WARNING,
            "hydro_timestep_energy_state",
            cell=diagnostic_index,
            energy_density=cell_energy_density,
            kinetic_density=cell_kinetic_density,
            thermal_density=cell_thermal_density,
            specific_thermal=cell_specific_thermal,
            cfl_density_floor=density_floor,
            masked_cells=int(np.count_nonzero(zero_density)),
        )
        neighbor_start = max(first, diagnostic_index - 2)
        neighbor_stop = min(
            first + int(par.mesh.grid_cells),
            diagnostic_index + 3,
        )
        neighbors = []
        for neighbor in range(neighbor_start, neighbor_stop):
            neighbors.append(
                {
                    "cell": neighbor,
                    "radius": np.asarray(mesh_coordinate)[neighbor],
                    "density": np.asarray(density_field)[neighbor],
                    "velocity": np.asarray(velocity)[neighbor],
                    "sound_speed": np.asarray(fluid.cs_code)[neighbor],
                    "pressure": np.asarray(pressure_field)[neighbor],
                },
            )
        log_diagnostic(
            logging.WARNING,
            "hydro_timestep_neighbors",
            cell=diagnostic_index,
            neighbors=neighbors,
        )
    return dt
