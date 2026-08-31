"""CFL timestep calculation for the finite-volume solver."""

import numpy as np


def get_time_step(solver, mesh, fluid, par, CFL=None):
    """Return a CFL-limited timestep in the active time coordinate."""
    if CFL is None:
        CFL = par.hydrodynamics.CFL
    fluid.SetSoundSpeed()
    vsignal = np.absolute(fluid.vel) + fluid.cs
    xdelta = mesh.xdelta
    density = np.asarray(fluid.rho, dtype=float)
    if xdelta.shape != vsignal.shape:
        interior = solver._interior_slice(par)
        if xdelta[interior].shape == vsignal.shape:
            xdelta = xdelta[interior]
            density = density[interior]
        elif vsignal[interior].shape == xdelta.shape:
            vsignal = vsignal[interior]
            density = density[interior]

    # Ghost zones are needed by the Riemann solve but must not determine
    # the CFL step.  In particular, reflecting/outflow boundary updates
    # can leave a ghost velocity temporarily very large while the active
    # solution remains valid.  Keep the full signal-speed array for later
    # flux work, and reduce only over the active cells here.
    active_xdelta = xdelta
    active_density = density
    active_vsignal = vsignal
    first = int(par.mesh.ghost_cells)
    active_count = int(par.mesh.grid_cells)
    active_slice = slice(first, first + active_count)
    if (
        xdelta.ndim == 1
        and vsignal.ndim == 1
        and len(vsignal) >= first + active_count + first
    ):
        active_xdelta = xdelta[active_slice]
        active_density = density[active_slice]
        active_vsignal = vsignal[active_slice]

    core_mask = getattr(par, '_hydrostatic_core_mask', None)
    if core_mask is not None:
        core_active = np.asarray(core_mask[active_slice], dtype=bool)
        active_vsignal = np.asarray(active_vsignal, dtype=float).copy()
        active_vsignal[core_active] = 0.0

    # A vacuum cell has no characteristic speed for the CFL constraint.
    # EOS sound-speed evaluation can produce ``inf`` for rho == 0 because
    # pressure/rho is undefined; exclude such cells from the minimum and
    # keep their interface signal speed neutral for the next flux update.
    density_floor = max(
        0.0, float(np.asarray(getattr(par, 'cfl_density_floor', 0.0)))
    )
    zero_density = active_density <= density_floor
    if np.any(zero_density):
        active_vsignal = np.asarray(active_vsignal, dtype=float).copy()
        active_vsignal[zero_density] = 0.0
    dt_array = solver._safe_divide(CFL * active_xdelta, active_vsignal)
    dtmax_value = par.timestep.dtmax
    dtmax = float(np.asarray(dtmax_value, dtype=float))
    dt_array = np.where(active_vsignal != 0.0, dt_array, dtmax)
    dt = np.amin(dt_array)
    fluid.vsignal = np.asarray(vsignal, dtype=float)
    if len(fluid.vsignal) == len(active_vsignal):
        fluid.vsignal[zero_density] = 0.0
    else:
        fluid.vsignal[active_slice] = active_vsignal
    solver.dt = dt
    if np.isnan(np.asarray(dt)):
        print('vsignal', vsignal)
        print('fluid.vel', fluid.vel)
        print('fluid.cs', fluid.cs)
        raise Exception(" time step is nan")
    dtmin_value = par.timestep.dtmin
    if dt < float(np.asarray(dtmin_value, dtype=float)):
        active_index = int(np.argmin(dt_array))
        min_index = active_index + first
        if len(np.asarray(fluid.vel)) == len(active_vsignal):
            diagnostic_index = active_index
        else:
            diagnostic_index = min_index
        raise ValueError(
            " time step %.2e smaller than the minimum time step %.2e "
            "at cell %d (rho=%.2e, vel=%.2e, cs=%.2e, dx=%.2e)"
            % (
                dt,
                dtmin_value,
                min_index,
                active_density[active_index],
                fluid.vel[diagnostic_index],
                fluid.cs[diagnostic_index],
                active_xdelta[active_index],
            )
        )
    if dt > dtmax:
        dt = dtmax
    if (
        getattr(par, 'verbose', 0) >= 1
        # Keep routine CFL reductions quiet; report only a timestep that
        # has fallen at least four decades below the configured maximum.
        and dt <= 1.0e-4 * dtmax
    ):
        min_index = int(np.argmin(dt_array))
        if len(np.asarray(fluid.vel)) == len(active_vsignal):
            diagnostic_index = min_index
        else:
            diagnostic_index = min_index + first
        print(
            '[hydro dt] t=%s dt=%s idx=%d radius=%s rho=%s vel=%s '
            'cs=%s vsignal=%s dx=%s pre=%s dtmin=%s dtmax=%s'
            % (
                fluid.time,
                dt,
                diagnostic_index,
                np.asarray(mesh.coordinate)[diagnostic_index],
                np.asarray(fluid.rho)[diagnostic_index],
                np.asarray(fluid.vel)[diagnostic_index],
                np.asarray(fluid.cs)[diagnostic_index],
                np.asarray(vsignal)[diagnostic_index],
                np.asarray(mesh.xdelta)[diagnostic_index],
                np.asarray(fluid.pre)[diagnostic_index],
                dtmin_value,
                dtmax_value,
            )
        )
        cell_volume = np.asarray(mesh.vol)[diagnostic_index]
        cell_rho = np.asarray(fluid.rho)[diagnostic_index]
        cell_vel = np.asarray(fluid.vel)[diagnostic_index]
        cell_energy_density = (
            np.asarray(fluid.Energy)[diagnostic_index] / cell_volume
        )
        cell_kinetic_density = 0.5 * cell_rho * cell_vel**2
        cell_thermal_density = cell_energy_density - cell_kinetic_density
        cell_specific_thermal = (
            cell_thermal_density / cell_rho
            if cell_rho > 0.0 else 0.0
        )
        print(
            '[hydro dt energy] idx=%d energy_density=%s '
            'kinetic_density=%s thermal_density=%s '
            'specific_thermal=%s'
            % (
                diagnostic_index,
                cell_energy_density,
                cell_kinetic_density,
                cell_thermal_density,
                cell_specific_thermal,
            )
        )
        print(
            '[hydro dt mask] cfl_density_floor=%s masked=%s' % (
                density_floor,
                int(np.count_nonzero(zero_density)),
            )
        )
        neighbor_start = max(first, diagnostic_index - 2)
        neighbor_stop = min(
        first + int(par.mesh.grid_cells),
            diagnostic_index + 3,
        )
        print('[hydro dt neighbors] idx radius rho vel cs pre')
        for neighbor in range(neighbor_start, neighbor_stop):
            print(
                '[hydro dt neighbors] %d %s %s %s %s %s'
                % (
                    neighbor,
                    np.asarray(mesh.coordinate)[neighbor],
                    np.asarray(fluid.rho)[neighbor],
                    np.asarray(fluid.vel)[neighbor],
                    np.asarray(fluid.cs)[neighbor],
                    np.asarray(fluid.pre)[neighbor],
                )
            )
    return dt
