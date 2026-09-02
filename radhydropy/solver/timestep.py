"""CFL timestep calculation for the finite-volume solver."""

import numpy as np


def get_time_step(solver, mesh, fluid, par, CFL=None):
    """Return a CFL-limited timestep in the active time coordinate."""
    if CFL is None:
        CFL = par.hydrodynamics.CFL
    fluid.SetSoundSpeed()
    vsignal = np.absolute(fluid.vel_code) + fluid.cs_code
    xdelta = mesh.xdelta
    density = np.asarray(fluid.rho_code, dtype=float)
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

    # A prescribed boundary state participates in the Riemann problem at the
    # active-domain edge.  Its signal speed must therefore constrain the CFL
    # step even though the rest of the ghost zone is excluded.  This matters
    # for imposed spherical winds, where the first ghost cell can be much
    # faster than every active cell.  Include only the two interface-adjacent
    # ghost cells; farther ghost cells cannot directly affect this update.
    cfl_xdelta = active_xdelta
    cfl_density = active_density
    cfl_vsignal = active_vsignal
    cfl_indices = np.arange(first, first + active_count)
    if (
        xdelta.ndim == 1
        and vsignal.ndim == 1
        and len(xdelta) == len(vsignal)
        and first > 0
        and first + active_count < len(vsignal)
        and getattr(getattr(par, 'boundary', None), 'condition', None)
        in ('InflowSph', 'OutflowSph', 'WindSph')
    ):
        interface_indices = np.array([first - 1, first + active_count])
        cfl_xdelta = np.concatenate((
            np.asarray(active_xdelta, dtype=float),
            np.asarray(xdelta[interface_indices], dtype=float),
        ))
        cfl_density = np.concatenate((
            np.asarray(active_density, dtype=float),
            np.asarray(density[interface_indices], dtype=float),
        ))
        cfl_vsignal = np.concatenate((
            np.asarray(active_vsignal, dtype=float),
            np.asarray(vsignal[interface_indices], dtype=float),
        ))
        cfl_indices = np.concatenate((cfl_indices, interface_indices))

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
    cfl_density_zero = cfl_density <= density_floor
    if np.any(cfl_density_zero):
        cfl_vsignal = np.asarray(cfl_vsignal, dtype=float).copy()
        cfl_vsignal[cfl_density_zero] = 0.0
    dt_array = solver._safe_divide(CFL * cfl_xdelta, cfl_vsignal)

    # A prescribed spherical wind/inflow can have a density very different
    # from the first active cell.  The wave-speed CFL condition alone then
    # permits a single update to replace many cell masses at once; the
    # positivity limiter would consequently suppress most of the boundary
    # flux.  Bound the step by the mass-loading time of the boundary-adjacent
    # active cell so the imposed flux is evolved conservatively.
    boundary = getattr(par, 'boundary', None)
    boundary_condition = getattr(boundary, 'condition', None)
    if (
        getattr(getattr(par, 'hydrodynamics', None),
                'boundary_mass_loading_timestep', False)
        and
        boundary_condition in ('InflowSph', 'OutflowSph')
        and hasattr(fluid, 'Mass_code')
        and first + 1 < len(fluid.Mass_code)
        and first < len(mesh.area)
    ):
        if boundary_condition == 'InflowSph':
            boundary_density = getattr(boundary, 'inflow_density', 0.0)
            boundary_velocity = getattr(boundary, 'inflow_velocity', 0.0)
        else:
            boundary_density = getattr(boundary, 'outflow_density', 0.0)
            boundary_velocity = getattr(boundary, 'outflow_velocity', 0.0)
        mass_flux = abs(float(np.asarray(boundary_density))) * abs(
            float(np.asarray(boundary_velocity))
        ) * abs(float(np.asarray(mesh.area[first])))
        # The reconstructed boundary/front stencil can deliver the imposed
        # flux into the next active cell as the wind front advances.  Use the
        # lower mass of the two receiving cells so the constraint follows a
        # newly formed low-density cavity instead of assuming that the first
        # cell remains the receiver.
        receiving_mass = np.asarray(fluid.Mass_code[first:first + 2], dtype=float)
        cell_mass = float(np.min(receiving_mass[receiving_mass > 0.0])) if np.any(
            receiving_mass > 0.0
        ) else 0.0
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
        print('vsignal', vsignal)
        print('fluid.vel_code', fluid.vel_code)
        print('fluid.cs_code', fluid.cs_code)
        raise Exception(" time step is nan")
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
                fluid.vel_code[diagnostic_index],
                fluid.cs_code[diagnostic_index],
                cfl_xdelta[active_index],
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
        if len(np.asarray(fluid.vel_code)) == len(active_vsignal):
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
                np.asarray(fluid.rho_code)[diagnostic_index],
                np.asarray(fluid.vel_code)[diagnostic_index],
                np.asarray(fluid.cs_code)[diagnostic_index],
                np.asarray(vsignal)[diagnostic_index],
                np.asarray(mesh.xdelta)[diagnostic_index],
                np.asarray(fluid.pre_code)[diagnostic_index],
                dtmin_value,
                dtmax_value,
            )
        )
        cell_volume = np.asarray(mesh.vol)[diagnostic_index]
        cell_rho_code = np.asarray(fluid.rho_code)[diagnostic_index]
        cell_vel_code = np.asarray(fluid.vel_code)[diagnostic_index]
        cell_energy_density = (
            np.asarray(fluid.Energy_code)[diagnostic_index] / cell_volume
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
                    np.asarray(fluid.rho_code)[neighbor],
                    np.asarray(fluid.vel_code)[neighbor],
                    np.asarray(fluid.cs_code)[neighbor],
                    np.asarray(fluid.pre_code)[neighbor],
                )
            )
    return dt
