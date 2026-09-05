"""Runtime diagnostics for hydro and thermo-chemistry simulations."""

from pathlib import Path

import numpy as np

from radhydropy.cosmological_variables import physical_temperature, supercomoving_scale


def temperature_physical_cgs_K(sim):
    """Return the simulation gas temperature in physical kelvin."""
    if not hasattr(sim.fluid, 'temp_code'):
        return None
    temperature = np.asarray(sim.fluid.temp_code, dtype=float)
    code = getattr(sim.par, 'CodeUnits', None)
    if code is not None:
        temperature = temperature * float(code.temperature_in_cgs)
    if getattr(sim.par, 'supercomoving_coordinates', False):
        scale_factor, _ = supercomoving_scale(sim.par, time=sim.fluid.time_code)
        temperature = physical_temperature(
            temperature, scale_factor, float(sim.fluid.eos.gamma)
        )
    return np.asarray(temperature, dtype=float)


def thermochemistry_active_mask(rho_physical_cgs_g_cm3, par, density_factor=1.0):
    """Return the source-update mask using the hydro CFL density floor.

    ``rho_physical_cgs_g_cm3`` is physical density, while ``cfl_density_floor``
    is expressed in the runtime code-density units.  ``density_factor`` is
    the supercomoving conversion factor (normally ``a**3``).
    """
    density_floor = max(
        0.0, float(np.asarray(getattr(par, 'cfl_density_floor', 0.0)))
    )
    if density_floor <= 0.0:
        return np.asarray(rho_physical_cgs_g_cm3, dtype=float) > 0.0
    code = getattr(par, 'CodeUnits', None)
    if code is None:
        return np.asarray(rho_physical_cgs_g_cm3, dtype=float) > 0.0
    physical_floor = (
        density_floor
        * float(code.unit_conversion['density_cgs_g_cm3'])
        / float(density_factor)
    )
    return np.asarray(rho_physical_cgs_g_cm3, dtype=float) > physical_floor


def check_conserved_energy_admissibility(
    sim, stage, relative_tolerance=1.0e-7,
):
    """Reject resolved cells whose kinetic energy exceeds total energy.

    The dual-energy variable may provide a pressure fallback when ``E-K``
    loses precision, but it cannot make an inadmissible conservative state
    valid. Numerical-vacuum cells are excluded using the configured CFL
    density floor. A small relative tolerance permits accumulated roundoff;
    larger deficits are reported at the update stage that created them.
    """
    par = sim.par
    if getattr(par, 'dual_energy', False):
        # In dual-energy mode, E-K may lose the tiny thermal component in a
        # cold spherical flow.  Pressure is reconstructed from InternalEnergy;
        # audit the conservative state with the same cancellation allowance.
        relative_tolerance = max(relative_tolerance, 1.0e-6)
    # Isothermal EOS runs do not evolve thermal energy as an Euler
    # conservative variable.  Their Energy field may therefore contain only
    # kinetic energy (or zero), while pressure is reconstructed from T and mu;
    # the adiabatic E >= K invariant is not applicable here.
    if getattr(getattr(sim, 'fluid', None), 'eos', None) is not None:
        if getattr(sim.fluid.eos, 'is_isothermal', False):
            return
    if not all(
        hasattr(sim.fluid, name) for name in ('Mass_code', 'Mom_code', 'Energy_code')
    ):
        # Source-only/unit-test states may not have been initialized with
        # hydrodynamic conserved fields.
        return
    first = int(par.mesh.ghost_cells)
    last = first + int(par.mesh.grid_cells)
    volume = np.asarray(sim.mesh.vol, dtype=float)
    mass = np.asarray(sim.fluid.Mass_code, dtype=float)
    momentum = np.asarray(sim.fluid.Mom_code, dtype=float)
    energy = np.asarray(sim.fluid.Energy_code, dtype=float)
    if last <= first:
        return
    physical = np.zeros(len(mass), dtype=bool)
    physical[first:min(last, len(mass))] = True
    density_floor = max(
        0.0, float(np.asarray(getattr(par, 'cfl_density_floor', 0.0)))
    )
    resolved = physical & (
        mass > density_floor * np.maximum(volume, 0.0)
    )
    finite = (
        np.isfinite(mass) & np.isfinite(momentum) & np.isfinite(energy)
    )
    positive_mass = resolved & finite & (mass > 0.0)
    kinetic = np.zeros_like(energy)
    kinetic[positive_mass] = (
        0.5 * momentum[positive_mass]**2 / mass[positive_mass]
    )
    scale = np.maximum(
        np.maximum(kinetic, np.abs(energy)), np.finfo(float).tiny
    )
    deficit = kinetic - energy
    invalid = positive_mass & (
        deficit > float(relative_tolerance) * scale
    )
    if not np.any(invalid):
        return
    index = int(np.flatnonzero(invalid)[0])
    diagnostic = (
        'conserved energy admissibility error after %s at cell %d: '
        'kinetic energy exceeds total energy; mass=%s momentum=%s '
        'energy=%s kinetic=%s deficit=%s relative_deficit=%s '
        'relative_tolerance=%s'
        % (
            stage, index, mass[index], momentum[index], energy[index],
            kinetic[index], deficit[index], deficit[index] / scale[index],
            relative_tolerance,
        )
    )
    print(diagnostic)
    raise ValueError(diagnostic)


def check_temperature_jump(sim, temperature_before, stage, source_result=None):
    """Raise and save a neighborhood dump when a new T exceeds the guard."""
    threshold = getattr(sim.par, 'temperature_jump_error_threshold', None)
    if threshold is None:
        return
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        return
    temperature_after = temperature_physical_cgs_K(sim)
    if temperature_after is None or temperature_before is None:
        return
    before = np.asarray(temperature_before, dtype=float)
    density = np.asarray(sim.fluid.rho_code, dtype=float)
    crossing = (
        (density > 0.0)
        & np.isfinite(temperature_after)
        & (temperature_after > threshold)
    )
    if before.shape == temperature_after.shape:
        crossing &= before <= threshold
    if not np.any(crossing):
        return
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    candidates = np.flatnonzero(crossing)
    candidates = candidates[(candidates >= first) & (candidates < last)]
    if candidates.size == 0:
        return
    index = int(candidates[0])
    radius = np.asarray(sim.mesh.coordinate, dtype=float)
    velocity = np.asarray(sim.fluid.vel_code, dtype=float)
    pressure = np.asarray(sim.fluid.pre_code, dtype=float)
    sound_speed = np.asarray(
        getattr(sim.fluid, 'cs_code', np.zeros_like(density)), dtype=float
    )
    energy = np.asarray(sim.fluid.Energy_code, dtype=float)
    mass = np.asarray(sim.fluid.Mass_code, dtype=float)
    lines = [
        'temperature jump error: physical gas temperature exceeded %.6e K '
        'during %s at cell %d (time=%s)' % (
            threshold, stage, index, sim.fluid.time_code,
        ),
        'cell: radius=%s T_before=%s K T_after=%s K rho=%s vel=%s '
        'pressure=%s cs=%s mass=%s energy=%s' % (
            radius[index], before[index], temperature_after[index],
            density[index], velocity[index], pressure[index],
            sound_speed[index], mass[index], energy[index],
        ),
        'neighborhood: idx radius T_before[K] T_after[K] rho vel pressure cs mass energy',
    ]
    for neighbor in range(max(first, index - 2), min(last, index + 3)):
        lines.append(
            '%d %s %s %s %s %s %s %s %s %s' % (
                neighbor, radius[neighbor],
                before[neighbor] if before.shape == temperature_after.shape else np.nan,
                temperature_after[neighbor], density[neighbor], velocity[neighbor],
                pressure[neighbor], sound_speed[neighbor], mass[neighbor], energy[neighbor],
            )
        )
    if source_result:
        lines.append(
            'source solver: %s relative_change=%s source_steps=%s' % (
                source_result.get('source_solver', 'unknown'),
                source_result.get('relative_change', 'unknown'),
                source_result.get('source_steps', 'unknown'),
            )
        )
    diagnostic = '\n'.join(lines)
    print(diagnostic)
    output_dir = sim.par.output.directory
    if output_dir is not None:
        try:
            filename = Path(output_dir) / 'temperature_jump_error.txt'
            filename.write_text(diagnostic + '\n', encoding='utf-8')
        except (OSError, TypeError, ValueError):
            pass
    raise RuntimeError(diagnostic)


def check_source_temperature(state, par, temperature_before, stage, source_step):
    """Reject a source substep that crosses the configured temperature guard.

    ``state['temperature_cgs_K']`` is already in physical kelvin and contains
    only the active mesh cells, unlike the full fluid state checked by
    :func:`check_temperature_jump`.
    """
    threshold = getattr(par, 'temperature_jump_error_threshold', None)
    if threshold is None:
        return
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        return
    temperature_after = np.asarray(state.get('temperature_cgs_K'), dtype=float)
    if temperature_after.ndim == 0:
        return
    before = np.asarray(temperature_before, dtype=float)
    active = np.asarray(
        state.get('active', np.ones_like(temperature_after, dtype=bool)),
        dtype=bool,
    )
    crossing = active & np.isfinite(temperature_after) & (
        temperature_after > threshold
    )
    if before.shape == temperature_after.shape:
        crossing &= before <= threshold
    if not np.any(crossing):
        return
    index = int(np.flatnonzero(crossing)[0])
    interior = state.get('interior', slice(0, len(temperature_after)))
    mesh_index = int(interior.start or 0) + index
    rho = np.asarray(state.get('rho_cgs_g_cm3', np.nan), dtype=float)
    xhi = np.asarray(state.get('xHI', np.nan), dtype=float)
    energy = np.asarray(
        state.get('specific_energy_cgs_erg_g', np.nan), dtype=float
    )
    diagnostic = (
        'temperature jump error: physical gas temperature exceeded '
        '%.6e K during %s source substep %d at cell %d '
        '(T_before=%s K T_after=%s K rho=%s g/cm^3 xHI=%s '
        'specific_energy=%s erg/g)' % (
            threshold, stage, int(source_step), mesh_index,
            before[index] if before.shape == temperature_after.shape else np.nan,
            temperature_after[index], rho[index], xhi[index], energy[index],
        )
    )
    print(diagnostic)
    output_dir = par.output.directory
    if output_dir is not None:
        try:
            filename = Path(output_dir) / 'temperature_jump_error.txt'
            filename.write_text(diagnostic + '\n', encoding='utf-8')
        except (OSError, TypeError, ValueError):
            pass
    raise RuntimeError(diagnostic)
