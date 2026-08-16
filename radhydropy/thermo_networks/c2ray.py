"""Causal C2-Ray source integration for hydrogen and hydrogen/helium.

This module intentionally keeps the original C2-Ray ordering: cells are
processed from the source outwards, and the time-averaged opacity of a cell is
converged before its outgoing photon rate is passed to the next cell.

The hydrogen path retains its analytic local update.  The optional H/He path
uses the same causal transport ordering with a coupled implicit local solve.
The ordinary instantaneous thermo-chemistry path remains the default.
"""

from dataclasses import dataclass
import warnings
from types import SimpleNamespace

import numpy as np
import unyt

from radhydropy.constants import (
    DEFAULT_EPSILON_GAMMA,
    DEFAULT_SIGMA_GAMMA,
    PROTON_MASS_CGS,
    SPEED_OF_LIGHT_CGS,
)
from radhydropy.units import (
    CGS_AREA_UNIT,
    PHOTON_FLUX_UNIT,
    PHOTON_RATE_UNIT,
    _code_units,
    code_quantity_to_cgs,
    time_seconds,
)
from radhydropy import radiative_transfer as rrt
from radhydropy.thermo_networks import hydrogen
from radhydropy.thermo_networks import hydrogen_helium


@dataclass
class C2RayResult:
    """Result of one causal C2-Ray source step."""

    photon_density: np.ndarray
    absorbed_photon_rate: np.ndarray
    outgoing_photon_rate: np.ndarray
    mean_neutral_fraction: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray


def _cgs_parameter(value, code, unit, scale_key):
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    if code is not None:
        return np.asarray(code_quantity_to_cgs(value, code, scale_key), dtype=float)
    return np.asarray(value, dtype=float)


def _group_parameters(par):
    edges = getattr(par, "radiation_group_edges_eV", None)
    if edges is None:
        ngroup = 1
        sigma_value = getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA)
        epsilon_value = getattr(par, "hydrogen_epsilon_gamma", DEFAULT_EPSILON_GAMMA)
    else:
        edges = np.asarray(edges, dtype=float)
        ngroup = edges.size - 1
        sigma_value = getattr(par, "radiation_group_sigma_gamma", None)
        if sigma_value is None:
            sigma_value = getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA)
        epsilon_value = getattr(par, "radiation_group_epsilon_gamma", None)
        if epsilon_value is None:
            epsilon_value = getattr(par, "hydrogen_epsilon_gamma", DEFAULT_EPSILON_GAMMA)

    code = _code_units(par)
    sigma = _cgs_parameter(sigma_value, code, CGS_AREA_UNIT, "area_cm2")
    epsilon = _cgs_parameter(
        epsilon_value,
        code,
        "erg",
        "energy_erg",
    )
    sigma = np.full(ngroup, float(sigma), dtype=float) if sigma.ndim == 0 else sigma
    epsilon = (
        np.full(ngroup, float(epsilon), dtype=float)
        if epsilon.ndim == 0
        else epsilon
    )
    if sigma.shape != (ngroup,) or epsilon.shape != (ngroup,):
        raise ValueError("C2-Ray hydrogen group arrays must match the number of groups")

    boundary_flux = getattr(
        par,
        "radiative_transfer_boundary_flux_groups",
        None,
    )
    if boundary_flux is None:
        boundary_flux = getattr(par, "radiative_transfer_boundary_flux", 0.0)
    source_rate = getattr(
        par,
        "radiative_transfer_source_photon_rate_groups",
        None,
    )
    if source_rate is None:
        source_rate = getattr(par, "radiative_transfer_source_photon_rate", 0.0)
    boundary_flux = _cgs_parameter(
        boundary_flux,
        code,
        PHOTON_FLUX_UNIT,
        "photon_flux_per_cm2_s",
    )
    source_rate = _cgs_parameter(
        source_rate,
        code,
        PHOTON_RATE_UNIT,
        "photon_rate_per_s",
    )
    boundary_flux = (
        np.full(ngroup, float(boundary_flux), dtype=float)
        if boundary_flux.ndim == 0
        else boundary_flux
    )
    source_rate = (
        np.full(ngroup, float(source_rate), dtype=float)
        if source_rate.ndim == 0
        else source_rate
    )
    if boundary_flux.shape != (ngroup,) or source_rate.shape != (ngroup,):
        raise ValueError("C2-Ray source arrays must match the number of groups")
    return sigma, epsilon, boundary_flux, source_rate


def _state_geometry(state, par):
    """Build shared RT geometry from a C²-Ray source state."""
    mesh = SimpleNamespace(
        coordsys=getattr(par, "coordsys", "spherical"),
        boundary=np.asarray(state["boundary_cm"], dtype=float),
        vol=np.asarray(state["volume_cm3"], dtype=float),
    )
    if "area_cm2" in state:
        mesh.area = np.asarray(state["area_cm2"], dtype=float)
    return rrt.build_transport_geometry(mesh, mesh.coordsys)


def _source_rates(par, face_area):
    sigma, epsilon, boundary_flux, source_rate = _group_parameters(par)
    direction = 1 if getattr(par, "radiative_transfer_direction", 1) >= 0 else -1
    source_face = 0 if direction > 0 else -1
    if getattr(par, "coordsys", "spherical") == "spherical":
        incoming = np.where(
            source_rate != 0.0,
            source_rate,
            boundary_flux * face_area[source_face],
        )
    else:
        incoming = boundary_flux * face_area[source_face]
    return sigma, epsilon, incoming, direction


def _cell_order(ncell, direction):
    return range(ncell) if direction > 0 else range(ncell - 1, -1, -1)


def _cell_values(value, ncell):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(ncell, float(array), dtype=float)
    return array


def trace_initial_state(state, par):
    """Trace the current state without changing its chemistry."""
    result = _advance(state, par, dt_s=0.0, update_chemistry=False)
    return result


def advance_state(state, par, dt_s):
    """Advance one hydrogen source step using strict causal C2-Ray ordering."""
    return _advance(state, par, float(dt_s), update_chemistry=True)


def _front_radius_kpc(state, neutral_fraction=0.5):
    ionized = np.asarray(state["xHI"]) <= neutral_fraction
    if not np.any(ionized):
        return 0.0
    radius = np.asarray(state["radius_kpc"], dtype=float)
    if np.all(ionized):
        return float(radius[-1])
    left = np.where(ionized)[0][-1]
    right = left + 1
    if state["xHI"][right] == state["xHI"][left]:
        return float(radius[left])
    weight = (neutral_fraction - state["xHI"][left]) / (
        state["xHI"][right] - state["xHI"][left]
    )
    return float(radius[left] + weight * (radius[right] - radius[left]))


def _recombination_rate(state, par):
    alpha = state.get("alpha_B_cm3_s", getattr(par, "hydrogen_alpha_B", None))
    if alpha is None:
        alpha = hydrogen._cgs_alpha_B(state["temperature_K"])
    elif hasattr(alpha, "to_value"):
        alpha = alpha.to_value(unyt.cm**3 / unyt.s)
    ionized = 1.0 - state["xHI"]
    return float(
        np.sum(
            np.asarray(alpha)
            * ionized**2
            * state["nH_cm3"]**2
            * state["volume_cm3"]
        )
    )


def _append_history(history, state, ngamma, time_s, recombined, source_rate_s):
    ionized = 1.0 - state["xHI"]
    ionized_atoms = np.sum(ionized * state["nH_cm3"] * state["volume_cm3"])
    volume_photons = np.sum(ngamma * state["volume_cm3"])
    history["time_Myr"].append(time_s / (1.0 * unyt.Myr).to_value(unyt.s))
    history["front_radius_kpc"].append(_front_radius_kpc(state))
    history["injected_photons"].append(source_rate_s * time_s)
    history["ionized_atoms"].append(ionized_atoms)
    history["recombined_photons"].append(recombined)
    history["volume_photons"].append(volume_photons)
    history["accounted_photons"].append(
        ionized_atoms + recombined + volume_photons
    )
    if "mean_ionized_temp_K" in history:
        weight = 1.0 - state["xHI"]
        history["mean_ionized_temp_K"].append(
            float(np.sum(weight * state["temperature_K"]) / np.sum(weight))
            if np.sum(weight) > 0.0
            else 0.0
        )


def evolve_static_state(
    state,
    par,
    final_time_s,
    dtmax_s,
    source_rate_s=0.0,
    include_thermal_history=False,
    reference_time_s=None,
):
    """Evolve a fixed-density hydrogen state with causal C²-Ray transport."""
    initial = trace_initial_state(state, par)
    ngamma = initial.photon_density[0] if initial.photon_density.shape[0] == 1 else initial.photon_density
    state["ngamma_cm3"] = ngamma
    history = {
        "time_Myr": [],
        "front_radius_kpc": [],
        "injected_photons": [],
        "ionized_atoms": [],
        "recombined_photons": [],
        "volume_photons": [],
        "accounted_photons": [],
    }
    if include_thermal_history:
        history["mean_ionized_temp_K"] = []
    recombined = 0.0
    time_s = 0.0
    _append_history(history, state, ngamma, time_s, recombined, source_rate_s)

    while time_s < final_time_s:
        remaining = final_time_s - time_s
        dt_s = min(float(dtmax_s), remaining)
        if (
            reference_time_s is not None
            and time_s < reference_time_s <= time_s + dt_s
        ):
            dt_s = reference_time_s - time_s
        start_recombination = _recombination_rate(state, par)
        result = advance_state(state, par, dt_s)
        end_recombination = _recombination_rate(state, par)
        recombined += 0.5 * (start_recombination + end_recombination) * dt_s
        time_s += dt_s
        state["time_s"] = time_s
        ngamma = result.photon_density[0] if result.photon_density.shape[0] == 1 else result.photon_density
        state["ngamma_cm3"] = ngamma
        _append_history(history, state, ngamma, time_s, recombined, source_rate_s)
        if reference_time_s is not None and "reference_snapshot" not in history and time_s >= reference_time_s:
            history["reference_snapshot"] = {
                "time_Myr": time_s / (1.0 * unyt.Myr).to_value(unyt.s),
                "radius_kpc": np.asarray(state["radius_kpc"]).copy(),
                "xHI": np.asarray(state["xHI"]).copy(),
                "temperature_K": np.asarray(state["temperature_K"]).copy(),
            }
    history["chemistry_steps"] = len(history["time_Myr"]) - 1
    history["evolution_steps"] = history["chemistry_steps"]
    history["radiative_transfer_updates"] = history["chemistry_steps"] + 1
    return history


def _advance(state, par, dt_s, update_chemistry):
    network = getattr(par, "thermochemistry_network", "hydrogen")
    if network == "hydrogen_helium":
        return _advance_hydrogen_helium(state, par, dt_s, update_chemistry)
    if network != "hydrogen":
        raise NotImplementedError(
            "C2-Ray supports thermochemistry_network='hydrogen' or "
            "'hydrogen_helium'"
        )

    geometry = _state_geometry(state, par)
    sigma, epsilon, incoming, direction = _source_rates(
        par,
        geometry.face_area_cm2,
    )
    width = geometry.width_cm
    volume = geometry.volume_cm3
    ncell = width.size
    ngroup = sigma.size
    nH = np.asarray(state["rho_g_cm3"], dtype=float) * float(
        state.get("hydrogen_mass_fraction", getattr(par, "hydrogen_mass_fraction", 1.0))
    ) / PROTON_MASS_CGS
    x_initial = np.clip(np.asarray(state["xHI"], dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    temperature = np.asarray(state["temperature_K"], dtype=float).copy()
    photon_density = np.zeros((ngroup, ncell), dtype=float)
    absorbed = np.zeros((ngroup, ncell), dtype=float)
    mean_fraction = x_initial.copy()
    final_fraction = x_initial.copy()
    converged = np.ones(ncell, dtype=bool)
    iterations = np.zeros(ncell, dtype=int)
    max_iterations = int(getattr(par, "radiative_transfer_c2ray_max_iterations", 32))
    tolerance = float(getattr(par, "radiative_transfer_c2ray_tolerance", 1.0e-6))
    relaxation = float(getattr(par, "radiative_transfer_c2ray_relaxation", 1.0))
    relaxation = np.clip(relaxation, 0.0, 1.0)
    recombination = bool(state.get("recombination", True))
    collisional = bool(state.get("collisional_ionization", True))
    alpha_parameter = state.get("alpha_B_cm3_s")
    beta_parameter = state.get("beta_cm3_s")
    alpha_values = (
        hydrogen._cgs_alpha_B(temperature)
        if alpha_parameter is None
        else _cell_values(alpha_parameter, ncell)
    )
    beta_values = (
        hydrogen._cgs_beta(temperature)
        if beta_parameter is None
        else _cell_values(beta_parameter, ncell)
    )

    for cell in _cell_order(ncell, direction):
        incoming_cell = incoming
        x0 = x_initial[cell]
        xmean = x0
        cell_converged = not update_chemistry or dt_s == 0.0
        tau = np.zeros(ngroup, dtype=float)
        xfinal = x0
        cell_transport = None

        iteration_range = range(1, max_iterations + 1) if update_chemistry else range(1)
        for iteration in iteration_range:
            tau = np.maximum(sigma * nH[cell] * xmean * width[cell], 0.0)
            cell_transport = rrt.propagate_causal_cell(
                geometry,
                incoming_cell,
                tau,
                cell,
                direction,
            )
            photon_rate = np.sum(cell_transport.absorbed_rate) / max(
                nH[cell] * xmean * volume[cell], 1.0e-99
            )
            if not update_chemistry or dt_s == 0.0:
                xfinal = x0
                xnew_mean = x0
                break
            else:
                alpha = alpha_values[cell]
                beta = beta_values[cell]
                electron_density = nH[cell] * (1.0 - xmean)
                recombination_rate = electron_density * alpha if recombination else 0.0
                collisional_rate = electron_density * beta if collisional else 0.0
                total_rate = photon_rate + recombination_rate + collisional_rate
                if total_rate > 0.0:
                    equilibrium = recombination_rate / total_rate
                    exponent = total_rate * dt_s
                    decay = np.exp(-exponent)
                    xfinal = equilibrium + (x0 - equilibrium) * decay
                    if exponent > 1.0e-12:
                        xnew_mean = equilibrium + (x0 - equilibrium) * (-np.expm1(-exponent)) / exponent
                    else:
                        xnew_mean = x0
                else:
                    xfinal = x0
                    xnew_mean = x0
                xnew_mean = float(np.clip(xnew_mean, 1.0e-12, 1.0 - 1.0e-12))
                if abs(xnew_mean - xmean) <= tolerance:
                    cell_converged = True
                xmean = (1.0 - relaxation) * xmean + relaxation * xnew_mean
                if cell_converged:
                    break

        if update_chemistry and not cell_converged:
            converged[cell] = False
            policy = getattr(par, "radiative_transfer_c2ray_nonconvergence", "warn")
            if policy == "raise":
                raise RuntimeError(f"C2-Ray did not converge in cell {cell}")
            if policy == "warn":
                warnings.warn(
                    f"C2-Ray did not converge in cell {cell} after {max_iterations} iterations",
                    RuntimeWarning,
                    stacklevel=2,
                )
        iterations[cell] = iteration if update_chemistry else 0
        mean_fraction[cell] = xmean
        final_fraction[cell] = np.clip(xfinal, 1.0e-12, 1.0 - 1.0e-12)

        absorbed[:, cell] = cell_transport.absorbed_rate / volume[cell]
        photon_density[:, cell] = cell_transport.photon_density
        incoming = cell_transport.outgoing_rate

    if update_chemistry:
        state["xHI"] = final_fraction
    state["ngamma_cm3"] = photon_density[0] if ngroup == 1 else photon_density
    if update_chemistry and dt_s > 0.0:
        thermal_rate = hydrogen._cgs_source_thermal_rate(
            state["rho_g_cm3"],
            temperature,
            mean_fraction,
            hydrogen_mass_fraction=state.get(
                "hydrogen_mass_fraction",
                getattr(par, "hydrogen_mass_fraction", 1.0),
            ),
            recombination=recombination,
            collisional_ionization=collisional,
            ngamma_cm3=photon_density,
            sigma_gamma_cm2=sigma,
            epsilon_gamma_erg=epsilon,
            compton_cmb_enabled=state.get("compton_cmb_enabled", False),
            compton_cmb_redshift=state.get("compton_cmb_redshift", 0.0),
            cmb_temperature_0_K=state.get("cmb_temperature_0_K", 2.7255),
        )
        if state.get("thermal_coupling", False):
            energy_update = thermal_rate / np.asarray(state["rho_g_cm3"], dtype=float) * dt_s
            if "specific_total_energy_erg_g" in state:
                state["specific_total_energy_erg_g"] = np.maximum(
                    state["specific_total_energy_erg_g"] + energy_update,
                    state.get("specific_kinetic_energy_erg_g", 0.0),
                )
                hydrogen._fast_update_temperature_from_energy(state)
            else:
                state["specific_energy_erg_g"] = np.maximum(
                    state["specific_energy_erg_g"] + energy_update,
                    1.0e6,
                )
                hydrogen.update_temperature_from_energy(state)

    return C2RayResult(
        photon_density=photon_density,
        absorbed_photon_rate=absorbed,
        outgoing_photon_rate=incoming,
        mean_neutral_fraction=mean_fraction,
        converged=converged,
        iterations=iterations,
    )


def _hhe_cell_state(state, cell):
    """Extract a one-cell H/He source state for the local implicit solve."""
    local = dict(state)
    for key in (
        "rho_g_cm3",
        "temperature_K",
        "specific_energy_erg_g",
        "xHI",
        "xHeI",
        "xHeIII",
    ):
        local[key] = np.asarray([np.asarray(state[key], dtype=float)[cell]], dtype=float)
    return local


def _hhe_project(values):
    """Project Newton variables onto the physical H/He state domain."""
    projected = np.asarray(values, dtype=float).copy()
    projected[0] = np.clip(projected[0], 1.0e-12, 1.0 - 1.0e-12)
    projected[1] = np.clip(projected[1], 1.0e-12, 1.0 - 2.0e-12)
    projected[2] = np.clip(
        projected[2], 0.0, max(0.0, 1.0 - projected[1] - 1.0e-12)
    )
    projected[3] = max(float(projected[3]), 1.0e6)
    return projected


def _hhe_set_trial(local, values):
    """Set a bounded H/He Newton trial and refresh its thermodynamic closure."""
    xhi, xhei, xheiii, energy = _hhe_project(values)
    local["xHI"][:] = xhi
    local["xHeI"][:] = xhei
    local["xHeIII"][:] = xheiii
    local["specific_energy_erg_g"][:] = max(float(energy), 1.0e6)
    hydrogen_helium.update_temperature_from_energy(local)


def _hhe_derivative(local, photon_density):
    """Return d(xHI,xHeI,xHeIII,u)/dt for one H/He cell.

    ``hydrogen_helium._rates`` includes the optional metal PIE closure in the
    thermal component.  Because this derivative is evaluated for every
    Newton trial, the PIE table is implicitly coupled to the trial
    temperature and local ``U`` rather than applied as an explicit correction.
    """
    photon_density = np.asarray(photon_density, dtype=float)
    # The shared multigroup rate helpers sum over the group axis only for a
    # two-dimensional (group, cell) field. The causal cell solver receives a
    # one-cell vector, so make that axis explicit; otherwise only group 0
    # reaches the local ODE, suppressing the high-energy He II channel.
    if photon_density.ndim == 1:
        photon_density = photon_density[:, None]
    d_hi, d_hei, d_heiii, thermal = hydrogen_helium._rates(
        local,
        photon_density,
    )
    return np.array(
        [
            float(np.asarray(d_hi).flat[0]),
            float(np.asarray(d_hei).flat[0]),
            float(np.asarray(d_heiii).flat[0]),
            float(np.asarray(thermal).flat[0])
            / max(float(np.asarray(local["rho_g_cm3"]).flat[0]), 1.0e-99),
        ],
        dtype=float,
    )


def _hhe_backward_euler_step(local, photon_density, dt_s, par):
    """Take one damped-Newton backward-Euler step for one H/He cell."""
    old = np.array(
        [
            float(local["xHI"][0]),
            float(local["xHeI"][0]),
            float(local["xHeIII"][0]),
            float(local["specific_energy_erg_g"][0]),
        ],
        dtype=float,
    )
    trial = _hhe_project(old)
    residual_tolerance = float(
        getattr(par, "radiative_transfer_c2ray_ode_tolerance", 1.0e-8)
    )
    max_iterations = int(
        getattr(par, "radiative_transfer_c2ray_ode_max_iterations", 24)
    )
    scales = np.maximum(np.abs(old), np.array([1.0, 1.0, 1.0, 1.0e10]))

    for _ in range(max_iterations):
        trial = _hhe_project(trial)
        _hhe_set_trial(local, trial)
        derivative = _hhe_derivative(local, photon_density)
        residual = trial - old - dt_s * derivative
        if np.max(np.abs(residual) / scales) <= residual_tolerance:
            return trial, True

        jacobian = np.empty((4, 4), dtype=float)
        for column in range(4):
            perturbation = max(abs(trial[column]) * 1.0e-6, 1.0e-8)
            if column == 3:
                perturbation = max(abs(trial[column]) * 1.0e-6, 1.0e3)
            # Neutral fractions and He III can start on a physical boundary.
            # Use a one-sided finite difference there; projecting a positive
            # perturbation back onto the same boundary would otherwise create
            # a zero Jacobian column and make Newton appear singular.
            direction = 1.0
            if column == 0 and trial[column] >= 1.0 - 2.0e-12:
                direction = -1.0
            elif column == 1 and trial[column] >= 1.0 - 2.0e-12:
                direction = -1.0
            elif column == 2 and trial[column] <= 2.0e-12:
                direction = 1.0
            perturbed = trial.copy()
            perturbed[column] += direction * perturbation
            if (
                column == 2
                and trial[column] <= 2.0e-12
                and trial[1] >= 1.0 - 2.0e-12
            ):
                # At initially neutral helium, He III can only appear after
                # He I is converted into He II. Perturb both coordinates so
                # the finite-difference state enters the simplex rather than
                # being projected back to x_HeIII = 0.
                perturbed[1] -= perturbation
            perturbed = _hhe_project(perturbed)
            _hhe_set_trial(local, perturbed)
            derivative_perturbed = _hhe_derivative(local, photon_density)
            residual_perturbed = perturbed - old - dt_s * derivative_perturbed
            jacobian[:, column] = (
                residual_perturbed - residual
            ) / (direction * perturbation)

        try:
            correction = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return trial, False

        accepted = False
        residual_norm = np.max(np.abs(residual) / scales)
        damping = 1.0
        for _ in range(12):
            candidate = _hhe_project(trial + damping * correction)
            _hhe_set_trial(local, candidate)
            candidate_residual = (
                candidate - old - dt_s * _hhe_derivative(local, photon_density)
            )
            candidate_norm = np.max(np.abs(candidate_residual) / scales)
            if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                trial = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            return trial, False

    trial = _hhe_project(trial)
    _hhe_set_trial(local, trial)
    return trial, False


def _hhe_backward_euler(local, photon_density, dt_s, par):
    """Advance one H/He cell, substepping only when the implicit solve needs it.

    The causal transport remains cell-by-cell, while this local fallback keeps
    large source steps robust in highly ionized, optically thick cells without
    changing the converged solution at ordinary steps.
    """
    initial = np.array(
        [
            float(local["xHI"][0]),
            float(local["xHeI"][0]),
            float(local["xHeIII"][0]),
            float(local["specific_energy_erg_g"][0]),
        ],
        dtype=float,
    )
    for subdivisions in (1, 2, 4, 8, 16, 32, 64):
        _hhe_set_trial(local, initial)
        step_dt = dt_s / subdivisions
        success = True
        values = initial.copy()
        for _ in range(subdivisions):
            values, success = _hhe_backward_euler_step(
                local, photon_density, step_dt, par
            )
            if not success:
                break
        if success:
            return values, True
    return values, False


def _hhe_group_parameters(state, par):
    sigma = {
        species: np.asarray(values, dtype=float)
        for species, values in state["sigma_gamma_cm2"].items()
    }
    epsilon = {
        species: np.asarray(values, dtype=float)
        for species, values in state["epsilon_gamma_erg"].items()
    }
    _, _, boundary_flux, source_rate = _group_parameters(par)
    return sigma, epsilon, boundary_flux, source_rate


def _advance_hydrogen_helium(state, par, dt_s, update_chemistry):
    """Advance coupled H/He chemistry with causal multigroup C²-Ray transport."""
    geometry = _state_geometry(state, par)
    sigma_species, epsilon_species, boundary_flux, source_rate = _hhe_group_parameters(
        state, par
    )
    sigma_h = sigma_species["HI"]
    ngroup = sigma_h.size
    direction = 1 if getattr(par, "radiative_transfer_direction", 1) >= 0 else -1
    source_face = 0 if direction > 0 else -1
    incoming = np.where(
        source_rate != 0.0,
        source_rate,
        boundary_flux * geometry.face_area_cm2[source_face],
    )
    width = geometry.width_cm
    volume = geometry.volume_cm3
    ncell = width.size
    n_h = np.asarray(state["rho_g_cm3"], dtype=float) * float(
        state.get("hydrogen_mass_fraction", getattr(par, "hydrogen_mass_fraction", 0.7))
    ) / PROTON_MASS_CGS
    n_he = np.asarray(state["rho_g_cm3"], dtype=float) * float(
        state.get("helium_mass_fraction", getattr(par, "helium_mass_fraction", 0.28))
    ) / (4.0 * PROTON_MASS_CGS)
    xhi_initial = np.clip(np.asarray(state["xHI"], dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    xhei_initial = np.clip(np.asarray(state["xHeI"], dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    xheiii_initial = np.clip(np.asarray(state["xHeIII"], dtype=float), 0.0, 1.0)
    photon_density = np.zeros((ngroup, ncell), dtype=float)
    absorbed = np.zeros((ngroup, ncell), dtype=float)
    mean_fraction = xhi_initial.copy()
    final_fraction = xhi_initial.copy()
    converged = np.ones(ncell, dtype=bool)
    iterations = np.zeros(ncell, dtype=int)
    max_iterations = int(getattr(par, "radiative_transfer_c2ray_max_iterations", 32))
    tolerance = float(getattr(par, "radiative_transfer_c2ray_tolerance", 1.0e-6))
    relaxation = np.clip(
        float(getattr(par, "radiative_transfer_c2ray_relaxation", 1.0)), 0.0, 1.0
    )

    for cell in _cell_order(ncell, direction):
        incoming_cell = incoming
        xhi_mean = xhi_initial[cell]
        xhei_mean = xhei_initial[cell]
        xheiii_mean = xheiii_initial[cell]
        local = _hhe_cell_state(state, cell)
        cell_transport = None
        cell_converged = not update_chemistry or dt_s == 0.0
        iteration_range = range(1, max_iterations + 1) if update_chemistry else range(1)
        for iteration in iteration_range:
            xheii_mean = np.clip(1.0 - xhei_mean - xheiii_mean, 0.0, 1.0)
            tau_species = {
                "HI": sigma_species["HI"] * n_h[cell] * xhi_mean * width[cell],
                "HeI": sigma_species["HeI"] * n_he[cell] * xhei_mean * width[cell],
                "HeII": sigma_species["HeII"] * n_he[cell] * xheii_mean * width[cell],
            }
            tau = np.maximum(sum(tau_species.values()), 0.0)
            cell_transport = rrt.propagate_causal_cell(
                geometry, incoming_cell, tau, cell, direction
            )
            if not update_chemistry or dt_s == 0.0:
                xfinal = xhi_initial[cell]
                xhei_final = xhei_initial[cell]
                xheiii_final = xheiii_initial[cell]
                break

            local["xHI"][:] = xhi_initial[cell]
            local["xHeI"][:] = xhei_initial[cell]
            local["xHeIII"][:] = xheiii_initial[cell]
            # Each opacity iteration solves the same physical time interval
            # from the cell's beginning-of-step state.  Do not carry thermal
            # energy from a rejected opacity iterate into the next trial.
            local["specific_energy_erg_g"][:] = state["specific_energy_erg_g"][cell]
            local["temperature_K"][:] = state["temperature_K"][cell]
            values, ode_converged = _hhe_backward_euler(
                local, cell_transport.photon_density, dt_s, par
            )
            xfinal, xhei_final, xheiii_final = values[:3]
            new_xhi_mean = 0.5 * (xhi_initial[cell] + xfinal)
            new_xhei_mean = 0.5 * (xhei_initial[cell] + xhei_final)
            new_xheiii_mean = 0.5 * (xheiii_initial[cell] + xheiii_final)
            change = max(
                abs(new_xhi_mean - xhi_mean),
                abs(new_xhei_mean - xhei_mean),
                abs(new_xheiii_mean - xheiii_mean),
            )
            xhi_mean = (1.0 - relaxation) * xhi_mean + relaxation * new_xhi_mean
            xhei_mean = (1.0 - relaxation) * xhei_mean + relaxation * new_xhei_mean
            xheiii_mean = (1.0 - relaxation) * xheiii_mean + relaxation * new_xheiii_mean
            cell_converged = ode_converged and change <= tolerance
            if cell_converged:
                break

        if update_chemistry and not cell_converged:
            converged[cell] = False
            policy = getattr(par, "radiative_transfer_c2ray_nonconvergence", "warn")
            if policy == "raise":
                raise RuntimeError(f"C2-Ray H/He did not converge in cell {cell}")
            if policy == "warn":
                warnings.warn(
                    f"C2-Ray H/He did not converge in cell {cell} after {max_iterations} iterations",
                    RuntimeWarning,
                    stacklevel=2,
                )

        iterations[cell] = iteration if update_chemistry else 0
        mean_fraction[cell] = xhi_mean
        final_fraction[cell] = np.clip(xfinal, 1.0e-12, 1.0 - 1.0e-12)
        state["xHI"][cell] = final_fraction[cell]
        state["xHeI"][cell] = np.clip(xhei_final, 1.0e-12, 1.0 - 1.0e-12)
        state["xHeIII"][cell] = np.clip(
            xheiii_final, 0.0, 1.0 - state["xHeI"][cell] - 1.0e-12
        )
        if update_chemistry:
            state["specific_energy_erg_g"][cell] = max(float(local["specific_energy_erg_g"][0]), 1.0e6)
            state["temperature_K"][cell] = float(local["temperature_K"][0])
            state["mu"][cell] = float(local["mu"][0])

        absorbed[:, cell] = cell_transport.absorbed_rate / volume[cell]
        photon_density[:, cell] = cell_transport.photon_density
        incoming = cell_transport.outgoing_rate

    hydrogen_helium._closure(state)
    state["ngamma_cm3"] = photon_density[0] if ngroup == 1 else photon_density
    return C2RayResult(
        photon_density=photon_density,
        absorbed_photon_rate=absorbed,
        outgoing_photon_rate=incoming,
        mean_neutral_fraction=mean_fraction,
        converged=converged,
        iterations=iterations,
    )


def apply_fast(dt, mesh, fluid, par):
    """Apply one strict C2-Ray source step to the fast hydrogen state."""
    network = getattr(par, "thermochemistry_network", "hydrogen")
    if network == "hydrogen_helium":
        state = hydrogen_helium.source_state(mesh, fluid, par)
    elif network == "hydrogen":
        state = hydrogen.c2ray_source_state(mesh, fluid, par)
    else:
        raise NotImplementedError(
            "C2-Ray supports thermochemistry_network='hydrogen' or "
            "'hydrogen_helium'"
        )
    code = _code_units(par)
    dt_s = time_seconds(dt, code)
    advance_state(state, par, dt_s)
    _ensure_fluid_photon_shape(fluid, state["ngamma_cm3"])
    if network == "hydrogen_helium":
        hydrogen_helium.apply_state(state, fluid, par)
    else:
        hydrogen.sync_c2ray_state(state, fluid, par)
    return 1


def _ensure_fluid_photon_shape(fluid, photon_density):
    """Resize the runtime photon field when a spectrum changes group count."""
    target = np.shape(photon_density)
    if np.shape(getattr(fluid, "ngamma", None)) == target:
        return
    fluid.ngamma = np.zeros((target[0], len(fluid.rho)), dtype=float) if len(target) == 2 else np.zeros(len(fluid.rho), dtype=float)
