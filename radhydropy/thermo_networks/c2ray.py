"""Causal C2-Ray source integration for hydrogen.

This module intentionally keeps the original C2-Ray ordering: cells are
processed from the source outwards, and the time-averaged opacity of a cell is
converged before its outgoing photon rate is passed to the next cell.

The implementation is hydrogen-only for now.  The ordinary instantaneous
thermo-chemistry path remains the default and is not changed by this module.
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
)
from radhydropy import radiative_transfer as rrt
from radhydropy.thermo_networks import hydrogen


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


def _cell_value(value, cell):
    array = np.asarray(value, dtype=float)
    return float(array if array.ndim == 0 else array[cell])


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
    if getattr(par, "thermochemistry_network", "hydrogen") != "hydrogen":
        raise NotImplementedError(
            "C2-Ray currently supports thermochemistry_network='hydrogen' only"
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

    for cell in _cell_order(ncell, direction):
        incoming_cell = incoming.copy()
        x0 = x_initial[cell]
        xmean = x0
        cell_converged = not update_chemistry or dt_s == 0.0
        tau = np.zeros(ngroup, dtype=float)
        xfinal = x0

        iteration_range = range(1, max_iterations + 1) if update_chemistry else range(1)
        for iteration in iteration_range:
            tau = np.maximum(sigma * nH[cell] * xmean * width[cell], 0.0)
            attenuation = np.exp(-np.clip(tau, 0.0, 700.0))
            qabs = incoming_cell * (-np.expm1(-tau))
            cell_transport = rrt.propagate_causal_cell(
                geometry,
                incoming_cell,
                tau,
                cell,
                direction,
            )
            ngamma_cell = cell_transport.photon_density
            photon_rate = np.sum(qabs) / max(nH[cell] * xmean * volume[cell], 1.0e-99)
            if not update_chemistry or dt_s == 0.0:
                xfinal = x0
                xnew_mean = x0
                break
            else:
                alpha_parameter = state.get("alpha_B_cm3_s")
                beta_parameter = state.get("beta_cm3_s")
                if alpha_parameter is None:
                    alpha = hydrogen._cgs_alpha_B(
                        np.asarray([temperature[cell]])
                    )[0]
                else:
                    alpha = _cell_value(alpha_parameter, cell)
                if beta_parameter is None:
                    beta = hydrogen._cgs_beta(
                        np.asarray([temperature[cell]])
                    )[0]
                else:
                    beta = _cell_value(beta_parameter, cell)
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

        cell_transport = rrt.propagate_causal_cell(
            geometry,
            incoming_cell,
            tau,
            cell,
            direction,
        )
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


def apply_fast(dt, mesh, fluid, par):
    """Apply one strict C2-Ray source step to the fast hydrogen state."""
    if getattr(par, "thermochemistry_network", "hydrogen") != "hydrogen":
        raise NotImplementedError(
            "C2-Ray currently supports thermochemistry_network='hydrogen' only"
        )
    state = hydrogen.c2ray_source_state(mesh, fluid, par)
    code = _code_units(par)
    dt_s = float(dt.to_value(code.time_unit)) if hasattr(dt, "to_value") else float(dt)
    advance_state(state, par, dt_s)
    _ensure_fluid_photon_shape(fluid, state["ngamma_cm3"])
    hydrogen.sync_c2ray_state(state, fluid, par)
    return 1


def _ensure_fluid_photon_shape(fluid, photon_density):
    """Resize the runtime photon field when a spectrum changes group count."""
    target = np.shape(photon_density)
    if np.shape(getattr(fluid, "ngamma", None)) == target:
        return
    fluid.ngamma = np.zeros((target[0], len(fluid.rho)), dtype=float) if len(target) == 2 else np.zeros(len(fluid.rho), dtype=float)
