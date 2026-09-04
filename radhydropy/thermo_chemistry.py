"""Dispatcher for pluggable thermo-chemistry networks."""

from radhydropy.thermo_networks import (
    CIECoolingNetwork,
    HydrogenNetwork,
    HydrogenHeliumNetwork,
    PIEUVBGCoolingNetwork,
)


_NETWORKS = {
    HydrogenNetwork.name: HydrogenNetwork,
    HydrogenHeliumNetwork.name: HydrogenHeliumNetwork,
    CIECoolingNetwork.name: CIECoolingNetwork,
    PIEUVBGCoolingNetwork.name: PIEUVBGCoolingNetwork,
}


def available_networks():
    """Return the names of available thermo-chemistry networks."""
    return tuple(sorted(_NETWORKS))


def get_network(par):
    """Return the thermo-chemistry network selected by the run parameters."""
    network_name = getattr(par, "thermochemistry_network", "hydrogen")
    try:
        network_class = _NETWORKS[network_name]
    except KeyError as exc:
        available = ", ".join(available_networks())
        raise ValueError(
            "Unknown thermo-chemistry network "
            f"{network_name!r}; available networks: {available}"
        ) from exc
    # The dispatcher returns a fresh network instance so each call can read the
    # current runtime parameters without sharing mutable state.
    return network_class()


def thermochemistry_enabled(fluid, par):
    return get_network(par).enabled(fluid, par)


def thermochemistry_radiation_enabled(fluid, par):
    return get_network(par).radiation_enabled(fluid, par)


def thermochemistry_radiation_evolution_enabled(fluid, par):
    return get_network(par).radiation_evolution_enabled(fluid, par)


def advect_ionization_fraction(dt, mesh, fluid, par, old_mass, mass_flux):
    """Advect chemistry scalars consistently with the mass flux."""
    return get_network(par).advect_ionization_fraction(
        dt,
        mesh,
        fluid,
        par,
        old_mass,
        mass_flux,
    )


def source_state(mesh, fluid, par):
    """Return a thermo-chemistry source state for the selected network."""
    return get_network(par).source_state(mesh, fluid, par)


def ionization_fraction_rate(state, ngamma_cgs_cm3, par):
    """Return the selected network's chemistry fraction rate."""
    return get_network(par).ionization_fraction_rate(state, ngamma_cgs_cm3)


def thermal_rate(state, ngamma_cgs_cm3, par):
    """Return the selected network's thermal source rate."""
    return get_network(par).thermal_rate(state, ngamma_cgs_cm3)


def get_timestep(state, ngamma_cgs_cm3, par, remaining_s, dtmax_s):
    """Return a source substep for the selected network."""
    return get_network(par).get_timestep(
        state,
        ngamma_cgs_cm3,
        remaining_s,
        dtmax_s,
    )


def update_temperature_from_energy(state):
    """Update temperature from source-state energy for the selected network."""
    if 'helium_mass_fraction' in state:
        return HydrogenHeliumNetwork().update_temperature_from_energy(state)
    return HydrogenNetwork().update_temperature_from_energy(state)


def ionization_fraction_implicit_update(state, ngamma_cgs_cm3, dt_s, par):
    """Implicitly update chemistry fractions for the selected network."""
    return get_network(par).ionization_fraction_implicit_update(
        state,
        ngamma_cgs_cm3,
        dt_s,
    )


def coupled_implicit_update(state, ngamma_cgs_cm3, dt_s, par):
    network = get_network(par)
    if hasattr(network, 'coupled_implicit_update'):
        return network.coupled_implicit_update(state, ngamma_cgs_cm3, dt_s)
    network.ionization_fraction_implicit_update(state, ngamma_cgs_cm3, dt_s)


def apply_state(state, fluid, par):
    """Copy a thermo-chemistry source state back to the fluid."""
    return get_network(par).apply_state(state, fluid, par)


def get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining):
    """Return a fast source substep for the selected network."""
    return get_network(par).get_source_timestep_fast(mesh, fluid, par, remaining)


def apply_thermochemistry_fast(dt, mesh, fluid, par, transport_result=None):
    """Apply the selected network's fast thermo-chemistry source update."""
    if (
        getattr(par, "radiative_transfer", False)
        and getattr(par, "radiative_transfer_temporal_scheme", "instantaneous")
        == "c2ray"
    ):
        from radhydropy.thermo_networks import c2ray

        return c2ray.apply_fast(dt, mesh, fluid, par)
    network = get_network(par)
    if transport_result is not None and getattr(network, "name", None) == "hydrogen":
        result = network.apply_fast(
            dt,
            mesh,
            fluid,
            par,
            transport_result=transport_result,
        )
    else:
        result = network.apply_fast(dt, mesh, fluid, par)
    if isinstance(result, dict):
        return result
    return {
        "source_steps": int(result or 0),
        "absorbed_photon_rate": None,
        "photon_energy_cgs_erg": None,
        "direction": int(getattr(par, "radiative_transfer_direction", 1)),
    }


def evolve_static_source_state(
    state,
    par,
    final_time_s,
    dtmax_s,
    source_rate_s=0.0,
    include_thermal_history=False,
    reference_time_s=None,
):
    """Evolve a fixed-density source state with the selected RT scheme."""
    scheme = getattr(par, "radiative_transfer_temporal_scheme", "instantaneous")
    if scheme == "c2ray":
        from radhydropy.thermo_networks import c2ray

        return c2ray.evolve_static_state(
            state,
            par,
            final_time_s,
            dtmax_s,
            source_rate_s=source_rate_s,
            include_thermal_history=include_thermal_history,
            reference_time_s=reference_time_s,
        )
    raise ValueError(f"unsupported static radiation scheme: {scheme!r}")
