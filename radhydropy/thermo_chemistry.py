"""Dispatcher for pluggable thermo-chemistry networks."""

from radhydropy.thermo_networks import CIECoolingNetwork, HydrogenNetwork


_NETWORKS = {
    HydrogenNetwork.name: HydrogenNetwork,
    CIECoolingNetwork.name: CIECoolingNetwork,
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


def ionization_fraction_rate(state, ngamma, par):
    """Return the selected network's chemistry fraction rate."""
    return get_network(par).ionization_fraction_rate(state, ngamma)


def thermal_rate(state, ngamma, par):
    """Return the selected network's thermal source rate."""
    return get_network(par).thermal_rate(state, ngamma)


def get_timestep(state, ngamma, par, remaining_s, dtmax_s):
    """Return a source substep for the selected network."""
    return get_network(par).get_timestep(
        state,
        ngamma,
        remaining_s,
        dtmax_s,
    )


def update_temperature_from_energy(state):
    """Update temperature from source-state energy for the selected network."""
    return HydrogenNetwork().update_temperature_from_energy(state)


def ionization_fraction_implicit_update(state, ngamma, dt_s, par):
    """Implicitly update chemistry fractions for the selected network."""
    return get_network(par).ionization_fraction_implicit_update(
        state,
        ngamma,
        dt_s,
    )


def apply_state(state, fluid, par):
    """Copy a thermo-chemistry source state back to the fluid."""
    return get_network(par).apply_state(state, fluid, par)


def get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining):
    """Return a fast source substep for the selected network."""
    return get_network(par).get_source_timestep_fast(mesh, fluid, par, remaining)


def apply_thermochemistry_fast(dt, mesh, fluid, par):
    """Apply the selected network's fast thermo-chemistry source update."""
    return get_network(par).apply_fast(dt, mesh, fluid, par)
