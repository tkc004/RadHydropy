"""Dispatcher for pluggable thermo-chemistry networks."""

from radhydropy.thermo_networks import HydrogenNetwork


_NETWORKS = {
    HydrogenNetwork.name: HydrogenNetwork,
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
    if not getattr(par, "hydrogen_chemistry", False):
        return False
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


def trace_spherical_photon_density_fast(mesh, fluid, par):
    """Update ``ngamma`` with the selected network's fast RT-coupled trace."""
    return get_network(par).trace_spherical_photon_density_fast(mesh, fluid, par)


def static_thermochemistry_state(mesh, fluid, par):
    """Return a static thermo-chemistry state for the selected network."""
    return get_network(par).static_state(mesh, fluid, par)


def trace_static_spherical_photon_density(state, par):
    """Trace a central source through a static thermo-chemistry state."""
    return get_network(par).trace_static_spherical_photon_density(state, par)


def static_ionization_fraction_rate(state, ngamma, par):
    """Return the selected network's static chemistry fraction rate."""
    return get_network(par).static_ionization_fraction_rate(state, ngamma, par)


def static_thermal_rate(state, ngamma, par):
    """Return the selected network's static thermal source rate."""
    return get_network(par).static_thermal_rate(state, ngamma, par)


def get_static_thermochemistry_timestep(state, ngamma, par, remaining_s, dtmax_s):
    """Return a static source substep for the selected network."""
    return get_network(par).get_static_timestep(
        state,
        ngamma,
        par,
        remaining_s,
        dtmax_s,
    )


def update_static_temperature_from_energy(state, par=None):
    """Update temperature from static-state energy for the selected network."""
    if par is None:
        network = HydrogenNetwork()
    else:
        network = get_network(par)
    return network.update_static_temperature_from_energy(state)


def static_ionization_fraction_implicit_update(state, ngamma, dt_s, par):
    """Implicitly update static chemistry fractions for the selected network."""
    return get_network(par).static_ionization_fraction_implicit_update(
        state,
        ngamma,
        dt_s,
        par,
    )


def apply_static_thermochemistry_state(state, fluid, par):
    """Copy a static thermo-chemistry state back to the fluid."""
    return get_network(par).apply_static_state(state, fluid, par)


def get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining):
    """Return a fast source substep for the selected network."""
    return get_network(par).get_source_timestep_fast(mesh, fluid, par, remaining)


def apply_thermochemistry_fast(dt, mesh, fluid, par):
    """Apply the selected network's fast thermo-chemistry source update."""
    return get_network(par).apply_fast(dt, mesh, fluid, par)


def get_thermochemistry_timestep(mesh, fluid, par):
    """Return a thermo-chemistry source timestep for the selected network."""
    return get_network(par).get_timestep(mesh, fluid, par)


def apply_thermochemistry(dt, mesh, fluid, par):
    """Apply the selected thermo-chemistry source update."""
    return get_network(par).apply(dt, mesh, fluid, par)
