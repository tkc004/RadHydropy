"""Base interface for thermo-chemistry networks."""


class ThermochemistryNetwork:
    """Interface implemented by concrete thermo-chemistry networks."""

    name = "base"
    scalar_fields = ()

    def enabled(self, fluid, par):
        raise NotImplementedError

    def radiation_enabled(self, fluid, par):
        raise NotImplementedError

    def radiation_evolution_enabled(self, fluid, par):
        raise NotImplementedError

    def advect_ionization_fraction(self, dt, mesh, fluid, par, old_mass, mass_flux):
        raise NotImplementedError

    def trace_spherical_photon_density_fast(self, mesh, fluid, par):
        raise NotImplementedError

    def source_state(self, mesh, fluid, par):
        raise NotImplementedError

    def trace_spherical_photon_density(self, state):
        raise NotImplementedError

    def ionization_fraction_rate(self, state, ngamma):
        raise NotImplementedError

    def thermal_rate(self, state, ngamma):
        raise NotImplementedError

    def get_timestep(self, state, ngamma, remaining_s, dtmax_s):
        raise NotImplementedError

    def update_temperature_from_energy(self, state):
        raise NotImplementedError

    def ionization_fraction_implicit_update(self, state, ngamma, dt_s):
        raise NotImplementedError

    def apply_state(self, state, fluid, par):
        raise NotImplementedError

    def get_source_timestep_fast(self, mesh, fluid, par, remaining):
        raise NotImplementedError

    def apply_fast(self, dt, mesh, fluid, par):
        raise NotImplementedError
