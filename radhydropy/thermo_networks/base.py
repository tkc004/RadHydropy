# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Base interface for thermo-chemistry networks."""

from typing import Any


class ThermochemistryNetwork:
    """Interface implemented by concrete thermo-chemistry networks."""

    name = "base"
    scalar_fields = ()

    def enabled(self, fluid: Any, par: Any) -> Any:
        raise NotImplementedError

    def radiation_enabled(self, fluid: Any, par: Any) -> Any:
        raise NotImplementedError

    def radiation_evolution_enabled(self, fluid: Any, par: Any) -> Any:
        raise NotImplementedError

    def advect_ionization_fraction(
        self,
        dt: Any,
        mesh: Any,
        fluid: Any,
        par: Any,
        old_mass: Any,
        mass_flux: Any,
    ) -> Any:
        raise NotImplementedError

    def source_state(self, mesh: Any, fluid: Any, par: Any) -> Any:
        raise NotImplementedError

    def ionization_fraction_rate(self, state: Any, ngamma_cgs_cm3: Any) -> Any:
        raise NotImplementedError

    def thermal_rate(self, state: Any, ngamma_cgs_cm3: Any) -> Any:
        raise NotImplementedError

    def get_timestep(
        self,
        state: Any,
        ngamma_cgs_cm3: Any,
        remaining_s: Any,
        dtmax_s: Any,
    ) -> Any:
        raise NotImplementedError

    def update_temperature_from_energy(self, state: Any) -> Any:
        raise NotImplementedError

    def ionization_fraction_implicit_update(
        self,
        state: Any,
        ngamma_cgs_cm3: Any,
        dt_s: Any,
    ) -> Any:
        raise NotImplementedError

    def apply_state(self, state: Any, fluid: Any, par: Any) -> Any:
        raise NotImplementedError

    def get_source_timestep_fast(
        self,
        mesh: Any,
        fluid: Any,
        par: Any,
        remaining: Any,
    ) -> Any:
        raise NotImplementedError

    def apply_fast(self, dt: Any, mesh: Any, fluid: Any, par: Any) -> Any:
        raise NotImplementedError
