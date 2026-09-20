# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Cosmological virial-shock runtime physics hooks."""  # noqa: CPY001

import numpy as np


class CosmologicalVirialShockPhysics:
    """Prepare time-dependent cosmological state before each hydro step."""

    def __init__(
        self,
        sim,
        cosmology,
        initial_condition,
        baryon_fraction,
        initial_a,
        cmb_temperature_0_cgs_K,
        reservoir_temperature_supercomoving_code,
        minimum_temperature=None,
        transition_redshift=None,
    ):
        self.sim = sim
        self.cosmology = cosmology
        self.initial_condition = initial_condition
        self.baryon_fraction = float(baryon_fraction)
        self.initial_a = float(initial_a)
        self.cmb_temperature_0_cgs_K = float(cmb_temperature_0_cgs_K)
        self.reservoir_temperature_supercomoving_code = float(
            reservoir_temperature_supercomoving_code,
        )
        self.minimum_temperature = minimum_temperature
        self.transition_redshift = (
            None if transition_redshift is None else float(transition_redshift)
        )
        self.active_network = None

    def configure_thermochemistry(self, time_cosmic_code):
        """Select the source network at the current redshift."""
        if self.transition_redshift is None:
            return
        scale_factor = float(self.cosmology.scale_factor(time_cosmic_code))
        redshift = max(0.0, 1.0 / scale_factor - 1.0)
        if redshift > self.transition_redshift:
            self.sim.par.thermochemistry_network = "hydrogen"
            self.sim.par.metal_pie_enabled = False
            self.sim.par.hydrogen_chemistry = True
            self.sim.par.hydrogen_recombination = True
            self.sim.par.hydrogen_collisional_ionization = True
            self.sim.par.hydrogen_atomic_cooling = True
            self.sim.par.hydrogen_update_mu = True
            self.sim.par.hydrogen_thermal_coupling = True
            self.sim.par.compton_cmb_enabled = True
        else:
            self.sim.par.thermochemistry_network = "pie_uvbg_cooling"
            self.sim.par.metal_pie_enabled = True
            self.sim.par.metal_pie_redshift = self.transition_redshift
            self.sim.par.hydrogen_chemistry = False
            self.sim.par.hydrogen_update_mu = False
            self.sim.par.hydrogen_thermal_coupling = False
            self.sim.par.compton_cmb_enabled = False
        if self.sim.par.thermochemistry_network != self.active_network:
            self.active_network = self.sim.par.thermochemistry_network

    def update_cosmic_boundary(self, time_cosmic_code):
        """Set the outer cosmic gas state in supercomoving hydro units."""
        scale_factor = float(self.cosmology.scale_factor(time_cosmic_code))
        background_physical = float(
            self.cosmology.background_density(time_cosmic_code),
        )
        temperature_initial_cgs_K = self.cmb_temperature_0_cgs_K / self.initial_a  # noqa: N806
        temperature_physical = temperature_initial_cgs_K * (self.initial_a / scale_factor) ** 2
        self.sim.par.boundary.rho_inflow_proper = (
            self.baryon_fraction * background_physical * scale_factor**3
        )
        self.sim.par.boundary.vel_inflow_proper = 0.0
        self.sim.par.boundary.temperature_inflow_proper = temperature_physical * scale_factor**2
        self.sim.par.boundary.inflow_mu = float(
            self.initial_condition.get("mu", 0.59),
        )
        self.sim.par.compton_cmb_redshift = 1.0 / scale_factor - 1.0
        self.sim.par.hydro_temperature_floor = (
            None if self.minimum_temperature is None else self.minimum_temperature * scale_factor**2
        )

    def preserve_outer_background_cell(self):
        """Reset the outer active cell to the analytic background reservoir."""
        first = int(self.sim.par.mesh.ghost_cells)
        index = first + int(self.sim.par.mesh.grid_cells) - 1
        old_mass = float(np.asarray(self.sim.fluid.Mass_code, dtype=float)[index])
        old_energy = float(np.asarray(self.sim.fluid.Energy_code, dtype=float)[index])
        rho_comoving_code = float(
            np.asarray(self.sim.par.boundary.rho_inflow_proper, dtype=float),
        )
        vel_supercomoving_code = float(
            np.asarray(self.sim.par.boundary.vel_inflow_proper, dtype=float),
        )
        temperature_supercomoving_code = float(
            np.asarray(self.sim.par.boundary.temperature_inflow_proper, dtype=float),
        )
        mu = float(np.asarray(self.sim.par.boundary.inflow_mu, dtype=float))
        volume_comoving_code = float(
            np.asarray(self.sim.mesh.volume_comoving_code, dtype=float)[index],
        )
        pre_supercomoving_code = float(
            np.asarray(
                self.sim.fluid.eos.pressure(
                    rho_comoving_code,
                    temperature_supercomoving_code,
                    mu,
                ),
                dtype=float,
            ),
        )
        self.sim.fluid.rho_comoving_code[index] = rho_comoving_code
        self.sim.fluid.vel_supercomoving_code[index] = vel_supercomoving_code
        self.sim.fluid.temp_supercomoving_code[index] = (
            self.reservoir_temperature_supercomoving_code
        )
        self.sim.fluid.mu[index] = mu
        self.sim.fluid.pre_supercomoving_code[index] = pre_supercomoving_code
        self.sim.fluid.Mass_code[index] = rho_comoving_code * volume_comoving_code
        self.sim.fluid.Mom_code[index] = (
            rho_comoving_code * vel_supercomoving_code * volume_comoving_code
        )
        self.sim.fluid.Energy_code[index] = (
            float(
                np.asarray(
                    self.sim.fluid.eos.total_energy_density(
                        rho_comoving_code,
                        vel_supercomoving_code,
                        pre_supercomoving_code,
                    ),
                    dtype=float,
                ),
            )
            * volume_comoving_code
        )
        thermal_energy_density = float(
            np.asarray(
                self.sim.fluid.eos.thermal_energy_density(pre_supercomoving_code),
                dtype=float,
            ),
        )
        if hasattr(self.sim.fluid, "eth"):
            self.sim.fluid.eth_code[index] = thermal_energy_density
        if hasattr(self.sim.fluid, "InternalEnergy_code"):
            self.sim.fluid.InternalEnergy_code[index] = (
                thermal_energy_density * volume_comoving_code
            )
        return (
            float(np.asarray(self.sim.fluid.Mass_code, dtype=float)[index]) - old_mass,
            float(np.asarray(self.sim.fluid.Energy_code, dtype=float)[index]) - old_energy,
        )

    def initialize(self, time_cosmic_code):
        """Prepare the initial reservoir before the first timestep estimate."""
        self.configure_thermochemistry(time_cosmic_code)
        self.update_cosmic_boundary(time_cosmic_code)
        return self.preserve_outer_background_cell()

    def before_step(self, sim=None):
        """Update all state required before the next timestep estimate."""
        sim = self.sim if sim is None else sim
        time_cosmic_code = float(
            self.cosmology.cosmic_time_from_supercomoving(
                float(sim.fluid.tau_supercomoving_code),
            ),
        )
        self.configure_thermochemistry(time_cosmic_code)
        self.update_cosmic_boundary(time_cosmic_code)
        self.preserve_outer_background_cell()
        sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
        sim.solver.SetConserved(sim.mesh, sim.fluid)
