"""Runtime callback coordination for the cosmological gas example."""

from pathlib import Path

import numpy as np
import virial_shock_tools as et

from radhydropy.gravity import Gravity
from radhydropy.thermo_networks.pie import MetalPIETable


class CosmologicalRunCallbacks:
    """Coordinate example-specific work attached to the standard runner."""

    def __init__(
        self,
        config_filename,
        thermo,
        hydro,
        cosmology,
        baryon_fraction,
        physics,
        diagnostics,
        initial_time,
        initial_a,
    ):
        self.config_filename = Path(config_filename)
        self.thermo = thermo
        self.hydro = hydro
        self.cosmology = cosmology
        self.baryon_fraction = baryon_fraction
        self.physics = physics
        self.diagnostics = diagnostics
        self.initial_time = initial_time
        self.initial_a = initial_a
        self.runtime_configured = False
        self.history_initialized = False
        self.step_metadata = {}

    def _configure_runtime(self, sim):
        if self.runtime_configured:
            return
        sim.par.cosmology = self.cosmology
        metal_table = getattr(sim.par, "metal_pie_table", None)
        if isinstance(metal_table, dict):
            table_filename = Path(self.thermo["metal_pie_table_filename"])
            if not table_filename.is_absolute():
                table_filename = self.config_filename.parent / table_filename
            sim.par.metal_pie_table = MetalPIETable(table_filename)
        sim.par.gravity = Gravity(
            selfgravity=True,
            cosmological=True,
            cosmology=self.cosmology,
            dark_matter=(
                et.VolumeSmoothedDarkMatter(sim.par.dark_matter)
                if bool(self.hydro.get("smooth_dm_force_for_gas", False))
                else sim.par.dark_matter
            ),
            code_units=sim.par.units.CodeUnits,
        )
        sim.par.dark_matter_background_fraction = 1.0 - self.baryon_fraction
        sim.par.gas_background_fraction = self.baryon_fraction
        self.runtime_configured = True

    def before_step(self, sim):
        self._configure_runtime(sim)
        self.physics.sim = sim
        self.physics.before_step(sim)
        if hasattr(sim.fluid, "vsignal_code"):
            wall_face = int(sim.par.mesh.ghost_cells)
            sim.solver.SetInterFaceFlux(
                sim.mesh,
                sim.fluid,
                sim.par.boundary.condition,
                method=getattr(sim.par, "riemann_solver", "Rusanov"),
                order=int(sim.par.hydrodynamics.order),
            )
            self.step_metadata["wall_momentum_flux"] = float(
                np.asarray(sim.fluid.Mom_code.flux, dtype=float)[wall_face],
            )
            self.step_metadata["wall_energy_flux"] = float(
                np.asarray(sim.fluid.Energy_code.flux, dtype=float)[wall_face],
            )
        first = int(sim.par.mesh.ghost_cells)
        last = first + int(sim.par.mesh.grid_cells)
        self.step_metadata["angular_before"] = (
            float(
                np.sum(
                    np.asarray(
                        sim.fluid.AngularMomentum_code[first:last],
                        dtype=float,
                    ),
                ),
            )
            if hasattr(sim.fluid, "AngularMomentum_code")
            else 0.0
        )
        if hasattr(sim.fluid, "AngularMomentum_code") and hasattr(
            sim.fluid.AngularMomentum_code,
            "flux",
        ):
            self.step_metadata["angular_flux_area"] = np.asarray(
                sim.fluid.AngularMomentum_code.flux,
                dtype=float,
            ) * np.asarray(sim.mesh.area_comoving_code, dtype=float)

    def history(self, sim):
        if not self.history_initialized:
            self.diagnostics.initialize_step_history(
                sim,
                self.initial_time,
                self.initial_a,
            )
            self.history_initialized = True
            return
        time_cosmic_code = float(
            self.cosmology.cosmic_time_from_supercomoving(
                float(sim.fluid.tau_supercomoving_code),
            ),
        )
        self.physics.update_cosmic_boundary(time_cosmic_code)
        reservoir_mass_change, reservoir_energy_change = (
            self.physics.preserve_outer_background_cell()
        )
        first = int(sim.par.mesh.ghost_cells)
        last = first + int(sim.par.mesh.grid_cells)
        angular_after = (
            float(
                np.sum(
                    np.asarray(
                        sim.fluid.AngularMomentum_code[first:last],
                        dtype=float,
                    ),
                ),
            )
            if hasattr(sim.fluid, "AngularMomentum_code")
            else 0.0
        )
        angular_boundary_change = 0.0
        if "angular_flux_area" in self.step_metadata:
            flux_area = self.step_metadata["angular_flux_area"].copy()
            flux_area *= np.asarray(
                getattr(sim.solver, "_last_face_limiter_factors", np.ones_like(flux_area)),
                dtype=float,
            )
            angular_boundary_change = float(
                sim.last_step_dt * (flux_area[first] - flux_area[last]),
            )
        angular_change = angular_after - self.step_metadata.get("angular_before", angular_after)
        self.diagnostics.on_step(
            sim,
            dt=float(sim.last_step_dt),
            time_cosmic_code=time_cosmic_code,
            reservoir_mass_change=reservoir_mass_change,
            reservoir_energy_change=reservoir_energy_change,
            wall_momentum_flux=self.step_metadata.get("wall_momentum_flux", 0.0),
            wall_energy_flux=self.step_metadata.get("wall_energy_flux", 0.0),
            angular_after=angular_after,
            angular_change=angular_change,
            angular_boundary_change=angular_boundary_change,
            angular_residual=angular_change - angular_boundary_change,
        )

    def snapshot(self, sim, snapshot_filename, output_index):
        sim.par.cosmology = self.cosmology
        metal_table = getattr(sim.par, "metal_pie_table", None)
        if isinstance(metal_table, dict):
            table_filename = Path(self.thermo["metal_pie_table_filename"])
            if not table_filename.is_absolute():
                table_filename = self.config_filename.parent / table_filename
            sim.par.metal_pie_table = MetalPIETable(table_filename)
        if output_index == 0 and not self.runtime_configured:
            self.physics.sim = sim
            self.physics.initialize(self.initial_time)
            sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
            sim.solver.SetConserved(sim.mesh, sim.fluid)
        self.diagnostics.on_snapshot(sim, snapshot_filename, output_index)
