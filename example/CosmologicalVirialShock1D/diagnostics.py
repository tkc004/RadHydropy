"""Snapshot diagnostics for the cosmological virial-shock example."""

from pathlib import Path

import numpy as np
import virial_shock_tools as et

import radhydropy.io as rio


class CosmologicalVirialShockDiagnostics:
    """Accumulate analysis products from serialized simulation snapshots.

    The writer remains owned by the example loop for now.  ``on_snapshot`` is
    deliberately shaped like the standard runner's snapshot callback so this
    component can be connected directly when the example adopts ``RunAll``.
    """

    def __init__(
        self,
        config,
        initial_condition,
        sim,
        dark_matter,
        units,
        cosmology,
        source_diagnostics,
        gas_energy_state,
        dark_matter_energy_state,
        energy_audit_state,
        finalization_context,
    ):
        self.config = config
        self.initial_condition = initial_condition
        self.sim = sim
        self.dark_matter = dark_matter
        self.units = units
        self.cosmology = cosmology
        self.source_diagnostics = source_diagnostics
        self.gas_energy_state = gas_energy_state
        self.dark_matter_energy_state = dark_matter_energy_state
        self.energy_audit_state = energy_audit_state
        self.finalization_context = dict(finalization_context)
        self.energy_audit = {}
        self.gas_profiles = []
        self.radius_history = []
        self.dm_profiles = []
        self.hdf5_snapshot_files = []
        self.gas_energy_history = []
        self.dm_energy_history = []
        self.halo_crossing_history = []
        self.previous_halo_mask = None

    def on_snapshot(self, sim, snapshot_filename, output_index):
        """Load a completed snapshot and append all derived diagnostics."""
        self.sim = sim
        if getattr(sim.par, "dark_matter", None) is not None:
            self.dark_matter = sim.par.dark_matter
        del output_index  # The serialized filename is the authoritative ID.
        snapshot_filename = Path(snapshot_filename)
        restored_snapshot = rio.loadhdf5(self.config, str(snapshot_filename))
        restored_dark_matter = restored_snapshot.dark_matter
        dark_matter_radius_comoving_code = np.asarray(
            restored_dark_matter.radius_radarray.to(self.units.length_unit).value,
            dtype=float,
        )
        dark_matter_mass_comoving_code = np.asarray(
            restored_dark_matter.dark_matter_mass_radarray.to(self.units.mass_unit).value,
            dtype=float,
        )
        time_cosmic_code = float(
            self.cosmology.cosmic_time_from_supercomoving(
                float(sim.fluid.tau_supercomoving_code),
            ),
        )
        gas_profile = et.gas_density_profile(sim, time_cosmic_code, self.config)
        first = int(sim.par.mesh.ghost_cells)
        last = first + int(sim.par.mesh.grid_cells)
        scale_factor = float(self.cosmology.scale_factor(time_cosmic_code))
        gas_profile["temperature_proper_cgs_K"] = (
            np.asarray(sim.fluid.temp_supercomoving_code[first:last], dtype=float) / scale_factor**2
        )
        if hasattr(sim.fluid, "specific_angular_momentum_code"):
            gas_profile["specific_angular_momentum_comoving_code"] = np.asarray(
                sim.fluid.specific_angular_momentum_code[first:last],
                dtype=float,
            ).copy()
        physical_velocity = self.cosmology.physical_velocity(
            np.asarray(sim.mesh.x_comoving_code[first:last], dtype=float),
            np.asarray(sim.fluid.vel_supercomoving_code[first:last], dtype=float),
            float(sim.fluid.tau_supercomoving_code),
        )
        signed_velocity_km_s = (
            np.asarray(physical_velocity, dtype=float)
            * float(sim.par.units.CodeUnits.velocity_in_cgs)
            / 1.0e5
        )
        gas_profile["radial_velocity_proper_km_s"] = signed_velocity_km_s
        gas_profile["velocity_proper_km_s"] = np.abs(signed_velocity_km_s)
        gas_profile.update(self.source_diagnostics(sim, gas_profile))
        radius_record = et.profiles(
            sim,
            self.dark_matter,
            time_cosmic_code,
            self.config,
            density_bin_count=self.config["example"].get("dm_density_bins", 128),
        )
        gas_radius = np.asarray(gas_profile["radius_proper_kpc"], dtype=float)
        gas_edges = np.asarray(sim.mesh.boundary_comoving_code[first : last + 1], dtype=float)
        gas_mass_comoving_code = (
            np.asarray(sim.fluid.rho_comoving_code[first:last], dtype=float)
            * (4.0 * np.pi / 3.0)
            * np.diff(gas_edges**3)
        )
        rvir = float(radius_record.get("rvir_kpc", np.nan))
        mvir = float(radius_record.get("mvir", np.nan))
        if np.isfinite(rvir) and np.isfinite(mvir) and mvir > 0.0:
            gas_inside = float(np.sum(gas_mass_comoving_code[gas_radius <= rvir]))
            radius_record["gas_mass_comoving_code_rvir"] = gas_inside
            radius_record["normalized_baryon_fraction"] = gas_inside / (
                float(self.initial_condition["baryon_fraction"]) * mvir
            )
        else:
            radius_record["gas_mass_comoving_code_rvir"] = np.nan
            radius_record["normalized_baryon_fraction"] = np.nan
        shock_index = int(radius_record.get("shock_cell_index", -1))
        shock_keys = (
            ("q_cgs_erg_cm3_s", "shock_q_cgs_erg_cm3_s"),
            ("rho_dot_cgs_g_cm3_s", "shock_rho_dot_cgs_g_cm3_s"),
            ("specific_energy_cgs_erg_g", "shock_specific_energy_cgs_erg_g"),
            ("local_mach", "shock_local_mach"),
            ("gamma_eff", "shock_gamma_eff"),
        )
        for source_key, report_key in shock_keys:
            radius_record[report_key] = (
                float(np.asarray(gas_profile[source_key], dtype=float)[shock_index])
                if 0 <= shock_index < int(sim.par.mesh.grid_cells)
                else np.nan
            )
        self.gas_profiles.append(gas_profile)
        self.radius_history.append(radius_record)
        dm_profile = et.density_profiles(
            sim,
            self.dark_matter,
            time_cosmic_code,
            self.config,
            dark_matter_snapshot=restored_dark_matter,
        )
        dm_profile["dm_radius_comoving_code"] = dark_matter_radius_comoving_code
        dm_profile["dm_mass_comoving_code"] = dark_matter_mass_comoving_code
        dm_profile["scale_factor"] = scale_factor
        self.dm_profiles.append(dm_profile)
        self.gas_energy_history.append(self.gas_energy_state(sim))
        self.dm_energy_history.append(self.dark_matter_energy_state(self.dark_matter))
        current_halo_radius = radius_record.get("rvir_kpc", np.nan)
        current_halo_mask = np.isfinite(current_halo_radius) & (
            gas_radius <= float(current_halo_radius)
        )
        previous = self.previous_halo_mask
        crossing = {
            "entered_cells": int(np.count_nonzero(current_halo_mask & ~previous))
            if previous is not None
            else 0,
            "exited_cells": int(np.count_nonzero(previous & ~current_halo_mask))
            if previous is not None
            else 0,
        }
        for key in (
            "total_energy_code",
            "thermal_energy_code",
            "kinetic_energy_code",
            "gravitational_work",
            "hydro_energy_change",
            "thermochemistry_energy_change",
            "compression_work",
            "shock_work",
        ):
            if previous is None:
                crossing[key] = 0.0
            else:
                entered = current_halo_mask & ~previous
                exited = previous & ~current_halo_mask
                old_values = np.asarray(self.gas_energy_history[-2][key], dtype=float)
                new_values = np.asarray(self.gas_energy_history[-1][key], dtype=float)
                crossing[key] = float(np.sum(new_values[entered]) - np.sum(old_values[exited]))
        self.halo_crossing_history.append(crossing)
        self.previous_halo_mask = current_halo_mask
        self.hdf5_snapshot_files.append(snapshot_filename)

    def initialize_step_history(self, sim, initial_time, initial_scale_factor):
        """Initialize the accepted-step audit with the loaded IC state."""
        audit_initial = self.energy_audit_state(sim)
        first = int(sim.par.mesh.ghost_cells)
        last = first + int(sim.par.mesh.grid_cells)
        angular_initial = (
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
        self.energy_audit = {
            "step": [0],
            "time_cosmic_Gyr": [initial_time * sim.par.units.CodeUnits.time_unit.to_value("Gyr")],
            "dt": [0.0],
            "scale_factor": [initial_scale_factor],
            **{key: [value] for key, value in audit_initial.items()},
            "gravitational_work": [0.0],
            "hydro_boundary_energy_flux": [0.0],
            "background_reservoir_mass_change": [0.0],
            "background_reservoir_energy_change": [0.0],
            "thermochemistry_energy_change": [0.0],
            "energy_closure_residual": [0.0],
            "inner_wall_momentum_flux": [0.0],
            "inner_wall_energy_flux": [0.0],
            "angular_momentum_total": [angular_initial],
            "angular_momentum_change": [0.0],
            "angular_momentum_boundary_change": [0.0],
            "angular_momentum_conservation_residual": [0.0],
        }

    def on_step(
        self,
        sim,
        dt=0.0,
        step=None,
        time_cosmic_code=None,
        reservoir_mass_change=0.0,
        reservoir_energy_change=0.0,
        wall_momentum_flux=0.0,
        wall_energy_flux=0.0,
        angular_after=None,
        angular_change=0.0,
        angular_boundary_change=0.0,
        angular_residual=0.0,
    ):
        """Record diagnostics for one accepted step.

        The optional metadata is supplied by the transitional example loop.
        Defaults keep this a valid standard-runner history callback while the
        runner is not yet exposing timestep bookkeeping metadata.
        """
        if not self.energy_audit:
            raise RuntimeError("initialize_step_history() must precede on_step()")
        self.sim = sim
        if getattr(sim.par, "dark_matter", None) is not None:
            self.dark_matter = sim.par.dark_matter
        if time_cosmic_code is None:
            time_cosmic_code = float(
                self.cosmology.cosmic_time_from_supercomoving(
                    float(sim.fluid.tau_supercomoving_code),
                ),
            )
        if step is None:
            step = len(self.energy_audit["step"])
        if dt == 0.0:
            dt = float(getattr(sim, "last_step_dt", 0.0))
        audit_state = self.energy_audit_state(sim)
        previous_energy = self.energy_audit["total_gas_energy_code"][-1]
        energy_change = audit_state["total_gas_energy_code"] - previous_energy
        gravity_work = float(getattr(sim, "last_gravity_work", 0.0))
        boundary_flux = float(getattr(sim, "last_hydro_boundary_energy_flux", 0.0))
        thermo_change = float(
            getattr(sim, "last_thermochemistry_energy_change", 0.0),
        )
        floor_injection = (
            audit_state["dual_energy_floor_injected_energy"]
            - self.energy_audit["dual_energy_floor_injected_energy"][-1]
        )
        for key, value in audit_state.items():
            self.energy_audit[key].append(value)
        self.energy_audit["step"].append(step)
        self.energy_audit["time_cosmic_Gyr"].append(
            time_cosmic_code * sim.par.units.CodeUnits.time_unit.to_value("Gyr"),
        )
        self.energy_audit["dt"].append(dt)
        self.energy_audit["scale_factor"].append(
            float(self.cosmology.scale_factor(time_cosmic_code)),
        )
        self.energy_audit["gravitational_work"].append(gravity_work)
        self.energy_audit["hydro_boundary_energy_flux"].append(boundary_flux)
        self.energy_audit["background_reservoir_mass_change"].append(
            reservoir_mass_change,
        )
        self.energy_audit["background_reservoir_energy_change"].append(
            reservoir_energy_change,
        )
        self.energy_audit["thermochemistry_energy_change"].append(thermo_change)
        self.energy_audit["energy_closure_residual"].append(
            energy_change
            - boundary_flux
            - reservoir_energy_change
            - gravity_work
            - thermo_change
            - floor_injection,
        )
        self.energy_audit["inner_wall_momentum_flux"].append(wall_momentum_flux)
        self.energy_audit["inner_wall_energy_flux"].append(wall_energy_flux)
        if angular_after is None:
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
            previous = self.energy_audit["angular_momentum_total"][-1]
            angular_change = angular_after - previous
        self.energy_audit["angular_momentum_total"].append(angular_after)
        self.energy_audit["angular_momentum_change"].append(angular_change)
        self.energy_audit["angular_momentum_boundary_change"].append(
            angular_boundary_change,
        )
        self.energy_audit["angular_momentum_conservation_residual"].append(
            angular_residual,
        )

    def finalize(self, sim):
        """Write, validate, plot, and report all accumulated diagnostics."""
        context = self.finalization_context
        output_dir = context["output_dir"]
        figure_prefix = context["figure_prefix"]
        example = context["example"]
        hydro = context["hydro"]
        units = context["units"]
        initial_condition = context["initial_condition"]
        baryon_fraction = context["baryon_fraction"]
        configured_temperature_plot_ymin = context["configured_temperature_plot_ymin"]
        minimum_temperature = context["minimum_temperature"]
        context["measured_fraction"]
        plot_mass_history = context["plot_mass_history"]
        plot_radius_history = context["plot_radius_history"]
        plot_temperature_evolution = context["plot_temperature_evolution"]
        plot_specific_angular_momentum_evolution = context[
            "plot_specific_angular_momentum_evolution"
        ]
        plot_density_evolution = context["plot_density_evolution"]
        plot_temperature_density_evolution = context["plot_temperature_density_evolution"]
        plot_velocity_evolution = context["plot_velocity_evolution"]
        plot_baryon_fraction_evolution = context["plot_baryon_fraction_evolution"]
        plot_dark_matter_density_evolution = context["plot_dark_matter_density_evolution"]
        plot_baryon_normalized_density_comparison = context[
            "plot_baryon_normalized_density_comparison"
        ]
        entropy_plotter = context["entropy_plotter"]
        energy_plotter = context["energy_plotter"]
        PROTON_MASS_CGS = context["PROTON_MASS_CGS"]
        quantity_to_value = context["quantity_to_value"]
        _pad_energy_history = context["_pad_energy_history"]
        _pad_profile_history = context["_pad_profile_history"]
        gas_profiles = self.gas_profiles
        radius_history = self.radius_history
        dm_profiles = self.dm_profiles
        gas_energy_history = self.gas_energy_history
        dm_energy_history = self.dm_energy_history
        halo_crossing_history = self.halo_crossing_history
        energy_audit = self.energy_audit
        cosmology = self.cosmology

        times = np.asarray([item["time_cosmic_Gyr"] for item in gas_profiles])
        radius_comoving_code = np.asarray(gas_profiles[0]["radius_comoving_kpc"])
        rho_comoving_code = np.asarray([item["rho_proper_code"] for item in gas_profiles])
        temperature_proper_cgs_K = np.asarray(
            [item["temperature_proper_cgs_K"] for item in gas_profiles],
        )
        vel_supercomoving_code = np.asarray(
            [item["velocity_proper_km_s"] for item in gas_profiles],
        )
        specific_angular_momentum_comoving_code = np.asarray(
            [
                item.get(
                    "specific_angular_momentum_comoving_code",
                    np.zeros_like(radius_comoving_code),
                )
                for item in gas_profiles
            ],
        )
        radial_velocity = np.asarray(
            [item["radial_velocity_proper_km_s"] for item in gas_profiles],
        )
        scale_factors = np.asarray([item["scale_factor"] for item in gas_profiles])
        plot_exclude_outer_cells = max(
            0,
            int(example.get("plot_exclude_outer_cells", 0)),
        )
        plot_cell_count = max(1, radius_comoving_code.size - plot_exclude_outer_cells)
        plot_radius = radius_comoving_code[:plot_cell_count]
        plot_density = rho_comoving_code[:, :plot_cell_count]
        plot_temperature = temperature_proper_cgs_K[:, :plot_cell_count]
        plot_velocity = vel_supercomoving_code[:, :plot_cell_count]
        plot_gas_profiles = []
        for profile in gas_profiles:
            trimmed = dict(profile)
            for key in (
                "radius_proper_kpc",
                "rho_proper_code",
                "temperature_proper_cgs_K",
                "velocity_proper_km_s",
                "radial_velocity_proper_km_s",
                "specific_angular_momentum_comoving_code",
            ):
                if key in trimmed:
                    trimmed[key] = np.asarray(trimmed[key])[:plot_cell_count]
            plot_gas_profiles.append(trimmed)
        virial_radius = np.asarray([item["rvir_kpc"] for item in radius_history])
        splashback_radius = np.asarray(
            [item["rsplashback_kpc"] for item in radius_history],
        )
        virial_temperature = np.asarray([item["tvir_cgs_K"] for item in radius_history])
        history = {
            key: np.asarray([item[key] for item in radius_history]) for key in radius_history[0]
        }
        history["scale_factor"] = scale_factors
        data_file = output_dir / (figure_prefix + ".npz")
        np.savez(
            data_file,
            **history,
            radius_comoving_kpc=radius_comoving_code,
            rho_proper_code=rho_comoving_code,
            temperature_proper_cgs_K=temperature_proper_cgs_K,
            velocity_proper_km_s=vel_supercomoving_code,
            radial_velocity_proper_km_s=radial_velocity,
            specific_angular_momentum_comoving_code=specific_angular_momentum_comoving_code,
            q_cgs_erg_cm3_s=np.asarray([item["q_cgs_erg_cm3_s"] for item in gas_profiles]),
            rho_dot_cgs_g_cm3_s=np.asarray([item["rho_dot_cgs_g_cm3_s"] for item in gas_profiles]),
            specific_energy_cgs_erg_g=np.asarray(
                [item["specific_energy_cgs_erg_g"] for item in gas_profiles],
            ),
            local_mach=np.asarray([item["local_mach"] for item in gas_profiles]),
            gamma_eff=np.asarray([item["gamma_eff"] for item in gas_profiles]),
            rvir_proper_kpc=virial_radius,
            virial_temperature_cgs_K=virial_temperature,
        )
        baryon_fraction_figure = output_dir / (
            figure_prefix + "_BaryonMassFraction_TimeEvolution.jpg"
        )
        plot_baryon_fraction_evolution(
            times,
            np.asarray([item["normalized_baryon_fraction"] for item in radius_history]),
            np.asarray([item["mvir"] for item in radius_history])
            * float(sim.par.units.CodeUnits.mass_in_cgs)
            / 1.98847e33,
            scale_factors,
            baryon_fraction_figure,
            float(initial_condition["baryon_fraction"]),
        )
        entropy_plotter.main(
            output_dir,
            figure_prefix,
            gamma=float(hydro["gamma"]),
            exclude_outer_cells=plot_exclude_outer_cells,
        )
        energy_audit_file = output_dir / (figure_prefix + "_EnergyAudit.npz")
        np.savez(
            energy_audit_file,
            **{key: np.asarray(value, dtype=float) for key, value in energy_audit.items()},
        )
        # Per-snapshot component histories complement the per-step global audit.
        # Gas rows are physical cells; DM rows are shell IDs, padded with NaN when
        # a shell has been absorbed into the unresolved central core.
        per_cell = {
            "time_cosmic_Gyr": np.asarray([item["time_cosmic_Gyr"] for item in gas_profiles]),
            "mass_code": _pad_energy_history(gas_energy_history, "mass_code"),
            "total_energy_code": _pad_energy_history(gas_energy_history, "total_energy_code"),
            "kinetic_energy_code": _pad_energy_history(gas_energy_history, "kinetic_energy_code"),
            "thermal_energy_code": _pad_energy_history(gas_energy_history, "thermal_energy_code"),
            "gravitational_work": _pad_energy_history(
                gas_energy_history,
                "gravitational_work",
            ),
            "hydro_energy_change": _pad_energy_history(
                gas_energy_history,
                "hydro_energy_change",
            ),
            "thermochemistry_energy_change": _pad_energy_history(
                gas_energy_history,
                "thermochemistry_energy_change",
            ),
            "compression_work": _pad_energy_history(
                gas_energy_history,
                "compression_work",
            ),
            "shock_work": _pad_energy_history(gas_energy_history, "shock_work"),
            "dual_energy_total_thermal": _pad_energy_history(
                gas_energy_history,
                "dual_energy_total_thermal",
            ),
            "dual_energy_internal_density": _pad_energy_history(
                gas_energy_history,
                "dual_energy_internal_density",
            ),
            "dual_energy_total_pressure": _pad_energy_history(
                gas_energy_history,
                "dual_energy_total_pressure",
            ),
            "dual_energy_dual_pressure": _pad_energy_history(
                gas_energy_history,
                "dual_energy_dual_pressure",
            ),
            "dual_energy_total_valid": _pad_energy_history(
                gas_energy_history,
                "dual_energy_total_valid",
            ),
            "dual_energy_dual_valid": _pad_energy_history(
                gas_energy_history,
                "dual_energy_dual_valid",
            ),
            "dual_energy_pressure_selection_code": _pad_energy_history(
                gas_energy_history,
                "dual_energy_pressure_selection_code",
            ),
            "halo_crossing_total_energy": np.asarray(
                [item["total_energy_code"] for item in halo_crossing_history],
                dtype=float,
            ),
            "halo_crossing_thermal_energy": np.asarray(
                [item["thermal_energy_code"] for item in halo_crossing_history],
                dtype=float,
            ),
            "halo_crossing_kinetic_energy": np.asarray(
                [item["kinetic_energy_code"] for item in halo_crossing_history],
                dtype=float,
            ),
            "halo_crossing_gravitational_work": np.asarray(
                [item["gravitational_work"] for item in halo_crossing_history],
                dtype=float,
            ),
            "halo_crossing_thermochemistry_energy_change": np.asarray(
                [item["thermochemistry_energy_change"] for item in halo_crossing_history],
                dtype=float,
            ),
            "halo_crossing_compression_work": np.asarray(
                [item["compression_work"] for item in halo_crossing_history],
                dtype=float,
            ),
            "halo_crossing_shock_work": np.asarray(
                [item["shock_work"] for item in halo_crossing_history],
                dtype=float,
            ),
            "halo_entered_cells": np.asarray(
                [item["entered_cells"] for item in halo_crossing_history],
                dtype=int,
            ),
            "halo_exited_cells": np.asarray(
                [item["exited_cells"] for item in halo_crossing_history],
                dtype=int,
            ),
        }
        per_cell["delta_total_energy_code"] = (
            per_cell["total_energy_code"] - per_cell["total_energy_code"][0]
        )
        per_cell["delta_kinetic_energy_code"] = (
            per_cell["kinetic_energy_code"] - per_cell["kinetic_energy_code"][0]
        )
        per_cell["delta_thermal_energy_code"] = (
            per_cell["thermal_energy_code"] - per_cell["thermal_energy_code"][0]
        )
        per_cell["energy_balance_residual"] = (
            per_cell["delta_total_energy_code"]
            - per_cell["hydro_energy_change"]
            - per_cell["gravitational_work"]
            - per_cell["thermochemistry_energy_change"]
        )
        per_shell = {
            "time_cosmic_Gyr": np.asarray([item["time_cosmic_Gyr"] for item in dm_profiles]),
            "shell_id": _pad_energy_history(dm_energy_history, "id", fill=-1),
            "radius_comoving_code": _pad_energy_history(dm_energy_history, "radius_comoving_code"),
            "vel_supercomoving_code": _pad_energy_history(
                dm_energy_history,
                "vel_supercomoving_code",
            ),
            "mass_code": _pad_energy_history(dm_energy_history, "mass_code"),
            "kinetic_energy_code": _pad_energy_history(dm_energy_history, "kinetic_energy_code"),
            "potential_energy_code": _pad_energy_history(
                dm_energy_history,
                "potential_energy_code",
            ),
            "total_energy_code": _pad_energy_history(dm_energy_history, "total_energy_code"),
        }
        per_shell["delta_kinetic_energy_code"] = (
            per_shell["kinetic_energy_code"] - per_shell["kinetic_energy_code"][0]
        )
        per_shell["delta_potential_energy_code"] = (
            per_shell["potential_energy_code"] - per_shell["potential_energy_code"][0]
        )
        per_shell["delta_total_energy_code"] = (
            per_shell["total_energy_code"] - per_shell["total_energy_code"][0]
        )
        energy_entity_file = output_dir / (figure_prefix + "_EnergyByCellAndShell.npz")
        np.savez(
            energy_entity_file,
            **{f"gas_{key}": value for key, value in per_cell.items()},
            **{f"dm_{key}": value for key, value in per_shell.items()},
        )
        energy_balance_figure = None
        if bool(hydro.get("energy_diagnostics", True)):
            try:
                energy_plotter.main(output_dir, figure_prefix, radius_factor=2.0)
                energy_balance_figure = output_dir / (
                    figure_prefix + "_2RvirEnergyBalance_TimeEvolution.jpg"
                )
            except RuntimeError:
                # A developing perturbation may not yet have a resolved r200.
                # Keep the run and its ordinary diagnostics usable; the energy
                # plot will be generated automatically once a resolved halo exists.
                pass
        figure = output_dir / (figure_prefix + ".jpg")
        radius_figure = output_dir / (figure_prefix + "_Radii.jpg")
        plot_mass_history(history, figure)
        plot_radius_history(history, radius_figure)
        temperature_figure = output_dir / (figure_prefix + "_Temperatures.jpg")
        temperature_plot_ymin = (
            configured_temperature_plot_ymin
            if configured_temperature_plot_ymin is not None
            else minimum_temperature
        )
        if temperature_plot_ymin is not None:
            if hasattr(temperature_plot_ymin, "to_value"):
                temperature_plot_ymin = float(temperature_plot_ymin.to_value("K"))
            else:
                temperature_plot_ymin = float(temperature_plot_ymin)
        plot_temperature_evolution(
            times,
            plot_radius,
            plot_density,
            plot_temperature,
            virial_radius,
            splashback_radius,
            scale_factors,
            virial_temperature,
            temperature_figure,
            minimum_temperature=temperature_plot_ymin,
            inner_radius=quantity_to_value(
                initial_condition.get(
                    "inner_wall_radius_comoving",
                    initial_condition["radius_inner_comoving"],
                ),
                units.length_unit,
            ),
            box_boundary=quantity_to_value(
                initial_condition["radius_outer_comoving"],
                units.length_unit,
            ),
        )
        specific_angular_momentum_figure = output_dir / (
            figure_prefix + "_SpecificAngularMomentum.jpg"
        )
        plot_specific_angular_momentum_evolution(
            times,
            plot_radius,
            plot_density,
            specific_angular_momentum_comoving_code[:, :plot_cell_count],
            virial_radius,
            splashback_radius,
            scale_factors,
            specific_angular_momentum_figure,
        )
        density_figure = output_dir / (figure_prefix + "_Densities.jpg")
        cosmic_gas_density_z0 = baryon_fraction * float(
            cosmology.background_density(cosmology.t_ref),
        )
        plot_density_evolution(
            times,
            plot_radius,
            plot_density,
            virial_radius,
            scale_factors,
            density_figure,
            ymin=0.1 * cosmic_gas_density_z0,
        )
        temperature_density_figure = output_dir / (figure_prefix + "_TemperatureDensity.jpg")
        plot_temperature_density_evolution(
            times,
            plot_density,
            plot_temperature,
            temperature_density_figure,
            ymin=0.1,
            density_to_nH_cgs_cm3=(
                float(sim.par.units.CodeUnits.mass_in_cgs)
                / float(sim.par.units.CodeUnits.length_in_cgs) ** 3
                * float(initial_condition["hydrogen_mass_fraction"])
                / PROTON_MASS_CGS
            ),
        )
        velocity_figure = output_dir / (figure_prefix + "_Velocity.jpg")
        plot_velocity_evolution(
            times,
            plot_radius,
            plot_density,
            plot_velocity,
            virial_radius,
            scale_factors,
            velocity_figure,
        )
        dm_figure = output_dir / (figure_prefix + "_DarkMatterDensities.jpg")
        plot_dark_matter_density_evolution(
            dm_profiles,
            plot_radius,
            dm_figure,
            density_bin_count=example.get("dm_density_bins"),
        )
        density_comparison_figure = output_dir / (
            figure_prefix + "_GasDarkMatterBaryonNormalized.jpg"
        )
        plot_baryon_normalized_density_comparison(
            plot_gas_profiles,
            dm_profiles,
            initial_condition["baryon_fraction"],
            virial_radius,
            density_comparison_figure,
        )
        dm_data_file = output_dir / (figure_prefix + "_DarkMatterDensities.npz")
        np.savez(
            dm_data_file,
            time_cosmic_Gyr=np.asarray([item["time_cosmic_Gyr"] for item in dm_profiles]),
            scale_factor=np.asarray([item["scale_factor"] for item in dm_profiles]),
            mean_rho_comoving_code=np.asarray(
                [item.get("dm_mean_density_code", np.nan) for item in dm_profiles],
            ),
            radius_proper_kpc=_pad_profile_history(dm_profiles, "dm_radius_proper_kpc"),
            rho_proper_code=_pad_profile_history(dm_profiles, "dm_rho_proper_code"),
            mass_code=_pad_profile_history(dm_profiles, "dm_mass_comoving_code"),
            softening_comoving_code=np.asarray(
                [item.get("dm_softening_comoving_code", np.nan) for item in dm_profiles],
            ),
            total_mass=np.asarray(
                [item.get("dm_total_mass_comoving_code", np.nan) for item in dm_profiles],
            ),
            crossing_events=np.asarray(
                [item.get("dm_crossing_events", 0) for item in dm_profiles],
                dtype=int,
            ),
            origin_reflections=np.asarray(
                [item.get("dm_origin_reflections", 0) for item in dm_profiles],
                dtype=int,
            ),
            central_core_mass=np.asarray(
                [item.get("dm_central_core_mass", 0.0) for item in dm_profiles],
            ),
            central_core_radius_kpc=np.asarray(
                [item.get("dm_central_core_radius_kpc", 0.0) for item in dm_profiles],
            ),
        )
        dm_substeps = np.asarray(sim.dark_matter_substep_history, dtype=int)
        dm_total_mass = np.asarray(
            [item.get("dm_total_mass_comoving_code", np.nan) for item in dm_profiles],
        )
        if dm_total_mass.size and np.isfinite(dm_total_mass[0]):
            mass_error = np.max(np.abs(dm_total_mass - dm_total_mass[0]))
            if mass_error > 1.0e-10 * max(abs(dm_total_mass[0]), 1.0):
                raise RuntimeError(
                    f"live dark-matter mass is not conserved: maximum error {mass_error:.8g}",
                )
        if dm_substeps.size:
            pass
        if energy_balance_figure is not None:
            pass
        return data_file


def _pad_profile_history(profiles, key):
    arrays = [np.asarray(item[key], dtype=float).ravel() for item in profiles]
    width = max((array.size for array in arrays), default=0)
    result = np.full((len(arrays), width), np.nan, dtype=float)
    for row, array in enumerate(arrays):
        result[row, : array.size] = array
    return result
