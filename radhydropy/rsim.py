"""High-level simulation runner."""

import copy
import radhydropy.utils as ru
import radhydropy.io as rio
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
from radhydropy.units import (
    _CODE_UNIT_GROUPS,
    apply_code_unit_specs,
    code_units_from_system,
    time_seconds,
    to_code_value,
)
from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.params import Par
from radhydropy.solver import Solver
from pathlib import Path
import unyt
import numpy as np
import time
start_time = time.time()

class Rsim():
    """Coordinate parameters, mesh, fluid state, solver, and output."""

    def __init__(self,params) -> None:
        """Create a simulation from a run-parameter dictionary."""
        print("--- Get simulation parameters ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.fluid = Fluid()
        self.mesh  = Mesh()
        self.par    = Par(params)
        self.solver = Solver()
        self.fluid.eos = EOS(
            self.par.EOStype,
            self.par.gamma,
            getattr(self.par, 'CodeUnits', None),
        )

    @classmethod
    def FromComponents(cls, par, mesh, fluid, solver=None):
        """Create a runner from already-initialized objects."""
        sim = cls.__new__(cls)
        sim.par = par
        sim.mesh = mesh
        sim.fluid = fluid
        sim.solver = solver if solver is not None else Solver()
        return sim
        

    def Callreadhdf5(self):
        """Read the configured initial-condition HDF5 file."""
        print("--- Read Initial Condition ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        self._require_code_units()
        rio.readhdf5(self.par, self.mesh, self.fluid, self.par.ICfilename)
        self.checkparams()
        self.fluid.SetFluidTime(self.par.time)
        print("--- Start Initial Time ---")

    def SetMesh(self):
        """Initialize mesh geometry and ghost cells."""
        print("--- Set up the Mesh ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.mesh.SetUpMesh(self.par)


    def SetFluid(self):
        """Initialize fluid ghost cells and pressure."""
        print("--- Set up the fluid ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.fluid.SetUpFluid(self.par)
    
    def SetInitFluid(self):
        """Apply initial boundaries and populate conserved variables."""
        print("--- Fill up the fluid---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.ConvertParametersToCodeUnits()
        self.solver.SetBoundary(self.mesh,self.fluid,self.par)
        self.solver.SetConserved(self.mesh,self.fluid, verbose=getattr(self.par, 'verbose', 0))
        self.solver.ApplyRadiativeTransfer(self.mesh, self.fluid, self.par)

    def ConvertToCodeUnits(self):
        """Convert the runtime mesh and fluid state into the internal unit system."""
        code = self._require_code_units()
        if getattr(self, '_runtime_converted_to_code_units', False):
            return

        units = code_units_from_system(code)
        length_unit = units['length']
        mass_unit = units['mass']
        time_unit = units['time']
        velocity_unit = units['velocity']
        area_unit = length_unit ** 2
        volume_unit = length_unit ** 3
        mass_flux_unit = mass_unit / (length_unit ** 2 * time_unit)

        derived_units = {
            'length_inv': 1.0 / length_unit,
            'area': area_unit,
            'volume': volume_unit,
            'number_density': 1.0 / volume_unit,
            'momentum': mass_unit * velocity_unit,
            'mass_flux': mass_flux_unit,
            'photon_flux': 1.0 / (area_unit * time_unit),
            'photon_rate': 1.0 / time_unit,
            'alpha': length_unit ** 3 / time_unit,
        }
        unit_map = {**units, **derived_units}

        for group in _CODE_UNIT_GROUPS:
            apply_code_unit_specs(getattr(self, group.target), group.specs, unit_map)
        self._runtime_converted_to_code_units = True

    def ConvertParametersToCodeUnits(self):
        """Convert only the runtime parameters into the internal unit system."""
        code = self._require_code_units()
        if getattr(self, '_runtime_parameters_converted_to_code_units', False):
            return

        units = code_units_from_system(code)
        length_unit = units['length']
        mass_unit = units['mass']
        time_unit = units['time']
        unit_map = {
            **units,
            'length_inv': 1.0 / length_unit,
            'area': length_unit ** 2,
            'volume': length_unit ** 3,
            'number_density': 1.0 / (length_unit ** 3),
            'momentum': mass_unit * units['velocity'],
            'mass_flux': mass_unit / (length_unit ** 2 * time_unit),
            'photon_flux': 1.0 / (length_unit ** 2 * time_unit),
            'photon_rate': 1.0 / time_unit,
            'alpha': length_unit ** 3 / time_unit,
            'acceleration': length_unit / time_unit ** 2,
            'potential': units['velocity'] ** 2,
        }
        apply_code_unit_specs(self.par, _CODE_UNIT_GROUPS[-1].specs, unit_map)
        self._runtime_parameters_converted_to_code_units = True

    def _require_code_units(self):
        """Return the active code-unit system or fail fast during startup."""
        code = getattr(self.par, 'CodeUnits', None)
        if code is None:
            raise ValueError("simulation startup requires par.CodeUnits")
        return code

    def WriteUsedParameters(self, filename="used_parameters.yaml"):
        """Write the active runtime parameters to a text file in the CWD."""
        return rio.write_used_parameters(Path.cwd() / filename, self.par)

    def GetStepTime(self, dt=None, final_time=None):
        """Return a timestep, clipped to ``final_time`` when supplied."""
        if dt is None:
            dt = self.solver.GetTimeStep(self.mesh, self.fluid, self.par)
        current_time = self.fluid.time
        if final_time is not None:
            if hasattr(final_time, "units"):
                target_units = final_time.units
                if not hasattr(current_time, "to_value"):
                    current_time = current_time * target_units
                if not hasattr(dt, "to_value"):
                    dt = dt * target_units
            elif hasattr(current_time, "units") and not hasattr(dt, "to_value"):
                dt = dt * current_time.units
        if final_time is not None and current_time + dt > final_time:
            dt = final_time - current_time
        return dt

    def PrepareConservedStep(self, fluid=None):
        """Apply boundaries and refresh conserved variables before a step."""
        if fluid is None:
            fluid = self.fluid
        self.solver.SetBoundary(self.mesh, fluid, self.par)
        self.solver.SetConserved(self.mesh, fluid, verbose=getattr(self.par, 'verbose', 0))

    def AdvanceHydroFluxes(self, dt, fluid=None):
        """Advance the Euler flux update and return mass data for scalar advection."""
        if fluid is None:
            fluid = self.fluid
        old_mass = fluid.Mass.copy()
        self.solver.SetInterFaceFlux(
            self.mesh,
            fluid,
            self.par.boundcond,
            verbose=getattr(self.par, 'verbose', 0),
            order=self.par.order,
        )
        mass_flux = fluid.Mass.flux.copy()
        self.solver.AddFluxes(dt, self.mesh, fluid, self.par.boundcond)
        return old_mass, mass_flux

    def AdvectChemistryScalars(self, dt, old_mass, mass_flux, fluid=None):
        """Advect passive thermo-chemistry scalars after a hydro flux update."""
        if fluid is None:
            fluid = self.fluid
        self.solver.AdvectIonizationFraction(
            dt,
            self.mesh,
            fluid,
            self.par,
            old_mass,
            mass_flux,
        )

    def UpdateThermochemistryPrimitiveState(self, update_pressure=True, fluid=None):
        """Refresh temperature, mean molecular weight, and optionally pressure."""
        if fluid is None:
            fluid = self.fluid
        if not rtc.thermochemistry_enabled(fluid, self.par):
            return
        if getattr(self.par, 'hydrogen_update_mu', False):
            fluid.SetHydrogenMu(
                hydrogen_mass_fraction=getattr(
                    self.par,
                    'hydrogen_mass_fraction',
                    1.0,
                )
            )
        fluid.SetTemperature()
        if update_pressure:
            fluid.SetPressure()

    def _sync_hydro_state(self, fluid=None):
        """Refresh primitive and conserved variables after a hydro update."""
        if fluid is None:
            fluid = self.fluid
        self.solver.SetPrimitive(self.mesh, fluid, verbose=getattr(self.par, 'verbose', 0))
        self.UpdateThermochemistryPrimitiveState(update_pressure=True, fluid=fluid)
        self.solver.SetConserved(self.mesh, fluid, verbose=getattr(self.par, 'verbose', 0))

    def FinalizeHydroStep(
        self,
        dt,
        old_mass,
        mass_flux,
        advect_chemistry=True,
        fluid=None,
    ):
        """Complete a hydro step after conserved variables have been advanced."""
        if fluid is None:
            fluid = self.fluid
        self.solver.ApplyExternalGravity(dt, self.mesh, fluid, self.par)
        if advect_chemistry:
            self.AdvectChemistryScalars(dt, old_mass, mass_flux, fluid=fluid)
        self._sync_hydro_state(fluid=fluid)

    def ApplyThermochemistrySources(self, dt):
        """Apply radiative-transfer and thermo-chemistry source updates."""
        self.solver.ApplyRadiativeTransfer(self.mesh, self.fluid, self.par)
        return self.solver.ApplyThermochemistryFast(
            dt,
            self.mesh,
            self.fluid,
            self.par,
        )

    def _clone_fluid(self, fluid=None):
        """Return a deep copy of the supplied fluid state."""
        if fluid is None:
            fluid = self.fluid
        return copy.deepcopy(fluid)

    def _hydro_step_once(self, dt, fluid=None, advect_chemistry=True):
        """Advance one explicit hydro step on the supplied fluid state."""
        if fluid is None:
            fluid = self.fluid
        self.PrepareConservedStep(fluid=fluid)
        old_mass, mass_flux = self.AdvanceHydroFluxes(dt, fluid=fluid)
        self.FinalizeHydroStep(
            dt,
            old_mass,
            mass_flux,
            advect_chemistry=advect_chemistry,
            fluid=fluid,
        )
        return {
            "dt": dt,
            "hydro_steps": 1,
            "source_steps": 0,
        }

    def _hydro_step_ssprk2(self, dt, advect_chemistry=True):
        """Advance hydro variables with the SSPRK2 strong-stability-preserving scheme."""
        initial_state = self._clone_fluid()
        stage1 = self._clone_fluid()
        self._hydro_step_once(
            dt,
            fluid=stage1,
            advect_chemistry=advect_chemistry,
        )
        stage2 = self._clone_fluid(stage1)
        self._hydro_step_once(
            dt,
            fluid=stage2,
            advect_chemistry=advect_chemistry,
        )

        for attr in ("Mass", "Mom", "Energy"):
            setattr(
                self.fluid,
                attr,
                0.5 * getattr(initial_state, attr) + 0.5 * getattr(stage2, attr),
            )
        if advect_chemistry and hasattr(initial_state, "xHI") and hasattr(stage2, "xHI"):
            self.fluid.xHI = 0.5 * initial_state.xHI + 0.5 * stage2.xHI

        self.fluid.time = initial_state.time + dt
        self._sync_hydro_state()
        return {
            "dt": dt,
            "hydro_steps": 1,
            "source_steps": 0,
        }

    def Step(
        self,
        dt=None,
        mode="hydro_sources",
        advect_chemistry=True,
        hydro_integrator="euler",
    ):
        """Advance one canonical simulation step in the requested mode."""
        valid_modes = ("hydro", "hydro_sources", "sources")
        if mode not in valid_modes:
            raise ValueError(
                "Unknown step mode %r; valid modes are %s"
                % (mode, ", ".join(valid_modes))
            )
        valid_hydro_integrators = ("euler", "ssprk2")
        if hydro_integrator not in valid_hydro_integrators:
            raise ValueError(
                "Unknown hydro integrator %r; valid options are %s"
                % (hydro_integrator, ", ".join(valid_hydro_integrators))
            )
        dt = self.GetStepTime(dt=dt)
        result = {
            "dt": dt,
            "hydro_steps": 0,
            "source_steps": 0,
        }

        if mode in ("hydro", "hydro_sources"):
            if hydro_integrator == "ssprk2":
                result = self._hydro_step_ssprk2(
                    dt,
                    advect_chemistry=advect_chemistry,
                )
            else:
                self.PrepareConservedStep()
                old_mass, mass_flux = self.AdvanceHydroFluxes(dt)
                self.FinalizeHydroStep(
                    dt,
                    old_mass,
                    mass_flux,
                    advect_chemistry=advect_chemistry,
                )
                result["hydro_steps"] = 1

        if mode in ("hydro_sources", "sources"):
            result["source_steps"] = self.ApplyThermochemistrySources(
                dt,
            )
            # Source updates can change temperature, pressure, and chemistry
            # fields, so refresh the boundary state before the next loop.
            if mode == "sources":
                self.fluid.time += dt
            self.solver.SetBoundary(self.mesh, self.fluid, self.par)
            self.solver.SetConserved(self.mesh, self.fluid, verbose=getattr(self.par, 'verbose', 0))

        return result

    def Evolve(
        self,
        final_time=None,
        mode="hydro_sources",
        advect_chemistry=True,
        history_callback=None,
        output_callback=None,
        stop_condition=None,
        step_backend=None,
        step_backend_kwargs=None,
    ):
        """Evolve the simulation with a pluggable step backend."""
        if final_time is None:
            final_time = self.par.timesim
        if step_backend is None:
            step_backend = self.Step
        if step_backend_kwargs is None:
            step_backend_kwargs = {}
        counters = {"hydro_steps": 0, "source_steps": 0}
        if history_callback is not None:
            history_callback(self)
        while self.fluid.time < final_time:
            if stop_condition is not None and stop_condition(self):
                break
            dt = self.GetStepTime(final_time=final_time)
            step = step_backend(
                dt=dt,
                mode=mode,
                advect_chemistry=advect_chemistry,
                **step_backend_kwargs,
            )
            counters["hydro_steps"] += step["hydro_steps"]
            counters["source_steps"] += step["source_steps"]
            if history_callback is not None:
                history_callback(self)
            if output_callback is not None:
                output_callback(self, step)
        return counters

    def _static_front_radius_from_state(self, state, neutral_fraction=0.5):
        ionized = state['xHI'] <= neutral_fraction
        if not np.any(ionized):
            return 0.0
        if np.all(ionized):
            return state['radius_kpc'][-1]
        left = np.where(ionized)[0][-1]
        right = left + 1
        x_left = state['xHI'][left]
        x_right = state['xHI'][right]
        if x_right == x_left:
            return state['radius_kpc'][left]
        weight = (neutral_fraction - x_left) / (x_right - x_left)
        return state['radius_kpc'][left] + weight * (
            state['radius_kpc'][right] - state['radius_kpc'][left]
        )

    def _append_static_history(
        self,
        history,
        state,
        ngamma,
        time_s,
        recombined_photons,
        source_rate_s,
        seconds_to_myr,
    ):
        ionized = 1.0 - state['xHI']
        ionized_atoms = np.sum(ionized * state['nH_cm3'] * state['volume_cm3'])
        volume_photons = np.sum(ngamma * state['volume_cm3'])
        history['time_Myr'].append(time_s * seconds_to_myr)
        history['front_radius_kpc'].append(self._static_front_radius_from_state(state))
        history['injected_photons'].append(source_rate_s * time_s)
        history['ionized_atoms'].append(ionized_atoms)
        history['recombined_photons'].append(recombined_photons)
        history['volume_photons'].append(volume_photons)
        history['accounted_photons'].append(
            ionized_atoms + recombined_photons + volume_photons
        )
        if 'mean_ionized_temp_K' in history:
            ionized_weight = 1.0 - state['xHI']
            if np.sum(ionized_weight) > 0.0:
                mean_temp = np.sum(ionized_weight * state['temperature_K']) / np.sum(
                    ionized_weight
                )
            else:
                mean_temp = 0.0
            history['mean_ionized_temp_K'].append(float(mean_temp))

    def _snapshot_static_state(self, state, time_s):
        return {
            'time_Myr': time_s,
            'radius_kpc': state['radius_kpc'].copy(),
            'xHI': state['xHI'].copy(),
            'temperature_K': state['temperature_K'].copy(),
        }

    def _initial_static_history(self, include_thermal_history=False):
        history = {
            'time_Myr': [],
            'front_radius_kpc': [],
            'injected_photons': [],
            'ionized_atoms': [],
            'recombined_photons': [],
            'volume_photons': [],
            'accounted_photons': [],
        }
        if include_thermal_history:
            history['mean_ionized_temp_K'] = []
        return history

    def _static_reference_time_seconds(self, reference_time):
        if reference_time is None:
            return None
        return time_seconds(
            reference_time,
            getattr(self.par, 'CodeUnits', None),
        )

    def _static_step_limit_seconds(self, time_s, final_time_s, dtmax_s, reference_time_s, history):
        remaining_s = final_time_s - time_s
        dtmax_step_s = min(dtmax_s, remaining_s)
        if (
            reference_time_s is not None
            and 'reference_snapshot' not in history
            and time_s < reference_time_s <= time_s + dtmax_step_s
        ):
            dtmax_step_s = reference_time_s - time_s
        return remaining_s, dtmax_step_s

    def _static_recombination_rate(self, state):
        alpha = getattr(self.par, 'hydrogen_alpha_B', None)
        if alpha is None:
            return 0.0
        if hasattr(alpha, 'to_value'):
            alpha_value = alpha.to_value(unyt.cm**3 / unyt.s)
        else:
            alpha_value = float(alpha)
        ionized = 1.0 - state['xHI']
        return np.sum(
            alpha_value
            * ionized**2
            * state['nH_cm3']**2
            * state['volume_cm3']
        )

    def _apply_static_thermal_update(self, state, ngamma, thermal_rate, dt_s):
        if not getattr(self.par, 'hydrogen_thermal_coupling', True):
            return
        if thermal_rate is None:
            thermal_rate = rtc.thermal_rate(state, ngamma, self.par)
        state['specific_energy_erg_g'] += thermal_rate / state['rho_g_cm3'] * dt_s
        state['specific_energy_erg_g'] = np.maximum(
            state['specific_energy_erg_g'],
            1.0e6,
        )
        rtc.update_temperature_from_energy(state)

    def _advance_source_thermochemistry_state(self, state, ngamma, dt_s, thermal_rate):
        recombination_rate_start = self._static_recombination_rate(state)
        self._apply_static_thermal_update(state, ngamma, thermal_rate, dt_s)
        rtc.ionization_fraction_implicit_update(state, ngamma, dt_s, self.par)
        if getattr(self.par, 'hydrogen_thermal_coupling', True):
            rtc.update_temperature_from_energy(state)
        recombination_rate_end = self._static_recombination_rate(state)
        return 0.5 * (recombination_rate_start + recombination_rate_end) * dt_s

    def _refresh_static_photon_density(self, state, step, time_s, final_time_s):
        if time_s < final_time_s:
            return None, 0
        ngamma = rrt.trace_photon_density(state, self.par)
        return ngamma, 1

    def _store_static_reference_snapshot(self, history, state, time_s, reference_time_s):
        if (
            reference_time_s is not None
            and 'reference_snapshot' not in history
            and time_s >= reference_time_s
        ):
            history['reference_snapshot'] = self._snapshot_static_state(state, time_s)

    def _finish_static_thermochemistry(self, state, time_s):
        state['ngamma'] = rrt.trace_photon_density(state, self.par)
        state['time_s'] = time_s
        rtc.apply_state(state, self.fluid, self.par)
        self.solver.SetBoundary(self.mesh, self.fluid, self.par)

    def EvolveStaticThermochemistry(
        self,
        final_time,
        source_timestep,
        include_thermal_history=False,
        reference_time=None,
    ):
        """Evolve fixed-density thermo-chemistry/radiation source terms."""
        state = rtc.source_state(self.mesh, self.fluid, self.par)
        ngamma = rrt.trace_photon_density(state, self.par)
        recombined_photons = 0.0
        time_s = 0.0
        code_units = getattr(self.par, 'CodeUnits', None)
        final_time_s = time_seconds(final_time, code_units)
        dtmax_s = time_seconds(source_timestep, code_units)
        reference_time_s = self._static_reference_time_seconds(reference_time)
        source_rate = getattr(
            self.par,
            'radiative_transfer_source_photon_rate',
            0.0 / unyt.s,
        )
        if hasattr(source_rate, 'to_value'):
            source_rate_s = source_rate.to_value(1.0 / unyt.s)
        else:
            source_rate_s = float(source_rate)
        seconds_to_myr = 1.0
        history = self._initial_static_history(
            include_thermal_history=include_thermal_history
        )
        self._append_static_history(
            history,
            state,
            ngamma,
            time_s,
            recombined_photons,
            source_rate_s,
            seconds_to_myr,
        )
        step = 0
        rt_updates = 1
        while time_s < final_time_s:
            remaining_s, dtmax_step_s = self._static_step_limit_seconds(
                time_s,
                final_time_s,
                dtmax_s,
                reference_time_s,
                history,
            )
            dt_s, thermal_rate = rtc.get_timestep(
                state,
                ngamma,
                self.par,
                remaining_s,
                dtmax_step_s,
            )
            recombined_photons += self._advance_source_thermochemistry_state(
                state,
                ngamma,
                dt_s,
                thermal_rate,
            )
            time_s += dt_s
            step += 1
            updated_ngamma, updates = self._refresh_static_photon_density(
                state,
                step,
                time_s,
                final_time_s
            )
            if updated_ngamma is not None:
                ngamma = updated_ngamma
                rt_updates += updates
            self._store_static_reference_snapshot(
                history,
                state,
                time_s,
                reference_time_s,
            )
            self._append_static_history(
                history,
                state,
                ngamma,
                time_s,
                recombined_photons,
                source_rate_s,
                seconds_to_myr,
            )

        self._finish_static_thermochemistry(state, time_s)
        history['chemistry_steps'] = step
        history['evolution_steps'] = step
        history['radiative_transfer_updates'] = rt_updates
        return history

    def Run(
        self,
        outputtime=0,
        mode="hydro_sources",
        advect_chemistry=True,
        stop_condition=None,
        step_backend=None,
        step_backend_kwargs=None,
    ):
        """Run the simulation loop and write periodic HDF5 outputs."""
        self.WriteUsedParameters()
        if getattr(self.par, 'outputtimefilename', None):
            rio.run_with_output_times(
                self,
                outputtime=outputtime,
                mode=mode,
                advect_chemistry=advect_chemistry,
                stop_condition=stop_condition,
                step_backend=step_backend,
                step_backend_kwargs=step_backend_kwargs,
            )
            return
        # Fixed-cadence output path: advance to `timesim` and write snapshots
        # whenever `outtime` reaches `outdeltatime`.
        print("--- Initization finished. Start running ... ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        rio.write_numbered_hdf5(self, 0)
        self.Evolve(
            final_time=self.par.timesim,
            mode=mode,
            advect_chemistry=advect_chemistry,
            output_callback=rio.hdf5_output_callback(
                self,
                outputtime=outputtime,
            ),
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )
        if stop_condition is not None:
            self.fluid.SetTemperature()
            rio.write_numbered_hdf5(self, 0)
        print("--- Simulation finished. ---") 
        print("--- %s seconds ---" % (time.time() - start_time))

    def RunAll(
        self,
        outputtime=0,
        mode="hydro_sources",
        advect_chemistry=True,
        stop_condition=None,
        step_backend=None,
        step_backend_kwargs=None,
    ):
        """Run the full workflow from initial-condition read through outputs."""
        self.Callreadhdf5()
        self.SetMesh()
        self.SetFluid()
        self.SetInitFluid()
        self.Run(
            outputtime=outputtime,
            mode=mode,
            advect_chemistry=advect_chemistry,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )

    def checkparams(self):
        """Validate dimensional consistency for selected parameters."""
        print("--- Check parameters ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        ru.CheckDimension(self.par.boxsize,1.0*unyt.pc)
        ru.CheckDimension(self.par.gamma,1.0) 



        
