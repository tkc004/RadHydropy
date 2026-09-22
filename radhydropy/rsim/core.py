# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""High-level simulation runner."""

import copy
import time
from pathlib import Path
from typing import Any

import unyt

import radhydropy.io as rio
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.params import Par
from radhydropy.solver import Solver


class Rsim:
    """Coordinate parameters, mesh, fluid state, solver, and output."""

    start_time: Any
    dark_matter: Any

    def _initialize_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        from .state import _initialize_runtime_state  # noqa: PLC0415

        return _initialize_runtime_state(self, *args, **kwargs)

    def initialize_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize runtime state through the public simulation API."""
        return self._initialize_runtime_state(*args, **kwargs)

    def __init__(self, params: Any) -> None:
        """Create a simulation from a run-parameter dictionary."""
        self._start_time = time.time()
        self.fluid = Fluid()
        self.mesh = Mesh()
        self.par = Par(params)
        self.solver = Solver()
        self.initialize_runtime_state()
        self.fluid.eos = EOS(
            self.par.hydrodynamics.eos_type,
            self.par.hydrodynamics.gamma,
            self.par.units.CodeUnits,
        )

    @classmethod
    def FromComponents(  # noqa: N802
        cls, par: Any, mesh: Any, fluid: Any, solver: Any = None,
    ) -> Any:
        """Create a runner from already-initialized objects."""
        sim = cls.__new__(cls)
        sim.par = par
        sim.mesh = mesh
        sim.fluid = fluid
        sim.solver = solver if solver is not None else Solver()
        sim.start_time = time.time()
        sim.initialize_runtime_state()
        return sim

    def Callreadhdf5(self) -> Any:  # noqa: N802
        from .initialization import Callreadhdf5  # noqa: PLC0415

        return Callreadhdf5(self)

    def SetMesh(self) -> Any:  # noqa: N802
        from .initialization import SetMesh  # noqa: PLC0415

        return SetMesh(self)

    def SetFluid(self) -> Any:  # noqa: N802
        from .initialization import SetFluid  # noqa: PLC0415

        return SetFluid(self)

    def SetInitFluid(self) -> Any:  # noqa: N802
        from .initialization import SetInitFluid  # noqa: PLC0415

        return SetInitFluid(self)

    def ConvertParametersToCodeUnits(self) -> Any:  # noqa: N802
        from .initialization import ConvertParametersToCodeUnits  # noqa: PLC0415

        return ConvertParametersToCodeUnits(self)

    def _require_code_units(self) -> Any:
        from .initialization import _require_code_units  # noqa: PLC0415

        return _require_code_units(self)

    def require_code_units(self) -> Any:
        """Return the configured code-unit system."""
        return self._require_code_units()

    def WriteUsedParameters(self, filename: str = "used_parameters.yaml") -> Any:  # noqa: N802
        """Write the active runtime parameters to a text file in the CWD."""
        return rio.write_used_parameters(Path.cwd() / filename, self.par)

    def GetStepTime(self, dt: Any = None, final_time: Any = None) -> Any:  # noqa: N802
        from .stepping import GetStepTime  # noqa: PLC0415

        return GetStepTime(self, dt=dt, final_time=final_time)

    def PrepareConservedStep(self, fluid: Any = None) -> Any:  # noqa: N802
        from .stepping import PrepareConservedStep  # noqa: PLC0415

        return PrepareConservedStep(self, fluid=fluid)

    def AdvanceHydroFluxes(self, dt: Any, fluid: Any = None) -> Any:  # noqa: N802
        from .stepping import AdvanceHydroFluxes  # noqa: PLC0415

        return AdvanceHydroFluxes(self, dt, fluid=fluid)

    def AdvectChemistryScalars(  # noqa: N802
        self, dt: Any, old_mass: Any, mass_flux: Any, fluid: Any = None,
    ) -> Any:
        from .sources import AdvectChemistryScalars  # noqa: PLC0415

        return AdvectChemistryScalars(self, dt, old_mass, mass_flux, fluid=fluid)

    def UpdateThermochemistryPrimitiveState(  # noqa: N802
        self, *, update_pressure: bool = True, fluid: Any = None,
    ) -> Any:
        from .sources import UpdateThermochemistryPrimitiveState  # noqa: PLC0415

        return UpdateThermochemistryPrimitiveState(
            self,
            update_pressure=update_pressure,
            fluid=fluid,
        )

    def _sync_hydro_state(self, fluid: Any = None) -> Any:
        from .stepping import _sync_hydro_state  # noqa: PLC0415

        return _sync_hydro_state(self, fluid=fluid)

    def sync_hydro_state(self, fluid: Any = None) -> Any:
        """Refresh hydro primitive and conserved state after an update."""
        return self._sync_hydro_state(fluid=fluid)

    def FinalizeHydroStep(  # noqa: N802
        self,
        dt: Any,
        old_mass: Any,
        mass_flux: Any,
        *,
        advect_chemistry: bool = True,
        fluid: Any = None,
        temperature_before: Any = None,
        gravity_dt: Any = None,
        apply_gravity: bool = True,
    ) -> Any:
        from .sources import FinalizeHydroStep  # noqa: PLC0415

        return FinalizeHydroStep(
            self,
            dt,
            old_mass,
            mass_flux,
            advect_chemistry=advect_chemistry,
            fluid=fluid,
            temperature_before=temperature_before,
            gravity_dt=gravity_dt,
            apply_gravity=apply_gravity,
        )

    def ApplyThermochemistrySources(self, dt: Any) -> Any:  # noqa: N802
        from .sources import ApplyThermochemistrySources  # noqa: PLC0415

        return ApplyThermochemistrySources(self, dt)

    def _synchronize_thermochemistry_internal_energy(self) -> Any:
        from .sources import _synchronize_thermochemistry_internal_energy  # noqa: PLC0415

        return _synchronize_thermochemistry_internal_energy(self)

    def synchronize_thermochemistry_internal_energy(self) -> Any:
        """Synchronize thermochemistry internal energy through the public API."""
        return self._synchronize_thermochemistry_internal_energy()

    def clone_fluid(self, fluid: Any = None) -> Any:
        """Return a deep copy of a fluid state."""
        return self._clone_fluid(fluid)

    def _clone_fluid(self, fluid: Any = None) -> Any:
        """Return a deep copy of the supplied fluid state."""
        if fluid is None:
            fluid = self.fluid
        return copy.deepcopy(fluid)

    def _hydro_step_once(
        self,
        dt: Any,
        fluid: Any = None,
        *,
        advect_chemistry: bool = True,
        apply_gravity: bool = True,
    ) -> Any:
        from .stepping import _hydro_step_once  # noqa: PLC0415

        return _hydro_step_once(
            self,
            dt,
            fluid=fluid,
            advect_chemistry=advect_chemistry,
            apply_gravity=apply_gravity,
        )

    def hydro_step_once(
        self,
        dt: Any,
        fluid: Any = None,
        *,
        advect_chemistry: bool = True,
        apply_gravity: bool = True,
    ) -> Any:
        """Advance one hydro step through the public simulation API."""
        return self._hydro_step_once(
            dt,
            fluid,
            advect_chemistry=advect_chemistry,
            apply_gravity=apply_gravity,
        )

    def _hydro_step_ssprk2(
        self, dt: Any, *, advect_chemistry: bool = True, apply_gravity: bool = True,
    ) -> Any:
        from .stepping import _hydro_step_ssprk2  # noqa: PLC0415

        return _hydro_step_ssprk2(
            self,
            dt,
            advect_chemistry=advect_chemistry,
            apply_gravity=apply_gravity,
        )

    def hydro_step_ssprk2(
        self, dt: Any, *, advect_chemistry: bool = True, apply_gravity: bool = True,
    ) -> Any:
        """Advance one SSPRK2 hydro step through the public simulation API."""
        return self._hydro_step_ssprk2(
            dt,
            advect_chemistry=advect_chemistry,
            apply_gravity=apply_gravity,
        )

    def _accumulate_gravity_work(self) -> Any:
        from .sources import _accumulate_gravity_work  # noqa: PLC0415

        return _accumulate_gravity_work(self)

    def accumulate_gravity_work(self) -> Any:
        """Accumulate diagnostics from the most recent gravity update."""
        return self._accumulate_gravity_work()

    def Step(  # noqa: N802
        self,
        dt: Any = None,
        mode: str = "hydro_sources",
        *,
        advect_chemistry: bool = True,
        hydro_integrator: str = "euler",
    ) -> Any:
        from .stepping import Step  # noqa: PLC0415

        return Step(
            self,
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
            hydro_integrator=hydro_integrator,
        )

    def Evolve(  # noqa: N802
        self,
        final_time: Any = None,
        mode: str = "hydro_sources",
        *,
        advect_chemistry: bool = True,
        history_callback: Any = None,
        output_callback: Any = None,
        stop_condition: Any = None,
        step_backend: Any = None,
        step_backend_kwargs: Any = None,
        before_step_callback: Any = None,
    ) -> Any:
        from .evolution import Evolve  # noqa: PLC0415

        return Evolve(
            self,
            final_time=final_time,
            mode=mode,
            advect_chemistry=advect_chemistry,
            before_step_callback=before_step_callback,
            history_callback=history_callback,
            output_callback=output_callback,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )

    def _static_front_radius_from_state(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _static_front_radius_from_state  # noqa: PLC0415

        return _static_front_radius_from_state(self, *args, **kwargs)

    def static_front_radius_from_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._static_front_radius_from_state(*args, **kwargs)

    def _append_static_history(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _append_static_history  # noqa: PLC0415

        return _append_static_history(self, *args, **kwargs)

    def append_static_history(self, *args: Any, **kwargs: Any) -> Any:
        return self._append_static_history(*args, **kwargs)

    def _snapshot_static_state(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _snapshot_static_state  # noqa: PLC0415

        return _snapshot_static_state(self, *args, **kwargs)

    def snapshot_static_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._snapshot_static_state(*args, **kwargs)

    def _initial_static_history(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _initial_static_history  # noqa: PLC0415

        return _initial_static_history(self, *args, **kwargs)

    def initial_static_history(self, *args: Any, **kwargs: Any) -> Any:
        return self._initial_static_history(*args, **kwargs)

    def _static_reference_time_seconds(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _static_reference_time_seconds  # noqa: PLC0415

        return _static_reference_time_seconds(self, *args, **kwargs)

    def static_reference_time_seconds(self, *args: Any, **kwargs: Any) -> Any:
        return self._static_reference_time_seconds(*args, **kwargs)

    def _static_step_limit_seconds(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _static_step_limit_seconds  # noqa: PLC0415

        return _static_step_limit_seconds(self, *args, **kwargs)

    def static_step_limit_seconds(self, *args: Any, **kwargs: Any) -> Any:
        return self._static_step_limit_seconds(*args, **kwargs)

    def _static_recombination_rate(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _static_recombination_rate  # noqa: PLC0415

        return _static_recombination_rate(self, *args, **kwargs)

    def static_recombination_rate(self, *args: Any, **kwargs: Any) -> Any:
        return self._static_recombination_rate(*args, **kwargs)

    def _apply_static_thermal_update(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _apply_static_thermal_update  # noqa: PLC0415

        return _apply_static_thermal_update(self, *args, **kwargs)

    def apply_static_thermal_update(self, *args: Any, **kwargs: Any) -> Any:
        return self._apply_static_thermal_update(*args, **kwargs)

    def _advance_source_thermochemistry_state(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _advance_source_thermochemistry_state  # noqa: PLC0415

        return _advance_source_thermochemistry_state(self, *args, **kwargs)

    def advance_source_thermochemistry_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._advance_source_thermochemistry_state(*args, **kwargs)

    def _refresh_static_photon_density(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _refresh_static_photon_density  # noqa: PLC0415

        return _refresh_static_photon_density(self, *args, **kwargs)

    def refresh_static_photon_density(self, *args: Any, **kwargs: Any) -> Any:
        return self._refresh_static_photon_density(*args, **kwargs)

    def _store_static_reference_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _store_static_reference_snapshot  # noqa: PLC0415

        return _store_static_reference_snapshot(self, *args, **kwargs)

    def store_static_reference_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return self._store_static_reference_snapshot(*args, **kwargs)

    def _finish_static_thermochemistry(self, *args: Any, **kwargs: Any) -> Any:
        from .static_thermochemistry import _finish_static_thermochemistry  # noqa: PLC0415

        return _finish_static_thermochemistry(self, *args, **kwargs)

    def finish_static_thermochemistry(self, *args: Any, **kwargs: Any) -> Any:
        return self._finish_static_thermochemistry(*args, **kwargs)

    def EvolveStaticThermochemistry(  # noqa: N802
        self,
        final_time: Any,
        source_timestep: Any,
        *,
        include_thermal_history: bool = False,
        reference_time: Any = None,
    ) -> Any:
        from .static_thermochemistry import EvolveStaticThermochemistry  # noqa: PLC0415

        return EvolveStaticThermochemistry(
            self,
            final_time,
            source_timestep,
            include_thermal_history=include_thermal_history,
            reference_time=reference_time,
        )

    def Run(  # noqa: N802
        self,
        outputtime: Any = 0,
        mode: str = "hydro_sources",
        *,
        advect_chemistry: bool = True,
        stop_condition: Any = None,
        step_backend: Any = None,
        step_backend_kwargs: Any = None,
        before_step_callback: Any = None,
        history_callback: Any = None,
        snapshot_callback: Any = None,
    ) -> Any:
        from .evolution import Run  # noqa: PLC0415

        return Run(
            self,
            outputtime=outputtime,
            mode=mode,
            advect_chemistry=advect_chemistry,
            before_step_callback=before_step_callback,
            history_callback=history_callback,
            snapshot_callback=snapshot_callback,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )

    def RunAll(  # noqa: N802
        self,
        outputtime: Any = 0,
        mode: str = "hydro_sources",
        *,
        advect_chemistry: bool = True,
        stop_condition: Any = None,
        step_backend: Any = None,
        step_backend_kwargs: Any = None,
        before_step_callback: Any = None,
        history_callback: Any = None,
        snapshot_callback: Any = None,
    ) -> Any:
        from .evolution import RunAll  # noqa: PLC0415

        return RunAll(
            self,
            outputtime=outputtime,
            mode=mode,
            advect_chemistry=advect_chemistry,
            before_step_callback=before_step_callback,
            history_callback=history_callback,
            snapshot_callback=snapshot_callback,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )

    def checkparams(self) -> None:
        """Validate dimensional consistency for selected parameters."""
        ru.CheckDimension(self.par.simulation.box_size_proper_code, 1.0 * unyt.pc)
        ru.CheckDimension(
            self.par.hydrodynamics.gamma,
            1.0,
        )
