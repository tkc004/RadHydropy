"""High-level simulation runner."""

import copy
import radhydropy.io as rio
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.params import Par
from radhydropy.solver import Solver
from pathlib import Path
import unyt
import time

class Rsim():
    """Coordinate parameters, mesh, fluid state, solver, and output."""

    def _initialize_runtime_state(self, *args, **kwargs):
        from .state import _initialize_runtime_state

        return _initialize_runtime_state(self, *args, **kwargs)

    def __init__(self,params) -> None:
        """Create a simulation from a run-parameter dictionary."""
        self._start_time = time.time()
        print("--- Get simulation parameters ---")
        print("--- %s seconds ---" % (
            time.time() - getattr(self, "_start_time", time.time())
        ))
        self.fluid = Fluid()
        self.mesh  = Mesh()
        self.par    = Par(params)
        self.solver = Solver()
        self._initialize_runtime_state()
        self.fluid.eos = EOS(
            self.par.hydrodynamics.eos_type,
            self.par.hydrodynamics.gamma,
            self.par.units.CodeUnits,
        )

    @classmethod
    def FromComponents(cls, par, mesh, fluid, solver=None):
        """Create a runner from already-initialized objects."""
        sim = cls.__new__(cls)
        sim.par = par
        sim.mesh = mesh
        sim.fluid = fluid
        sim.solver = solver if solver is not None else Solver()
        sim._start_time = time.time()
        sim._initialize_runtime_state()
        return sim
        

    def Callreadhdf5(self):
        from .initialization import Callreadhdf5

        return Callreadhdf5(self)

    def SetMesh(self):
        from .initialization import SetMesh

        return SetMesh(self)


    def SetFluid(self):
        from .initialization import SetFluid

        return SetFluid(self)
    
    def SetInitFluid(self):
        from .initialization import SetInitFluid

        return SetInitFluid(self)

    def ConvertParametersToCodeUnits(self):
        from .initialization import ConvertParametersToCodeUnits

        return ConvertParametersToCodeUnits(self)

    def _require_code_units(self):
        from .initialization import _require_code_units

        return _require_code_units(self)

    def WriteUsedParameters(self, filename="used_parameters.yaml"):
        """Write the active runtime parameters to a text file in the CWD."""
        return rio.write_used_parameters(Path.cwd() / filename, self.par)

    def GetStepTime(self, dt=None, final_time=None):
        from .stepping import GetStepTime

        return GetStepTime(self, dt=dt, final_time=final_time)

    def PrepareConservedStep(self, fluid=None):
        from .stepping import PrepareConservedStep

        return PrepareConservedStep(self, fluid=fluid)

    def AdvanceHydroFluxes(self, dt, fluid=None):
        from .stepping import AdvanceHydroFluxes

        return AdvanceHydroFluxes(self, dt, fluid=fluid)

    def AdvectChemistryScalars(self, dt, old_mass, mass_flux, fluid=None):
        from .sources import AdvectChemistryScalars

        return AdvectChemistryScalars(self, dt, old_mass, mass_flux, fluid=fluid)

    def UpdateThermochemistryPrimitiveState(self, update_pressure=True, fluid=None):
        from .sources import UpdateThermochemistryPrimitiveState

        return UpdateThermochemistryPrimitiveState(self, update_pressure=update_pressure, fluid=fluid)

    def _sync_hydro_state(self, fluid=None):
        from .stepping import _sync_hydro_state

        return _sync_hydro_state(self, fluid=fluid)

    def FinalizeHydroStep(
        self,
        dt,
        old_mass,
        mass_flux,
        advect_chemistry=True,
        fluid=None,
        temperature_before=None,
        gravity_dt=None,
        apply_gravity=True,
    ):
        from .sources import FinalizeHydroStep

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

    def ApplyThermochemistrySources(self, dt):
        from .sources import ApplyThermochemistrySources

        return ApplyThermochemistrySources(self, dt)

    def _synchronize_thermochemistry_internal_energy(self):
        from .sources import _synchronize_thermochemistry_internal_energy

        return _synchronize_thermochemistry_internal_energy(self)

    def _clone_fluid(self, fluid=None):
        """Return a deep copy of the supplied fluid state."""
        if fluid is None:
            fluid = self.fluid
        return copy.deepcopy(fluid)

    def _hydro_step_once(self, dt, fluid=None, advect_chemistry=True, apply_gravity=True):
        from .stepping import _hydro_step_once

        return _hydro_step_once(self, dt, fluid=fluid, advect_chemistry=advect_chemistry, apply_gravity=apply_gravity)

    def _hydro_step_ssprk2(self, dt, advect_chemistry=True, apply_gravity=True):
        from .stepping import _hydro_step_ssprk2

        return _hydro_step_ssprk2(self, dt, advect_chemistry=advect_chemistry, apply_gravity=apply_gravity)

    def _accumulate_gravity_work(self):
        from .sources import _accumulate_gravity_work

        return _accumulate_gravity_work(self)

    def Step(
        self,
        dt=None,
        mode="hydro_sources",
        advect_chemistry=True,
        hydro_integrator="euler",
    ):
        from .stepping import Step

        return Step(
            self,
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
            hydro_integrator=hydro_integrator,
        )

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
        from .evolution import Evolve

        return Evolve(
            self,
            final_time=final_time,
            mode=mode,
            advect_chemistry=advect_chemistry,
            history_callback=history_callback,
            output_callback=output_callback,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )

    def _static_front_radius_from_state(self, *args, **kwargs):
        from .static_thermochemistry import _static_front_radius_from_state

        return _static_front_radius_from_state(self, *args, **kwargs)

    def _append_static_history(self, *args, **kwargs):
        from .static_thermochemistry import _append_static_history

        return _append_static_history(self, *args, **kwargs)

    def _snapshot_static_state(self, *args, **kwargs):
        from .static_thermochemistry import _snapshot_static_state

        return _snapshot_static_state(self, *args, **kwargs)

    def _initial_static_history(self, *args, **kwargs):
        from .static_thermochemistry import _initial_static_history

        return _initial_static_history(self, *args, **kwargs)

    def _static_reference_time_seconds(self, *args, **kwargs):
        from .static_thermochemistry import _static_reference_time_seconds

        return _static_reference_time_seconds(self, *args, **kwargs)

    def _static_step_limit_seconds(self, *args, **kwargs):
        from .static_thermochemistry import _static_step_limit_seconds

        return _static_step_limit_seconds(self, *args, **kwargs)

    def _static_recombination_rate(self, *args, **kwargs):
        from .static_thermochemistry import _static_recombination_rate

        return _static_recombination_rate(self, *args, **kwargs)

    def _apply_static_thermal_update(self, *args, **kwargs):
        from .static_thermochemistry import _apply_static_thermal_update

        return _apply_static_thermal_update(self, *args, **kwargs)

    def _advance_source_thermochemistry_state(self, *args, **kwargs):
        from .static_thermochemistry import _advance_source_thermochemistry_state

        return _advance_source_thermochemistry_state(self, *args, **kwargs)

    def _refresh_static_photon_density(self, *args, **kwargs):
        from .static_thermochemistry import _refresh_static_photon_density

        return _refresh_static_photon_density(self, *args, **kwargs)

    def _store_static_reference_snapshot(self, *args, **kwargs):
        from .static_thermochemistry import _store_static_reference_snapshot

        return _store_static_reference_snapshot(self, *args, **kwargs)

    def _finish_static_thermochemistry(self, *args, **kwargs):
        from .static_thermochemistry import _finish_static_thermochemistry

        return _finish_static_thermochemistry(self, *args, **kwargs)

    def EvolveStaticThermochemistry(
        self,
        final_time,
        source_timestep,
        include_thermal_history=False,
        reference_time=None,
    ):
        from .static_thermochemistry import EvolveStaticThermochemistry

        return EvolveStaticThermochemistry(
            self,
            final_time,
            source_timestep,
            include_thermal_history=include_thermal_history,
            reference_time=reference_time,
        )

    def Run(
        self,
        outputtime=0,
        mode="hydro_sources",
        advect_chemistry=True,
        stop_condition=None,
        step_backend=None,
        step_backend_kwargs=None,
    ):
        from .evolution import Run

        return Run(
            self,
            outputtime=outputtime,
            mode=mode,
            advect_chemistry=advect_chemistry,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )

    def RunAll(
        self,
        outputtime=0,
        mode="hydro_sources",
        advect_chemistry=True,
        stop_condition=None,
        step_backend=None,
        step_backend_kwargs=None,
    ):
        from .evolution import RunAll

        return RunAll(
            self,
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
        print("--- %s seconds ---" % (
            time.time() - getattr(self, "_start_time", time.time())
        ))
        ru.CheckDimension(self.par.simulation.box_size, 1.0 * unyt.pc)
        ru.CheckDimension(
            self.par.hydrodynamics.gamma,
            1.0,
        )



        
