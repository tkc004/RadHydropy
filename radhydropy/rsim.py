"""High-level simulation runner."""

import radhydropy.utils as ru
import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.params import Par
from radhydropy.solver import Solver
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
        self.fluid.eos = EOS(self.par.EOStype,self.par.gamma)

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
        self.solver.SetBoundary(self.mesh,self.fluid,self.par)
        self.solver.SetConserved(self.mesh,self.fluid)
        self.solver.ApplyRadiativeTransfer(self.mesh,self.fluid,self.par)

    def RunOneStep(self):
        """Advance the simulation by one timestep and return that timestep."""
        dt = self.solver.GetTimeStep(self.mesh,self.fluid, self.par)
        self.solver.SetBoundary(self.mesh,self.fluid,self.par)
        self.solver.SetConserved(self.mesh,self.fluid)
        self.solver.SetInterFaceFlux(self.mesh,self.fluid,self.par.boundcond,order=self.par.order)
        self.solver.AddFluxes(dt,self.mesh,self.fluid,self.par.boundcond)
        self.solver.SetPrimitive(self.mesh,self.fluid)
        self.solver.ApplyRadiativeTransfer(self.mesh,self.fluid,self.par)
        self.solver.ApplyThermochemistry(dt,self.mesh,self.fluid,self.par)
        self.solver.SetPrimitive(self.mesh,self.fluid)
        if getattr(self.par, 'hydrogen_chemistry', False):
            self.fluid.SetTemperature()
        return dt

    def RunHydroStep(self, dt=None, advect_chemistry=True):
        """Advance one hydrodynamic step, optionally advecting chemistry scalars."""
        if dt is None:
            dt = self.solver.GetTimeStep(self.mesh, self.fluid, self.par)
        self.solver.SetBoundary(self.mesh, self.fluid, self.par)
        self.solver.SetConserved(self.mesh, self.fluid)
        old_mass = self.fluid.Mass.copy()
        self.solver.SetInterFaceFlux(
            self.mesh,
            self.fluid,
            self.par.boundcond,
            order=self.par.order,
        )
        mass_flux = self.fluid.Mass.flux.copy()
        self.solver.AddFluxes(dt, self.mesh, self.fluid, self.par.boundcond)
        if advect_chemistry:
            self.solver.AdvectIonizationFraction(
                dt,
                self.mesh,
                self.fluid,
                self.par,
                old_mass,
                mass_flux,
            )
        self.solver.SetPrimitive(self.mesh, self.fluid)
        if getattr(self.par, 'hydrogen_chemistry', False):
            if getattr(self.par, 'hydrogen_update_mu', False):
                self.fluid.SetHydrogenMu(
                    hydrogen_mass_fraction=getattr(
                        self.par,
                        'hydrogen_mass_fraction',
                        1.0,
                    )
                )
            self.fluid.SetTemperature()
            self.fluid.SetPressure()
        self.solver.SetConserved(self.mesh, self.fluid)
        return dt

    def RunCoupledHydroSourceStep(
        self,
        dt=None,
        fast_thermochemistry=False,
    ):
        """Advance hydrodynamics and then thermo-chemistry source terms."""
        dt = self.RunHydroStep(dt=dt)
        if fast_thermochemistry:
            source_steps = self.solver.ApplyThermochemistryFast(
                dt,
                self.mesh,
                self.fluid,
                self.par,
            )
        else:
            self.solver.ApplyRadiativeTransfer(self.mesh, self.fluid, self.par)
            self.solver.ApplyThermochemistry(dt, self.mesh, self.fluid, self.par)
            source_steps = 1
        self.solver.SetBoundary(self.mesh, self.fluid, self.par)
        self.solver.SetConserved(self.mesh, self.fluid)
        return dt, source_steps

    def EvolveCoupledHydroSources(
        self,
        final_time,
        fast_thermochemistry=False,
        history_callback=None,
    ):
        """Evolve hydro plus source terms to ``final_time`` and return counters."""
        hydro_steps = 0
        source_steps = 0
        if history_callback is not None:
            history_callback(self)
        while self.fluid.time < final_time:
            dt = self.solver.GetTimeStep(self.mesh, self.fluid, self.par)
            if self.fluid.time + dt > final_time:
                dt = final_time - self.fluid.time
            _, step_sources = self.RunCoupledHydroSourceStep(
                dt=dt,
                fast_thermochemistry=fast_thermochemistry,
            )
            hydro_steps += 1
            source_steps += step_sources
            if history_callback is not None:
                history_callback(self)
        return {'hydro_steps': hydro_steps, 'source_steps': source_steps}

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

    def _append_static_history(self, history, state, ngamma, time_s, recombined_photons):
        ionized = 1.0 - state['xHI']
        ionized_atoms = np.sum(ionized * state['nH_cm3'] * state['volume_cm3'])
        volume_photons = np.sum(ngamma * state['volume_cm3'])
        history['time_Myr'].append((time_s * unyt.s).to_value(unyt.Myr))
        history['front_radius_kpc'].append(self._static_front_radius_from_state(state))
        history['injected_photons'].append(
            getattr(
                self.par,
                'radiative_transfer_source_photon_rate',
                0.0 / unyt.s,
            ).to_value(1.0 / unyt.s)
            * time_s
        )
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
            'time_Myr': (time_s * unyt.s).to_value(unyt.Myr),
            'radius_kpc': state['radius_kpc'].copy(),
            'xHI': state['xHI'].copy(),
            'temperature_K': state['temperature_K'].copy(),
        }

    def EvolveStaticThermochemistry(
        self,
        final_time,
        source_timestep,
        include_thermal_history=False,
        reference_time=None,
    ):
        """Evolve fixed-density thermo-chemistry/radiation source terms."""
        state = self.solver.StaticThermochemistryState(self.mesh, self.fluid, self.par)
        ngamma = self.solver.TraceStaticSphericalPhotonDensity(state, self.par)
        recombined_photons = 0.0
        time_s = 0.0
        final_time_s = final_time.to_value(unyt.s)
        dtmax_s = source_timestep.to_value(unyt.s)
        reference_time_s = None
        if reference_time is not None:
            reference_time_s = reference_time.to_value(unyt.s)
        rt_update_interval = max(
            1,
            int(getattr(self.par, 'radiative_transfer_update_interval', 1)),
        )
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
        self._append_static_history(history, state, ngamma, time_s, recombined_photons)
        step = 0
        rt_updates = 1
        while time_s < final_time_s:
            remaining_s = final_time_s - time_s
            dtmax_step_s = min(dtmax_s, remaining_s)
            if (
                reference_time_s is not None
                and 'reference_snapshot' not in history
                and time_s < reference_time_s <= time_s + dtmax_step_s
            ):
                dtmax_step_s = reference_time_s - time_s
            dt_s, thermal_rate = self.solver.GetStaticThermochemistryTimeStep(
                state,
                ngamma,
                self.par,
                remaining_s,
                dtmax_step_s,
            )
            ionized_start = 1.0 - state['xHI']
            alpha = getattr(self.par, 'hydrogen_alpha_B', None)
            if alpha is None:
                alpha_value = 0.0
            else:
                alpha_value = alpha.to_value(unyt.cm**3 / unyt.s)
            recombination_rate_start = np.sum(
                alpha_value
                * ionized_start**2
                * state['nH_cm3']**2
                * state['volume_cm3']
            )
            if getattr(self.par, 'hydrogen_thermal_coupling', True):
                if thermal_rate is None:
                    thermal_rate = self.solver.StaticThermalRate(
                        state,
                        ngamma,
                        self.par,
                    )
                state['specific_energy_erg_g'] += thermal_rate / state['rho_g_cm3'] * dt_s
                state['specific_energy_erg_g'] = np.maximum(
                    state['specific_energy_erg_g'],
                    1.0e6,
                )
                self.solver.UpdateStaticTemperatureFromEnergy(state, self.par)
            self.solver.StaticIonizationFractionImplicitUpdate(
                state,
                ngamma,
                dt_s,
                self.par,
            )
            if getattr(self.par, 'hydrogen_thermal_coupling', True):
                self.solver.UpdateStaticTemperatureFromEnergy(state, self.par)
            ionized_end = 1.0 - state['xHI']
            recombination_rate_end = np.sum(
                alpha_value
                * ionized_end**2
                * state['nH_cm3']**2
                * state['volume_cm3']
            )
            recombined_photons += (
                0.5 * (recombination_rate_start + recombination_rate_end) * dt_s
            )
            time_s += dt_s
            step += 1
            if step % rt_update_interval == 0 or time_s >= final_time_s:
                ngamma = self.solver.TraceStaticSphericalPhotonDensity(state, self.par)
                rt_updates += 1
            if (
                reference_time_s is not None
                and 'reference_snapshot' not in history
                and time_s >= reference_time_s
            ):
                history['reference_snapshot'] = self._snapshot_static_state(state, time_s)
            self._append_static_history(
                history,
                state,
                ngamma,
                time_s,
                recombined_photons,
            )

        state['ngamma'] = self.solver.TraceStaticSphericalPhotonDensity(state, self.par)
        state['time_s'] = time_s
        self.solver.ApplyStaticThermochemistryState(state, self.fluid, self.par)
        self.solver.SetBoundary(self.mesh, self.fluid, self.par)
        history['chemistry_steps'] = step
        history['evolution_steps'] = step
        history['radiative_transfer_updates'] = rt_updates
        return history

    def Run(self,outputtime=0):
        """Run the simulation loop and write periodic HDF5 outputs."""
        print("--- Initization finished. Start running ... ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        outtime = 0.0 * self.par.timesim 
        outindex = 0
        # write the initial condition
        
        rio.writehdf5(self,self.par.outdir+'/'+self.par.outfileprefix+'_%03d'%outindex+'.hdf5') 
        outtime = 0.0 * self.par.timesim 
        outindex += 1
        while self.fluid.time <self.par.timesim:
            dt = self.RunOneStep()
            if outputtime==1:
                print("time, dt", self.fluid.time, dt)  
            if outtime > self.par.outdeltatime:
                self.fluid.SetTemperature()
                rio.writehdf5(self,self.par.outdir+'/'+self.par.outfileprefix+'_%03d'%outindex+'.hdf5') 
                outtime = 0.0 * self.par.timesim 
                outindex += 1
            else:
                outtime += dt 
        print("--- Simulation finished. ---") 
        print("--- %s seconds ---" % (time.time() - start_time))

    def RunAll(self,outputtime=0):
        """Run the full workflow from initial-condition read through outputs."""
        self.Callreadhdf5()
        self.SetMesh()
        self.SetFluid()
        self.SetInitFluid()
        self.Run(outputtime)

    def checkparams(self):
        """Validate dimensional consistency for selected parameters."""
        print("--- Check parameters ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        ru.CheckDimension(self.par.boxsize,1.0*unyt.pc)
        ru.CheckDimension(self.par.gamma,1.0) 



        
