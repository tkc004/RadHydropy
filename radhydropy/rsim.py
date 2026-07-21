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
        self.solver.AddHydrogenSources(dt,self.mesh,self.fluid,self.par)
        self.solver.SetPrimitive(self.mesh,self.fluid)
        if getattr(self.par, 'hydrogen_chemistry', False):
            self.fluid.SetTemperature()
        return dt

    def RunHydroStep(self, dt=None, advect_hydrogen=True):
        """Advance one hydrodynamic step, optionally advecting hydrogen scalars."""
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
        if advect_hydrogen:
            self.solver.AdvectHydrogenNeutralFraction(
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

    def RunCoupledHydroSourceStep(self, dt=None, fast_hydrogen_sources=False):
        """Advance hydrodynamics and then hydrogen source terms for one step."""
        dt = self.RunHydroStep(dt=dt)
        if fast_hydrogen_sources:
            source_steps = self.solver.AddHydrogenSourcesFast(
                dt,
                self.mesh,
                self.fluid,
                self.par,
            )
        else:
            self.solver.ApplyRadiativeTransfer(self.mesh, self.fluid, self.par)
            self.solver.AddHydrogenSources(dt, self.mesh, self.fluid, self.par)
            source_steps = 1
        self.solver.SetBoundary(self.mesh, self.fluid, self.par)
        self.solver.SetConserved(self.mesh, self.fluid)
        return dt, source_steps

    def EvolveCoupledHydroSources(
        self,
        final_time,
        fast_hydrogen_sources=False,
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
                fast_hydrogen_sources=fast_hydrogen_sources,
            )
            hydro_steps += 1
            source_steps += step_sources
            if history_callback is not None:
                history_callback(self)
        return {'hydro_steps': hydro_steps, 'source_steps': source_steps}

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



        
