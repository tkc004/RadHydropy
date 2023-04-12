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
    def __init__(self,params) -> None:
        print("--- Get simulation parameters ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.params = params
        self.fluid = Fluid()
        self.mesh  = Mesh()
        self.par    = Par(params)
        self.solver = Solver()
        self.fluid.eos = EOS(self.par.EOStype,self.par.gamma)
        

    def Callreadhdf5(self):
        print("--- Read Initial Condition ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        rio.readhdf5(self.par, self.mesh, self.fluid, self.par.ICfilename)
        self.checkparams()
        self.fluid.SetFluidTime(self.par.time)
        print("--- Start Initial Time ---")

    def SetMesh(self):
        print("--- Set up the Mesh ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.mesh.SetUpMesh(self.par)


    def SetFluid(self):
        print("--- Set up the fluid ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.fluid.SetUpFluid()
    
    def SetInitFluid(self):
        print("--- Fill up the fluid---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.solver.SetConserved(self.mesh,self.fluid)        

    def RunOneStep(self):
        dt = self.solver.GetTimeStep(self.mesh,self.fluid)
        self.fluid.SetBoundary(self.par.boundcond)
        self.solver.SetConserved(self.mesh,self.fluid)
        self.solver.SetInterFaceFlux(self.mesh,self.fluid)
        self.solver.AddFluxes(dt,self.mesh,self.fluid)
        self.solver.SetPrimitive(self.mesh,self.fluid)
        return dt

    def Run(self,outputtime=0):
        print("--- Initization finished. Start running ... ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        outtime = 0.0 * self.par.timesim 
        outindex = 0
        while self.fluid.time <self.par.timesim:
            dt = self.RunOneStep()
            if outputtime==1:
                print("time, dt", self.fluid.time, dt)  
            if outtime > self.par.outdeltatime:
                rio.writehdf5(self,self.par.outdir+'/'+self.par.outfileprefix+'_%03d'%outindex+'.hdf5') 
                outtime = 0.0 * self.par.timesim 
                outindex += 1
            else:
                outtime += dt 
        print("--- Simulation finished. ---") 
        print("--- %s seconds ---" % (time.time() - start_time))

    def RunAll(self,outputtime=0):
        self.Callreadhdf5()
        self.SetMesh()
        self.SetFluid()
        self.SetInitFluid()
        self.Run(outputtime)

    def checkparams(self):
        print("--- Check parameters ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        ru.CheckDimension(self.par.boxsize,1.0*unyt.pc)
        ru.CheckDimension(self.par.gamma,1.0) 



        

