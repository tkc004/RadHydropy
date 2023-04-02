import matplotlib.pyplot as plt
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
import unyt
import numpy as np
import time
start_time = time.time()


class Rsim():
    def __init__(self,params) -> None:
        print("--- Get simulation parameters ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.params = params
        self.checkparams()

    def SetMesh(self):
        print("--- Set up the boundary mesh ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.mesh = Mesh(self.params['boxsize'],self.params['ngrid'],self.params['coordinate'])

    def SetEOS(self):
        print("--- Set up the equation of state ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.eos = EOS(self.params['EOStype'],self.params['gamma'])

    def SetFluid(self):
        print("--- Set up the empty fluid ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.fluid = Fluid(self.mesh,self.eos,self.params['tini'],self.params['ftype'])
    
    def SetInitFluid(self):
        print("--- Fill up the fluid---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        self.fluid.SetInitFluid(self.params['rhoini'],self.params['vini'],self.params['tempini'],self.params['verbose'])
        self.fluid.SetConserved()        

    def RunOneStep(self):
        dt = self.fluid.GetTimeStep()
        self.fluid.SetBoundary(self.params['boundcond'])
        self.fluid.SetConserved()
        self.fluid.SetInterFaceFlux()
        self.fluid.AddFluxes(dt)
        self.fluid.SetPrimitive()

    def Run(self,outputtime=0):
        print("--- Initization finished. Start running ... ---") 
        print("--- %s seconds ---" % (time.time() - start_time))
        while self.fluid.time <self.params['timesim']:
            self.RunOneStep()
            if outputtime==1:
                print("time, dt", self.fluid.time, dt)  
        print("--- Simulation finished. ---") 
        print("--- %s seconds ---" % (time.time() - start_time))

    def RunAll(self):
        self.SetMesh()
        self.SetEOS()
        self.SetFluid()
        self.SetInitFluid()
        self.Run() 

    def checkparams(self):
        print("--- Check parameters ---")
        print("--- %s seconds ---" % (time.time() - start_time))
        ru.CheckDimension(self.params['boxsize'],1.0*unyt.pc)
        ru.CheckDimension(self.params['tini'],1.0*unyt.yr)
        ru.CheckDimension(self.params['vini'],1.0*unyt.pc/unyt.yr)
        ru.CheckDimension(self.params['rhoini'],1.0*unyt.g/unyt.cm**3)
        ru.CheckDimension(self.params['tempini'],1.0*unyt.K)
        ru.CheckDimension(self.params['gamma'],1.0) 



        

