import numpy as np
import unyt
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.mesh import Mesh


# set up fluid properties

class Fluid():
    # import mesh and EOS information into Fluid
    def __init__(self):
        pass 

    def SetPressure(self):
        self.p = self.eos.CalPressure(self.rho,self.temp,self.mu)
        
    def SetEnergyDensity(self):
        self.eth = self.eos.CalEnergyDensity(self.p)
        
    def SetSoundSpeed(self):
        self.cs = self.eos.CalSoundSpeed(self.p,self.rho)
        
    def SetUpFluid(self):
        self.SetPressure() 

    def SetFluidTime(self, time): 
        self.time = time
    
    def SetBoundary(self,btype:str):
        if btype == 'Periodic':
            self.rho[0] = self.rho[-2]
            self.u[0] = self.u[-2]
            self.p[0] = self.p[-2]
            self.rho[-1] = self.rho[1]
            self.u[-1] = self.u[1]
            self.p[-1] = self.p[1]            
        else:
            print('Boundary condition unknown')           
