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
        self.pre = ru.CalPressure(self.rho,self.temp,self.mu)
        
    def SetEnergyDensity(self):
        self.eth = ru.CalEnergyDensity(self.pre,self.eos.gamma)
        
    def SetSoundSpeed(self):
        self.cs = ru.CalSoundSpeed(self.pre,self.rho,self.eos.gamma)
        
    def SetUpFluid(self):
        # check if the required attributes exist
        attrlist = ['rho','temp','mu','vel'] 
        for attr in attrlist:
            if not hasattr(self, attr):
                raise Exception("%s does not exist in fluid; quitting."%attr)
        self.SetPressure() 

    def SetFluidTime(self, time): 
        self.time = time
    
    def SetBoundary(self,btype:str):
        if btype == 'Periodic':
            self.rho[0] = self.rho[-2]
            self.vel[0] = self.vel[-2]
            self.pre[0] = self.pre[-2]
            self.rho[-1] = self.rho[1]
            self.vel[-1] = self.vel[1]
            self.pre[-1] = self.pre[1]            
        else:
            print('Boundary condition unknown')           
