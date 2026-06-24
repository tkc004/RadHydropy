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
        
    def SetUpFluid(self, par):
        # check if the required attributes exist
        attrlist = ['rho','temp','mu','vel'] 
        valuelist = [1.0, 0.0, 1.0, 0.0]
        for attr in attrlist:
            if not hasattr(self, attr):
                raise Exception("%s does not exist in fluid; quitting."%attr)
            

        #add ghost cells:
        noghost = par.noghost
        for iattr, attr in enumerate(attrlist): 
            quan = getattr(self, attr)
            #print('attr,qaun',attr,quan)
            try:
                units = quan.units
            except AttributeError:
                units = 1.0
            ghost = np.ones(noghost) * units * valuelist[iattr] 
            quan = unyt.uconcatenate((ghost,quan,ghost))
            setattr(self, attr, quan)
        self.SetPressure() 

    def SetTemperature(self):
        self.temp = ru.CalTemperature(self.rho,self.pre,self.mu) 

    def SetFluidTime(self, time): 
        self.time = time       
