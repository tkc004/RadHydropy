import numpy as np
import unyt

# set up the equation of state
class EOS:
    def __init__(self,EOStype: str,gamma=5./3.):
        self.EOStype = EOStype
        self.gamma = gamma
        if gamma==1.0:
            print('gamma cannot be equal to 1')
            exit()
        if ((self.EOStype != 'polytropic') and (self.EOStype != 'isothermal')):
            print("EOS not recognized: only polytropic or isothermal")
        
    def CalPressure(self,rho,temp,mu):
        pressure = rho / (mu * unyt.mp) * unyt.kb * temp
        return pressure
    
    def CalEnergyDensity(self,pressure):
        energydensity = pressure / (self.gamma-1.0)
        return energydensity
    
    def CalSoundSpeed(self,pressure,rho):
        soundspeed = np.sqrt(self.gamma * pressure / rho)
        return soundspeed
        
    def PressuretoDensity(self,Pressure):
        if self.EOStype == 'polytropic':
            Density = Pressure**(1.0/self.gamma)
            return Density
        elif self.EOStype == 'isothermal':
            Density = Pressure            
            return Density
            
    def DensitytoPressure(self,Density):
        if self.EOStype == 'polytropic':
            Pressure = Density**self.gamma
            return Pressure
        elif self.EOStype == 'isothermal':
            Pressure = Density
            return Pressure  