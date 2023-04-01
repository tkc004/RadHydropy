import numpy as np
import unyt
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.mesh import Mesh


# set up fluid properties

class Fluid():
    # import mesh and EOS information into Fluid
    def __init__(self, mesh: Mesh, eos: EOS, tini: float, ftype: str):
        Fluid.mesh = mesh
        Fluid.eos = eos
        Fluid.time = tini 
        Fluid.ftype = ftype
    
    def SetPressure(self):
        self.p = self.eos.CalPressure(self.rho,self.temp,self.mu)
        
    def SetEnergyDensity(self):
        self.eth = self.eos.CalEnergyDensity(self.p)
        
    def SetSoundSpeed(self):
        self.cs = self.eos.CalSoundSpeed(self.p,self.rho)
        
    def SetPrimitive(self, verbose=0):
        vol = self.mesh.vol
        self.rho = self.Mass / vol
        self.u = self.Mom / self.Mass
        self.p = (self.Energy/vol-0.5*self.rho*self.u**2)*(self.eos.gamma-1.0)
        self.rho[np.logical_or(self.rho<0.0, np.isnan(self.rho))] = 0.0
        self.p[np.logical_or(self.p<0.0, np.isnan(self.p))] = 0.0            
        if verbose == 1:
            print('self.rho',self.rho)
            print('self.u',self.u)
            print('self.p',self.p)            
        
        
    def SetConserved(self, verbose=0):
        vol = self.mesh.vol
        self.Mass = self.rho * vol
        self.Mom = self.rho * self.u * vol
        self.Energy = (0.5*self.rho*self.u**2 + self.p/(self.eos.gamma-1.0))*vol
        self.Mass[np.logical_or(self.Mass<0.0, np.isnan(self.Mass))] = 0.0
        self.Energy[np.logical_or(self.Energy<0.0, np.isnan(self.Energy))] = 0.0
        if verbose == 1:
            print('self.Mass',self.Mass)
            print('self.Mom',self.Mom)
            print('self.Energy',self.Energy)
        
        
    def SetGradient(self):
        xdelta = self.mesh.xdelta
        coordinate = self.mesh.coordinate
        self.rho.grad = ru.CalGradient(self.rho, xdelta, coordinate)
        self.u.grad   = ru.CalGradient(self.u, xdelta, coordinate)
        self.p.grad   = ru.CalGradient(self.p, xdelta, coordinate)
        
        
    def SetConservedDensityFlux(self):
        gamma = self.eos.gamma
        self.Mass.F, self.Mass.q, self.Mom.F, self.Mom.q, self.Energy.F, self.Energy.q = ru.GetFQ(self.rho,self.u,self.p,self.eos.gamma)
        
    def SetFaceLR(self,order=0):
        #numpy roll Rroll, put the right value to this cell
        Lroll = 1
        Rroll = -1
        if order == 0 or order == 1:
            self.rho.R = self.rho
            self.rho.L = np.roll(self.rho, Lroll)
            self.u.R = self.u
            self.u.L = np.roll(self.u, Lroll)
            self.p.R = self.p
            self.p.L = np.roll(self.p, Lroll)
            if order == 1:
                self.SetGradient()
                self.rho.R.first, self.rho.L.first = ru.extrapolateToFace(self.rho, self.mesh.xbound, self.rho.grad, order=1)
                self.u.R.first, self.u.L.first = ru.extrapolateToFace(self.u, self.mesh.xbound, self.u.grad, order=1)
                self.p.R.first, self.p.L.first = ru.extrapolateToFace(self.p, self.mesh.xbound, self.p.grad, order=1)
        else:
            print('order unknown')
        
        
    
    def SetInitFluid(self, rhoini: float, uini: float, tempini: float, verbose=0):  
        nogrid = self.mesh.nogrid
        if self.ftype=='uniform':
            self.rho = np.ones(nogrid+2) * rhoini
            self.u = np.ones(nogrid+2) * uini
            self.temp = np.ones(nogrid+2) * tempini
            # mean molecular weight
            self.mu = np.ones(nogrid+2) * 1.28 # for primordial neutral gas
        elif self.ftype=='half':
            self.rho = np.ones(nogrid+2) * rhoini
            self.u = np.ones(nogrid+2) * uini
            self.temp = np.ones(nogrid+2) * tempini
            self.rho[np.logical_or(self.mesh.xmesh<0.25*self.mesh.boxsize, self.mesh.xmesh>0.75*self.mesh.boxsize)] *= 0.5
            # mean molecular weight
            self.mu = np.ones(nogrid+2) * 1.28 # for primordial neutral gas  
        elif self.ftype=='gaussian':
            self.rho = (1.0+gaussian(self.mesh.xmesh, 0.5*self.mesh.boxsize, 0.1*self.mesh.boxsize))* rhoini
            self.u = np.ones(nogrid+2) * uini
            self.temp = np.ones(nogrid+2) * tempini
            # mean molecular weight
            self.mu = np.ones(nogrid+2) * 1.28 # for primordial neutral gas  
        elif self.ftype=='sodshock':
            self.mu = np.ones(nogrid+2) * 1.28 # for primordial neutral gas    
            self.rho = np.ones(nogrid+2) * rhoini
            self.u = np.ones(nogrid+2) * uini
            self.temp = np.ones(nogrid+2) * tempini
            self.rho[self.mesh.xmesh>0.5*self.mesh.boxsize] *= 0.1
            self.temp[self.mesh.xmesh>0.5*self.mesh.boxsize] *= 0.1/0.125
            # mean molecular weight
            self.mu = np.ones(nogrid+2) * 1.28 # for primordial neutral gas            
        else:
            print('ftype unknown')
        self.SetPressure()
        if verbose == 1:
            print('self.rho',self.rho)
            print('self.u',self.u)
            print('self.p',self.p)
            
 

        
        
    def SetFluxOnFace(self,order=0):
        #numpy roll Rroll, put the right value to this cell
        Lroll = 1
        Rroll = -1
        Mass_flux_0, Mom_flux_0, Energy_flux_0 = ru.CalFluxFromLR(self.rho.L,self.rho.R,
                                                               self.u.L,self.u.R,
                                                               self.p.L,self.p.R,
                                                               self.eos.gamma,self.cmax)
        if order==0:
            self.Mass.flux, self.Mom.flux, self.Energy.flux = Mass_flux_0, Mom_flux_0, Energy_flux_0
        elif order==1:
            Mass_flux_1, Mom_flux_1, Energy_flux_1 = ru.CalFluxFromLR(self.rho.L.first, self.rho.R.first,
                                                                    self.u.L.first, self.u.R.first,
                                                                    self.p.L.first, self.p.R.first,
                                                                    self.eos.gamma, self.cmax)
            self.SetConservedDensityFlux()
            self.Mass.flux, self.philim_Mass= ru.ApplyFluxLimiter(self.Mass.q,Mass_flux_1,Mass_flux_0)
            self.Mom.flux, self.philim_Mom  = ru.ApplyFluxLimiter(self.Mom.q,Mom_flux_1,Mom_flux_0)
            self.Energy.flux, self.philim_Energy  = ru.ApplyFluxLimiter(self.Energy.q,Energy_flux_1,Energy_flux_0)
            
        else:
            print('order unknown')

        
    def SetInterFaceFlux(self,method='Rusanov',verbose=0, order=0):
        if method=='GLF' or method=='Rusanov':
            #numpy roll Rroll, put the right value to this cell
            Lroll = 1
            Rroll = -1
            if method=='GLF':
                # Global Lax Friedrich scheme
                # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)  
                # simple to implement but very diffusive
                # calculate cmax
                self.cmax = self.mesh.xdelta/np.amin(self.dt)
            elif method=='Rusanov':
                # Local Lax Friedrich schem
                # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)  
                # simple to implement but less diffusive
                self.cmax = np.maximum(self.vsignal, np.roll(self.vsignal,Lroll))
            
            self.SetFaceLR(order=order)
            self.SetFluxOnFace(order=order)
        else:
            print('Interface flux method unknown')
            exit()
        if (verbose==1):
            print('self.Mass.flux',self.Mass.flux)
            print('self.Mom.flux',self.Mom.flux)
            print('self.Energy.flux',self.Energy.flux)
            
            
    def AddFluxes(self, dt: float):
        #numpy roll Rroll, put the right value to this cell
        Lroll = 1
        Rroll = -1
        # Add the interface fluxes to the cells:
        area = self.mesh.area
        vol = self.mesh.vol
        self.Mass += (self.Mass.flux*area - np.roll(self.Mass.flux*area,Rroll))*dt
        self.Mom  += (self.Mom.flux*area - np.roll(self.Mom.flux*area,Rroll))*dt
        self.Energy  += (self.Energy.flux*area - np.roll(self.Energy.flux*area,Rroll))*dt
        self.time += dt
        
    
    def GetTimeStep(self, CFL=0.1, vsmin = 0.001*unyt.km/unyt.s):
        self.SetSoundSpeed()
        vsignal = np.absolute(self.u) + self.cs
        #if np.any(vsignal==0) or np.any(np.isnan(vsignal)):
        #    print('vsignal vanished')    
        #else:
        vsignal[np.logical_or(vsignal==0.0,np.isnan(vsignal))] = vsmin
        dx_c = self.mesh.xdelta / vsignal
        self.dt =  CFL * np.amin(dx_c)
        self.vsignal = vsignal
        dt = np.amin(self.dt)
        #print('vsignal', vsignal)
        return dt
     
    
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
