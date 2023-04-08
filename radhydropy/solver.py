import radhydropy.utils as ru
import unyt
import numpy as np

class Solver():
    def __init__(self) -> None:
        # should add information like
        # limiter, first order, what method
        # and time
        pass

    def SetPrimitive(self, mesh, fluid, verbose=0):
        vol = mesh.vol
        fluid.rho = fluid.Mass / vol
        fluid.vel = fluid.Mom / fluid.Mass
        fluid.pre = (fluid.Energy/vol-0.5*fluid.rho*fluid.vel**2)*(fluid.eos.gamma-1.0)
        fluid.rho[np.logical_or(fluid.rho<0.0, np.isnan(fluid.rho))] = 0.0
        fluid.pre[np.logical_or(fluid.pre<0.0, np.isnan(fluid.pre))] = 0.0            
        if verbose == 1:
            print('fluid.rho',fluid.rho)
            print('fluid.vel',fluid.vel)
            print('fluid.pre',fluid.pre)            
        
        
    def SetConserved(self, mesh, fluid, verbose=0):
        vol = mesh.vol
        fluid.Mass = fluid.rho * vol
        fluid.Mom = fluid.rho * fluid.vel * vol
        fluid.Energy = (0.5*fluid.rho*fluid.vel**2 + fluid.pre/(fluid.eos.gamma-1.0))*vol
        fluid.Mass[np.logical_or(fluid.Mass<0.0, np.isnan(fluid.Mass))] = 0.0
        fluid.Energy[np.logical_or(fluid.Energy<0.0, np.isnan(fluid.Energy))] = 0.0
        if verbose == 1:
            print('fluid.Mass',fluid.Mass)
            print('fluid.Mom',fluid.Mom)
            print('fluid.Energy',fluid.Energy)
        
        
    def SetGradient(self, mesh, fluid):
        xdelta = mesh.xdelta
        fluid.rho.grad = ru.CalGradient(fluid.rho, xdelta)
        fluid.vel.grad   = ru.CalGradient(fluid.vel, xdelta)
        fluid.pre.grad   = ru.CalGradient(fluid.pre, xdelta)
        
        
    def SetConservedDensityFlux(self, fluid):
        fluid.Mass.F, fluid.Mass.q, fluid.Mom.F, fluid.Mom.q, fluid.Energy.F, fluid.Energy.q = ru.GetFQ(fluid.rho,fluid.vel,fluid.pre,fluid.eos.gamma)
        
    def SetFaceLR(self, mesh, fluid, order=0):
        #numpy roll Rroll, put the right value to this cell
        Lroll = 1
        Rroll = -1
        if order == 0 or order == 1:
            fluid.rho.R = fluid.rho
            fluid.rho.L = np.roll(fluid.rho, Lroll)
            fluid.vel.R = fluid.vel
            fluid.vel.L = np.roll(fluid.vel, Lroll)
            fluid.pre.R = fluid.pre
            fluid.pre.L = np.roll(fluid.pre, Lroll)
            if order == 1:
                self.SetGradient(mesh, fluid)
                fluid.rho.R.first, fluid.rho.L.first = ru.extrapolateToFace(fluid.rho, mesh.boundary, fluid.rho.grad, order=1)
                fluid.vel.R.first, fluid.vel.L.first = ru.extrapolateToFace(fluid.vel, mesh.boundary, fluid.vel.grad, order=1)
                fluid.pre.R.first, fluid.pre.L.first = ru.extrapolateToFace(fluid.pre, mesh.boundary, fluid.pre.grad, order=1)
        else:
            print('order unknown')


    def SetFluxOnFace(self,fluid,order=0):
        Mass_flux_0, Mom_flux_0, Energy_flux_0 = ru.CalFluxFromLR(fluid.rho.L,fluid.rho.R,
                                                               fluid.vel.L,fluid.vel.R,
                                                               fluid.pre.L,fluid.pre.R,
                                                               fluid.eos.gamma,fluid.cmax)
        if order==0:
            fluid.Mass.flux, fluid.Mom.flux, fluid.Energy.flux = Mass_flux_0, Mom_flux_0, Energy_flux_0
        elif order==1:
            Mass_flux_1, Mom_flux_1, Energy_flux_1 = ru.CalFluxFromLR(fluid.rho.L.first, fluid.rho.R.first,
                                                                    fluid.vel.L.first, fluid.vel.R.first,
                                                                    fluid.pre.L.first, fluid.pre.R.first,
                                                                    fluid.eos.gamma, fluid.cmax)
            self.SetConservedDensityFlux(fluid)
            fluid.Mass.flux, fluid.philim_Mass= ru.ApplyFluxLimiter(fluid.Mass.q,Mass_flux_1,Mass_flux_0)
            fluid.Mom.flux, fluid.philim_Mom  = ru.ApplyFluxLimiter(fluid.Mom.q,Mom_flux_1,Mom_flux_0)
            fluid.Energy.flux, fluid.philim_Energy  = ru.ApplyFluxLimiter(fluid.Energy.q,Energy_flux_1,Energy_flux_0)
            
        else:
            print('order unknown')

        
    def SetInterFaceFlux(self,mesh,fluid,method='Rusanov',verbose=0, order=0):
        if method=='GLF' or method=='Rusanov':
            #numpy roll Rroll, put the right value to this cell
            Lroll = 1
            Rroll = -1
            if method=='GLF':
                # Global Lax Friedrich scheme
                # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)  
                # simple to implement but very diffusive
                # calculate cmax
                fluid.cmax = mesh.xdelta/np.amin(self.dt)
            elif method=='Rusanov':
                # Local Lax Friedrich schem
                # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)  
                # simple to implement but less diffusive
                fluid.cmax = np.maximum(fluid.vsignal, np.roll(fluid.vsignal,Lroll))
            
            self.SetFaceLR(mesh,fluid, order=order)
            self.SetFluxOnFace(fluid, order=order)
        else:
            raise Exception("Interface flux method unknown") 
        if (verbose==1):
            print('self.Mass.flux',self.Mass.flux)
            print('self.Mom.flux',self.Mom.flux)
            print('self.Energy.flux',self.Energy.flux)
            
            
    def AddFluxes(self, dt: float, mesh, fluid):
        #numpy roll Rroll, put the right value to this cell
        Lroll = 1
        Rroll = -1
        # Add the interface fluxes to the cells:
        area = mesh.area
        fluid.Mass += (fluid.Mass.flux*area - np.roll(fluid.Mass.flux*area,Rroll))*dt
        fluid.Mom  += (fluid.Mom.flux*area - np.roll(fluid.Mom.flux*area,Rroll))*dt
        fluid.Energy  += (fluid.Energy.flux*area - np.roll(fluid.Energy.flux*area,Rroll))*dt
        # advance time
        fluid.time += dt
        
    
    def GetTimeStep(self, mesh, fluid, CFL=0.1, vsmin = 0.001*unyt.km/unyt.s):
        fluid.SetSoundSpeed()
        vsignal = np.absolute(fluid.vel) + fluid.cs
        #if np.any(vsignal==0) or np.any(np.isnan(vsignal)):
        #    print('vsignal vanished')    
        #else:
        vsignal[np.logical_or(vsignal==0.0,np.isnan(vsignal))] = vsmin
        dx_c = mesh.xdelta / vsignal
        self.dt =  CFL * np.amin(dx_c)
        fluid.vsignal = vsignal
        dt = np.amin(self.dt)
        #print('vsignal', vsignal)
        return dt