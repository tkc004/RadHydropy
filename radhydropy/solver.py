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
        
    def SetFaceLR(self, mesh, fluid, boundcond, order=0):
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
            #if boundcond == "OpenSph":
            #    fluid.vel.L[0] = 0.0
            #    fluid.rho.L[0] = fluid.rho.L[1]
            #    fluid.pre.L[0] = fluid.pre.L[1] 
            if order == 1:
                self.SetGradient(mesh, fluid)
                fluid.rho.R.first, fluid.rho.L.first = ru.extrapolateToFace(fluid.rho, mesh.boundary, fluid.rho.grad, order=1)
                fluid.vel.R.first, fluid.vel.L.first = ru.extrapolateToFace(fluid.vel, mesh.boundary, fluid.vel.grad, order=1)
                fluid.pre.R.first, fluid.pre.L.first = ru.extrapolateToFace(fluid.pre, mesh.boundary, fluid.pre.grad, order=1)
        else:
            raise Exception('order unknown')


    def SetFluxOnFace(self,fluid,boundcond,order=0):
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
            raise Exception('order unknown')
        #zero out all flux at the center for symmetric boundary at origin:
        #if boundcond == 'OpenSph':
        #    fluid.Mass.flux[0] = 0.0 * unyt.g / unyt.cm**2 / unyt.s 
        #    fluid.Mom.flux[0] = 0.0 * unyt.g / unyt.cm / unyt.s **2
        #    fluid.Energy.flux[0] = 0.0 * unyt.g / unyt.s**3

        
    def SetInterFaceFlux(self,mesh,fluid,boundcond, method='Rusanov',verbose=0, order=0):
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
            
            self.SetFaceLR(mesh,fluid, boundcond, order=order)
            self.SetFluxOnFace(fluid, boundcond, order=order)
        else:
            raise Exception("Interface flux method unknown") 
        if (verbose==1):
            print('self.Mass.flux',self.Mass.flux)
            print('self.Mom.flux',self.Mom.flux)
            print('self.Energy.flux',self.Energy.flux)
            
            
    def AddFluxes(self, dt: float, mesh, fluid, boundcond):
        #numpy roll Rroll, put the right value to this cell
        Lroll = 1
        Rroll = -1

        # Add the interface fluxes to the cells:
        area = mesh.area
        df_Mass = fluid.Mass.flux*area - np.roll(fluid.Mass.flux*area,Rroll)
        df_Mom = fluid.Mom.flux*area - np.roll(fluid.Mom.flux*area,Rroll)
        df_Energy = fluid.Energy.flux*area - np.roll(fluid.Energy.flux*area,Rroll)

        # we zero out the flux from the inner most boundary? 
        #if boundcond == "OpenSph":
        #    df_Mass[0] = - fluid.Mass.flux[1]*area[1]
        #    df_Mom[0] = - fluid.Mom.flux[1]*area[1]
        #    df_Energy[0] = - fluid.Energy.flux[1]*area[1]

        fluid.Mass += df_Mass*dt
        fluid.Mom  += df_Mom*dt
        fluid.Energy  += df_Energy*dt

        # advance time
        fluid.time += dt


    def SetBoundary(self, mesh, fluid, par):
        btype = par.boundcond
        noghost = par.noghost
        nogrid = par.nogrid
        nolast = noghost + nogrid -1
        if btype == 'Periodic':
            for ig in range(1,noghost+1):
                fluid.rho[noghost-ig] = fluid.rho[nolast + 1 -ig]
                fluid.vel[noghost-ig] = fluid.vel[nolast + 1 -ig]
                fluid.pre[noghost-ig] = fluid.pre[nolast + 1 -ig]
                fluid.rho[nolast+ig] = fluid.rho[noghost -1 + ig]
                fluid.vel[nolast+ig] = fluid.vel[noghost -1 + ig]
                fluid.pre[nolast+ig] = fluid.pre[noghost -1 + ig] 
        elif btype == 'Open':
            for ig in range(1,noghost+1):
                fluid.rho[noghost-ig] = fluid.rho[noghost]
                fluid.vel[noghost-ig] = fluid.vel[noghost]
                fluid.pre[noghost-ig] = fluid.pre[noghost]
                fluid.rho[nolast+ig] = fluid.rho[nolast]
                fluid.vel[nolast+ig] = fluid.vel[nolast]
                fluid.pre[nolast+ig] = fluid.pre[nolast]
        elif btype == 'Reflecting': 
            for ig in range(1,noghost+1): 
                fluid.rho[noghost-ig] = fluid.rho[noghost-1+ig]
                fluid.vel[noghost-ig] = -fluid.vel[noghost-1+ig]
                fluid.pre[noghost-ig] = fluid.pre[noghost-1+ig]
                fluid.rho[nolast+ig] = fluid.rho[nolast + 1 -ig]
                fluid.vel[nolast+ig] = -fluid.vel[nolast + 1 -ig]
                fluid.pre[nolast+ig] = fluid.pre[nolast + 1 -ig]
        elif btype == 'OpenSph':
            # spherical open boundary condition
            # open only at outer boundary
            # symmetric at the center
            # this means zero flux at r=0 
            # imply zero gradient?
            # zero velocity at r=0
            fluid.vel[noghost+1] *= 0.0
            for ig in range(1,noghost+1): 
                fluid.rho[noghost-ig] = fluid.rho[noghost+ig]
                fluid.vel[noghost-ig] = -fluid.vel[noghost+ig]
                fluid.pre[noghost-ig] = fluid.pre[noghost+ig]
                fluid.rho[nolast+ig] = fluid.rho[nolast]
                fluid.vel[nolast+ig] = fluid.vel[nolast]
                fluid.pre[nolast+ig] = fluid.pre[nolast]  
            fluid.rho[noghost] = fluid.rho[noghost+1]
            fluid.vel[noghost] *= 0.0
            fluid.pre[noghost] = fluid.pre[noghost+1] 
            fluid.rho[noghost-1] = fluid.rho[noghost+1]
            fluid.vel[noghost-1] *= 0.0
            fluid.pre[noghost-1] = fluid.pre[noghost+1]         
        elif btype == 'InflowSph':
            fluid.vel[noghost+1] *= 0.0
            pre_inflow = ru.CalPressure(par.rho_inflow,par.temp_inflow,par.mu_inflow)
            for ig in range(1,noghost+1): 
                fluid.rho[noghost-ig] = fluid.rho[noghost+ig]
                fluid.vel[noghost-ig] = -fluid.vel[noghost+ig]
                fluid.pre[noghost-ig] = fluid.pre[noghost+ig]
                fluid.rho[nolast+ig] = par.rho_inflow
                fluid.vel[nolast+ig] = par.vel_inflow
                fluid.pre[nolast+ig] = pre_inflow 
            fluid.rho[noghost] = fluid.rho[noghost+1]
            fluid.vel[noghost] *= 0.0
            fluid.pre[noghost] = fluid.pre[noghost+1] 
            fluid.rho[noghost-1] = fluid.rho[noghost+1]
            fluid.vel[noghost-1] *= 0.0
            fluid.pre[noghost-1] = fluid.pre[noghost+1] 
        elif btype == 'OutflowSph':
            pre_outflow = ru.CalPressure(par.rho_outflow,par.temp_outflow,par.mu_outflow)
            for ig in range(1,noghost+1):
                fluid.rho[noghost-ig] = par.rho_outflow
                fluid.vel[noghost-ig] = par.vel_outflow
                fluid.pre[noghost-ig] = pre_outflow
                fluid.rho[nolast+ig] = fluid.rho[nolast]
                fluid.vel[nolast+ig] = fluid.vel[nolast]
                fluid.pre[nolast+ig] = fluid.pre[nolast]                   
        else:
            raise Exception('Boundary condition unknown') 
        
    
    def GetTimeStep(self, mesh, fluid, par, CFL=0.1):
        fluid.SetSoundSpeed()
        vsignal = np.absolute(fluid.vel) + fluid.cs
        dt_array =  CFL * mesh.xdelta / vsignal
        dt_array[vsignal==0] = par.dtmax 
        self.dt = np.amin(dt_array)
        fluid.vsignal = vsignal
        dt = np.amin(self.dt)
        if np.isnan(np.array(dt)):
            print('vsignal', vsignal)
            print('fluid.vel',fluid.vel)
            print('fluid.cs',fluid.cs)
            raise Exception(" time step is nan")
        if dt < par.dtmin:
            raise Exception(" time step %.2e smaller than the minimum time step %.2e"%(dt,par.dtmin))
        if dt > par.dtmax:
            dt = par.dtmax
            #raise Exception(" time step %.2e larger than the maximum time step %.2e"%(dt,par.dtmax))
        return dt