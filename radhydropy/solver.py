"""Finite-volume hydrodynamics solver operations."""

import radhydropy.utils as ru
import unyt
import numpy as np

class Solver():
    """Advance one-dimensional Euler equations on a RadHydropy mesh."""

    def __init__(self) -> None:
        # should add information like
        # limiter, first order, what method
        # and time
        pass

    def _safe_divide(self, numerator, denominator):
        return ru.SafeDivide(numerator, denominator)

    def _spherical_center_cell_index(self, mesh):
        if getattr(mesh, 'coordsys', None) != 'spherical' or not hasattr(mesh, 'boundary'):
            return None
        origin = 0.0 * mesh.boundary.units
        origin_faces = np.where(mesh.boundary[:-1] == origin)[0]
        if len(origin_faces) > 0:
            return int(origin_faces[0])
        origin_cells = np.where(
            np.logical_and(mesh.boundary[:-1] < origin, mesh.boundary[1:] > origin)
        )[0]
        if len(origin_cells) > 0:
            return int(origin_cells[0])
        return None

    def _spherical_origin_face_index(self, mesh):
        if getattr(mesh, 'coordsys', None) != 'spherical' or not hasattr(mesh, 'boundary'):
            return None
        origin = 0.0 * mesh.boundary.units
        origin_faces = np.where(mesh.boundary[:-1] == origin)[0]
        if len(origin_faces) > 0:
            return int(origin_faces[0])
        return None

    def _zero_spherical_origin_flux(self, mesh, fluid):
        origin_face = self._spherical_origin_face_index(mesh)
        if origin_face is None:
            return
        fluid.Mass.flux[origin_face] = 0.0 * fluid.Mass.flux.units
        fluid.Mom.flux[origin_face] = 0.0 * fluid.Mom.flux.units
        fluid.Energy.flux[origin_face] = 0.0 * fluid.Energy.flux.units

    def _zero_spherical_center_momentum(self, mesh, fluid):
        center_cell = self._spherical_center_cell_index(mesh)
        if center_cell is None:
            return
        fluid.Mom[center_cell] = 0.0 * fluid.Mom.units

    def SetPrimitive(self, mesh, fluid, verbose=0):
        """Update primitive variables from conserved quantities."""
        vol = mesh.vol
        fluid.rho = self._safe_divide(fluid.Mass, vol)
        fluid.vel = self._safe_divide(fluid.Mom, fluid.Mass)
        energy_density = self._safe_divide(fluid.Energy, vol)
        fluid.pre = (energy_density-0.5*fluid.rho*fluid.vel**2)*(fluid.eos.gamma-1.0)
        fluid.rho[np.logical_or(fluid.rho<0.0, np.isnan(fluid.rho))] = 0.0
        fluid.vel[np.isnan(fluid.vel)] = 0.0 * fluid.vel.units
        fluid.pre[np.logical_or(fluid.pre<0.0, np.isnan(fluid.pre))] = 0.0            
        center_cell = self._spherical_center_cell_index(mesh)
        if center_cell is not None:
            fluid.vel[center_cell] = 0.0 * fluid.vel.units
        if verbose == 1:
            print('fluid.rho',fluid.rho)
            print('fluid.vel',fluid.vel)
            print('fluid.pre',fluid.pre)            
        
        
    def SetConserved(self, mesh, fluid, verbose=0):
        """Update conserved mass, momentum, and energy from primitive variables."""
        vol = mesh.vol
        fluid.Mass = fluid.rho * vol
        fluid.Mom = fluid.rho * fluid.vel * vol
        fluid.Energy = (0.5*fluid.rho*fluid.vel**2 + fluid.pre/(fluid.eos.gamma-1.0))*vol
        fluid.Mass[np.logical_or(fluid.Mass<0.0, np.isnan(fluid.Mass))] = 0.0
        fluid.Energy[np.logical_or(fluid.Energy<0.0, np.isnan(fluid.Energy))] = 0.0
        self._zero_spherical_center_momentum(mesh, fluid)
        if verbose == 1:
            print('fluid.Mass',fluid.Mass)
            print('fluid.Mom',fluid.Mom)
            print('fluid.Energy',fluid.Energy)
        
        
    def SetGradient(self, mesh, fluid):
        """Calculate centered gradients for density, velocity, and pressure."""
        xdelta = mesh.xdelta
        fluid.rho.grad = ru.CalGradient(fluid.rho, xdelta)
        fluid.vel.grad   = ru.CalGradient(fluid.vel, xdelta)
        fluid.pre.grad   = ru.CalGradient(fluid.pre, xdelta)
        
        
    def SetConservedDensityFlux(self, fluid):
        """Store Euler fluxes and conserved densities on fluid arrays."""
        fluid.Mass.F, fluid.Mass.q, fluid.Mom.F, fluid.Mom.q, fluid.Energy.F, fluid.Energy.q = ru.GetFQ(fluid.rho,fluid.vel,fluid.pre,fluid.eos.gamma)
        
    def SetFaceLR(self, mesh, fluid, boundcond, order=0):
        """Construct left and right states at cell faces.

        ``order=0`` uses piecewise constant states. ``order=1`` applies a
        gradient reconstruction before limiting the fluxes.
        """
        #numpy roll Rroll, put the right value to this cell
        Lroll = 1
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
            raise ValueError('order unknown: %s'%order)


    def SetFluxOnFace(self,fluid,boundcond,order=0):
        """Calculate mass, momentum, and energy fluxes at interfaces."""
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
            raise ValueError('order unknown: %s'%order)
        #zero out all flux at the center for symmetric boundary at origin:
        #if boundcond == 'OpenSph':
        #    fluid.Mass.flux[0] = 0.0 * unyt.g / unyt.cm**2 / unyt.s 
        #    fluid.Mom.flux[0] = 0.0 * unyt.g / unyt.cm / unyt.s **2
        #    fluid.Energy.flux[0] = 0.0 * unyt.g / unyt.s**3

        
    def SetInterFaceFlux(self,mesh,fluid,boundcond, method='Rusanov',verbose=0, order=0):
        """Set interface fluxes using GLF or Rusanov numerical fluxes."""
        if method=='GLF' or method=='Rusanov':
            #numpy roll Rroll, put the right value to this cell
            Lroll = 1
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
            self._zero_spherical_origin_flux(mesh, fluid)
        else:
            raise ValueError("Interface flux method unknown: %s"%method) 
        if (verbose==1):
            print('fluid.Mass.flux',fluid.Mass.flux)
            print('fluid.Mom.flux',fluid.Mom.flux)
            print('fluid.Energy.flux',fluid.Energy.flux)
            
            
    def AddFluxes(self, dt: float, mesh, fluid, boundcond):
        """Apply interface fluxes to conserved quantities and advance time."""
        #numpy roll Rroll, put the right value to this cell
        Rroll = -1

        # Add the interface fluxes to the cells:
        area = mesh.area
        df_Mass = fluid.Mass.flux*area - np.roll(fluid.Mass.flux*area,Rroll)
        df_Mom = fluid.Mom.flux*area - np.roll(fluid.Mom.flux*area,Rroll)
        df_Energy = fluid.Energy.flux*area - np.roll(fluid.Energy.flux*area,Rroll)
        if getattr(mesh, 'coordsys', None) == 'spherical':
            area_right = np.roll(area, Rroll)
            df_Mom += fluid.pre * (area_right - area)

        # we zero out the flux from the inner most boundary? 
        #if boundcond == "OpenSph":
        #    df_Mass[0] = - fluid.Mass.flux[1]*area[1]
        #    df_Mom[0] = - fluid.Mom.flux[1]*area[1]
        #    df_Energy[0] = - fluid.Energy.flux[1]*area[1]

        fluid.Mass += df_Mass*dt
        fluid.Mom  += df_Mom*dt
        fluid.Energy  += df_Energy*dt
        self._zero_spherical_center_momentum(mesh, fluid)

        # advance time
        fluid.time += dt


    def SetBoundary(self, mesh, fluid, par):
        """Fill ghost cells according to the selected boundary condition."""
        btype = par.boundcond
        noghost = par.noghost
        nogrid = par.nogrid
        nolast = noghost + nogrid -1
        first = noghost
        right_start = noghost + nogrid
        interior = slice(first, right_start)
        left_ghost = slice(0, noghost)
        right_ghost = slice(right_start, right_start + noghost)
        fields = ('rho', 'vel', 'pre')

        def copy_left(values):
            for attr, value in values.items():
                getattr(fluid, attr)[left_ghost] = value

        def copy_right(values):
            for attr, value in values.items():
                getattr(fluid, attr)[right_ghost] = value

        def apply_spherical_inner_boundary():
            mirror_start = first
            if mesh is not None and hasattr(mesh, 'boundary'):
                origin = 0.0 * mesh.boundary.units
                if mesh.boundary[first] < origin and mesh.boundary[first+1] > origin:
                    mirror_start = first + 1
            copy_left({
                'rho': fluid.rho[mirror_start:mirror_start+noghost][::-1],
                'vel': -fluid.vel[mirror_start:mirror_start+noghost][::-1],
                'pre': fluid.pre[mirror_start:mirror_start+noghost][::-1],
            })

        if btype == 'Periodic':
            for attr in fields:
                quan = getattr(fluid, attr)
                quan[left_ghost] = quan[interior][-noghost:]
                quan[right_ghost] = quan[interior][:noghost]
        elif btype == 'Open':
            # open boundary condition does not mean the gradient is zero.
            for attr in fields:
                quan = getattr(fluid, attr)
                quan[left_ghost] = quan[first]
                quan[right_ghost] = quan[nolast]
        elif btype == 'Reflecting': 
            fluid.rho[left_ghost] = fluid.rho[interior][:noghost][::-1]
            fluid.vel[left_ghost] = -fluid.vel[interior][:noghost][::-1]
            fluid.pre[left_ghost] = fluid.pre[interior][:noghost][::-1]
            fluid.rho[right_ghost] = fluid.rho[interior][-noghost:][::-1]
            fluid.vel[right_ghost] = -fluid.vel[interior][-noghost:][::-1]
            fluid.pre[right_ghost] = fluid.pre[interior][-noghost:][::-1]
        elif btype == 'OpenSph':
            # spherical open boundary condition
            # open only at outer boundary
            # symmetric at the center
            # this means zero flux at r=0 
            # imply zero gradient?
            apply_spherical_inner_boundary()
            copy_right({
                'rho': fluid.rho[nolast],
                'vel': fluid.vel[nolast],
                'pre': fluid.pre[nolast],
            })
        elif btype == 'InflowSph':
            pre_inflow = ru.CalPressure(par.rho_inflow,par.temp_inflow,par.mu_inflow)
            apply_spherical_inner_boundary()
            copy_right({
                'rho': par.rho_inflow,
                'vel': par.vel_inflow,
                'pre': pre_inflow,
            })
        elif btype == 'OutflowSph':
            pre_outflow = ru.CalPressure(par.rho_outflow,par.temp_outflow,par.mu_outflow)
            copy_left({
                'rho': par.rho_outflow,
                'vel': par.vel_outflow,
                'pre': pre_outflow,
            })
            copy_right({
                'rho': fluid.rho[nolast],
                'vel': fluid.vel[nolast],
                'pre': fluid.pre[nolast],
            })
        else:
            raise ValueError('Boundary condition unknown: %s'%btype) 
        
    
    def GetTimeStep(self, mesh, fluid, par, CFL=None):
        """Return a CFL-limited timestep."""
        if CFL is None:
            CFL = par.CFL
        fluid.SetSoundSpeed()
        vsignal = np.absolute(fluid.vel) + fluid.cs
        dt_array = np.divide(
            CFL * mesh.xdelta,
            vsignal,
            out=np.ones_like(vsignal.value) * par.dtmax,
            where=vsignal != 0.0,
        )
        self.dt = np.amin(dt_array)
        fluid.vsignal = vsignal
        dt = np.amin(self.dt)
        if np.isnan(np.array(dt)):
            print('vsignal', vsignal)
            print('fluid.vel',fluid.vel)
            print('fluid.cs',fluid.cs)
            raise Exception(" time step is nan")
        if dt < par.dtmin:
            raise ValueError(" time step %.2e smaller than the minimum time step %.2e"%(dt,par.dtmin))
        if dt > par.dtmax:
            dt = par.dtmax
            #raise Exception(" time step %.2e larger than the maximum time step %.2e"%(dt,par.dtmax))
        return dt
