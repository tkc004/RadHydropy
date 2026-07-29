"""Finite-volume hydrodynamics solver operations."""

import radhydropy.utils as ru
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
import radhydropy.gravity as rg
from radhydropy.units import (
    CGS_AREA_UNIT,
    CGS_MASS_DENSITY_UNIT,
    CGS_NUMBER_DENSITY_UNIT,
    CGS_PHOTON_FLUX_UNIT,
    CGS_RATE_UNIT,
    CGS_VOLUME_UNIT,
    _as_cgs_float,
)
import numpy as np
from types import SimpleNamespace
import unyt

class Solver():
    """Advance one-dimensional Euler equations on a RadHydropy mesh."""

    def __init__(self) -> None:
        # should add information like
        # limiter, first order, what method
        # and time
        pass

    def _safe_divide(self, numerator, denominator):
        return ru.SafeDivide(numerator, denominator)

    def _interior_slice(self, par):
        first = par.noghost
        return slice(first, first + par.nogrid)

    def _thermochemistry_enabled(self, fluid, par):
        return getattr(par, 'hydrogen_chemistry', False) and hasattr(fluid, 'xHI')

    def _thermochemistry_radiation_enabled(self, fluid, par):
        return (
            self._thermochemistry_enabled(fluid, par)
            and (
                getattr(par, 'hydrogen_radiation_field', False)
                or getattr(par, 'radiative_transfer', False)
            )
            and hasattr(fluid, 'ngamma')
        )

    def _thermochemistry_radiation_evolution_enabled(self, fluid, par):
        return (
            self._thermochemistry_radiation_enabled(fluid, par)
            and getattr(par, 'hydrogen_radiation_evolution', True)
            and not getattr(par, 'radiative_transfer', False)
        )

    def ApplyRadiativeTransfer(self, mesh, fluid, par):
        """Refresh photon density from the shared radiative-transfer solver."""
        if not getattr(par, 'radiative_transfer', False):
            return None
        if not hasattr(fluid, 'ngamma'):
            fluid.ngamma = np.zeros(np.shape(fluid.rho), dtype=float) * CGS_NUMBER_DENSITY_UNIT
        interior = self._interior_slice(par)
        submesh = SimpleNamespace(
            coordsys=getattr(mesh, 'coordsys', 'cartesian'),
            boundary=mesh.boundary[interior.start : interior.stop + 1].to_value(unyt.cm),
            vol=mesh.vol[interior].to_value(CGS_VOLUME_UNIT),
        )
        if hasattr(mesh, 'area'):
            submesh.area = mesh.area[interior].to_value(CGS_AREA_UNIT)
        sigma_gamma_cm2 = _as_cgs_float(
            getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA),
            CGS_AREA_UNIT,
        )
        result = rrt.trace_long_characteristics(
            submesh,
            fluid.rho[interior].to_value(CGS_MASS_DENSITY_UNIT),
            np.asarray(fluid.xHI[interior], dtype=float),
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
            sigma_gamma=sigma_gamma_cm2,
            boundary_flux=_as_cgs_float(
                getattr(par, 'radiative_transfer_boundary_flux', 0.0),
                CGS_PHOTON_FLUX_UNIT,
            ),
            source_photon_rate=_as_cgs_float(
                getattr(par, 'radiative_transfer_source_photon_rate', 0.0),
                CGS_RATE_UNIT,
            ),
            direction=getattr(par, 'radiative_transfer_direction', 1),
            coordsys=getattr(mesh, 'coordsys', 'cartesian'),
        )
        fluid.ngamma[interior] = result.cell_photon_density.to_value(fluid.ngamma.units)
        return result

    def _spherical_center_cell_index(self, mesh):
        if getattr(mesh, 'coordsys', None) != 'spherical' or not hasattr(mesh, 'boundary'):
            return None
        origin = 0.0 * getattr(mesh.boundary, 'units', 1.0)
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
        origin = 0.0 * getattr(mesh.boundary, 'units', 1.0)
        origin_faces = np.where(mesh.boundary[:-1] == origin)[0]
        if len(origin_faces) > 0:
            return int(origin_faces[0])
        return None

    def _zero_spherical_origin_flux(self, mesh, fluid):
        origin_face = self._spherical_origin_face_index(mesh)
        if origin_face is None:
            return
        if hasattr(fluid.Mass.flux, 'units'):
            fluid.Mass.flux[origin_face] = 0.0 * fluid.Mass.flux.units
            fluid.Mom.flux[origin_face] = 0.0 * fluid.Mom.flux.units
            fluid.Energy.flux[origin_face] = 0.0 * fluid.Energy.flux.units
        else:
            fluid.Mass.flux[origin_face] = 0.0
            fluid.Mom.flux[origin_face] = 0.0
            fluid.Energy.flux[origin_face] = 0.0

    def _zero_spherical_center_momentum(self, mesh, fluid):
        center_cell = self._spherical_center_cell_index(mesh)
        if center_cell is None:
            return
        if hasattr(fluid.Mom, 'units'):
            fluid.Mom[center_cell] = 0.0 * fluid.Mom.units
        else:
            fluid.Mom[center_cell] = 0.0

    def SetPrimitive(self, mesh, fluid, verbose=0):
        """Update primitive variables from conserved quantities."""
        vol = mesh.vol
        fluid.rho = self._safe_divide(fluid.Mass, vol)
        fluid.vel = self._safe_divide(fluid.Mom, fluid.Mass)
        energy_density = self._safe_divide(fluid.Energy, vol)
        fluid.pre = fluid.eos.pressure_from_conserved(
            fluid.rho,
            fluid.vel,
            energy_density,
            temp=getattr(fluid, 'temp', None),
            mu=getattr(fluid, 'mu', None),
        )
        fluid.rho[np.logical_or(fluid.rho<0.0, np.isnan(fluid.rho))] = 0.0
        fluid.vel[np.isnan(fluid.vel)] = 0.0
        fluid.pre[np.logical_or(fluid.pre<0.0, np.isnan(fluid.pre))] = 0.0
        center_cell = self._spherical_center_cell_index(mesh)
        if center_cell is not None:
            fluid.vel[center_cell] = 0.0
        if verbose == 1:
            print('fluid.rho',fluid.rho)
            print('fluid.vel',fluid.vel)
            print('fluid.pre',fluid.pre)            
    
    def SetConserved(self, mesh, fluid, verbose=0):
        """Update conserved mass, momentum, and energy from primitive variables."""
        vol = mesh.vol
        fluid.Mass = fluid.rho * vol
        fluid.Mom = fluid.rho * fluid.vel * vol
        fluid.Energy = fluid.eos.total_energy_density(
            fluid.rho,
            fluid.vel,
            fluid.pre,
        ) * vol
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
        fluid.vel.grad = ru.CalGradient(fluid.vel, xdelta)
        fluid.pre.grad = ru.CalGradient(fluid.pre, xdelta)
        
        
    def SetConservedDensityFlux(self, fluid):
        """Store Euler fluxes and conserved densities on fluid arrays."""
        (
            fluid.Mass.F,
            fluid.Mass.q,
            fluid.Mom.F,
            fluid.Mom.q,
            fluid.Energy.F,
            fluid.Energy.q,
        ) = fluid.eos.fluxes(fluid.rho, fluid.vel, fluid.pre)
        
    def SetFaceLR(self, mesh, fluid, boundcond, order=0):
        """Construct left and right states at cell faces.

        ``order=0`` uses piecewise constant states. ``order=1`` applies a
        gradient reconstruction before limiting the fluxes.
        """
        # Start from neighbor-shifted cell states, then optionally replace them
        # with reconstructed face values for second-order updates.
        Lroll = 1
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
            raise ValueError('order unknown: %s'%order)


    def SetFluxOnFace(self,fluid,boundcond,order=0):
        """Calculate mass, momentum, and energy fluxes at interfaces."""
        (
            Fmass_L,
            qmass_L,
            Fmom_L,
            qmom_L,
            FEn_L,
            qEn_L,
        ) = fluid.eos.fluxes(fluid.rho.L, fluid.vel.L, fluid.pre.L)
        (
            Fmass_R,
            qmass_R,
            Fmom_R,
            qmom_R,
            FEn_R,
            qEn_R,
        ) = fluid.eos.fluxes(fluid.rho.R, fluid.vel.R, fluid.pre.R)
        Mass_flux_0 = ru.CalInterFaceFluxGLF(Fmass_L, Fmass_R, qmass_L, qmass_R, fluid.cmax)
        Mom_flux_0 = ru.CalInterFaceFluxGLF(Fmom_L, Fmom_R, qmom_L, qmom_R, fluid.cmax)
        Energy_flux_0 = ru.CalInterFaceFluxGLF(FEn_L, FEn_R, qEn_L, qEn_R, fluid.cmax)
        if order==0:
            fluid.Mass.flux, fluid.Mom.flux, fluid.Energy.flux = Mass_flux_0, Mom_flux_0, Energy_flux_0
        elif order==1:
            (
                Fmass_L,
                qmass_L,
                Fmom_L,
                qmom_L,
                FEn_L,
                qEn_L,
            ) = fluid.eos.fluxes(fluid.rho.L.first, fluid.vel.L.first, fluid.pre.L.first)
            (
                Fmass_R,
                qmass_R,
                Fmom_R,
                qmom_R,
                FEn_R,
                qEn_R,
            ) = fluid.eos.fluxes(fluid.rho.R.first, fluid.vel.R.first, fluid.pre.R.first)
            Mass_flux_1 = ru.CalInterFaceFluxGLF(Fmass_L, Fmass_R, qmass_L, qmass_R, fluid.cmax)
            Mom_flux_1 = ru.CalInterFaceFluxGLF(Fmom_L, Fmom_R, qmom_L, qmom_R, fluid.cmax)
            Energy_flux_1 = ru.CalInterFaceFluxGLF(FEn_L, FEn_R, qEn_L, qEn_R, fluid.cmax)
            self.SetConservedDensityFlux(fluid)
            fluid.Mass.flux, fluid.philim_Mass = ru.ApplyFluxLimiter(fluid.Mass.q, Mass_flux_1, Mass_flux_0)
            fluid.Mom.flux, fluid.philim_Mom = ru.ApplyFluxLimiter(fluid.Mom.q, Mom_flux_1, Mom_flux_0)
            fluid.Energy.flux, fluid.philim_Energy = ru.ApplyFluxLimiter(fluid.Energy.q, Energy_flux_1, Energy_flux_0)
        else:
            raise ValueError('order unknown: %s'%order)
        
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
        # Shift the face fluxes so each cell receives the net in-flow minus
        # out-flow through its two bounding faces.
        Rroll = -1
        area = mesh.area
        df_Mass = fluid.Mass.flux*area - np.roll(fluid.Mass.flux*area,Rroll)
        df_Mom = fluid.Mom.flux*area - np.roll(fluid.Mom.flux*area,Rroll)
        df_Energy = fluid.Energy.flux*area - np.roll(fluid.Energy.flux*area,Rroll)
        if getattr(mesh, 'coordsys', None) == 'spherical':
            # Spherical momentum needs the geometric pressure term from the
            # changing face area, not just the flux divergence.
            area_right = np.roll(area, Rroll)
            df_Mom += fluid.pre * (area_right - area)

        fluid.Mass += df_Mass*dt
        fluid.Mom  += df_Mom*dt
        fluid.Energy  += df_Energy*dt
        self._zero_spherical_center_momentum(mesh, fluid)

        # advance time
        fluid.time += dt

    def _gravity_model(self, par):
        """Return the configured gravity model, if any."""
        gravity = getattr(par, "gravity", None)
        if isinstance(gravity, rg.Gravity):
            return gravity
        if gravity is not None and hasattr(gravity, "acceleration_on_mesh"):
            return gravity
        if not getattr(par, "externalgravity", False):
            return None
        return rg.Gravity(
            selfgravity=getattr(par, "selfgravity", False),
            externalgravity=getattr(par, "externalgravity", False),
            potential=getattr(par, "gravity_potential", None),
            coordinate=getattr(par, "gravity_coordinate", None),
            acceleration=getattr(par, "gravity_acceleration", None),
            code_units=getattr(par, "code_units", getattr(par, "CodeUnits", None)),
        )

    def ApplyExternalGravity(self, dt, mesh, fluid, par):
        """Apply a source update from an optional external gravitational field."""
        gravity = self._gravity_model(par)
        if gravity is None or not gravity.externalgravity:
            return 0
        acceleration = gravity.acceleration_on_mesh(mesh)
        code_units = getattr(par, "code_units", getattr(par, "CodeUnits", None))
        if code_units is not None:
            target_unit = code_units.length_unit / code_units.time_unit**2
            if hasattr(acceleration, "to"):
                acceleration = acceleration.to(target_unit)
            else:
                acceleration = np.asarray(acceleration, dtype=float) * target_unit
        if np.shape(acceleration) != np.shape(fluid.rho):
            raise ValueError(
                "Gravity acceleration shape %s does not match fluid state shape %s"
                % (np.shape(acceleration), np.shape(fluid.rho))
            )
        fluid.Mom += fluid.rho * acceleration * mesh.vol * dt
        fluid.Energy += fluid.rho * fluid.vel * acceleration * mesh.vol * dt
        self._zero_spherical_center_momentum(mesh, fluid)
        return 1

    def AdvectIonizationFraction(self, dt, mesh, fluid, par, old_mass, mass_flux):
        """Advect the chemistry fraction consistently with the mass flux."""
        return rtc.advect_ionization_fraction(
            dt,
            mesh,
            fluid,
            par,
            old_mass,
            mass_flux,
        )

    def SourceState(self, mesh, fluid, par):
        """Return a float source state for thermo-chemistry tests."""
        return rtc.source_state(mesh, fluid, par)

    def TracePhotonDensity(self, state, par):
        """Trace a photon field through a float source state."""
        return rrt.trace_photon_density(state, par)

    def IonizationFractionRate(self, state, ngamma, par):
        """Return the chemistry fraction rate for a float source state."""
        return rtc.ionization_fraction_rate(state, ngamma, par)

    def ThermalRate(self, state, ngamma, par):
        """Return thermal source rate for a float source state."""
        return rtc.thermal_rate(state, ngamma, par)

    def GetTimestep(self, state, ngamma, par, remaining_s, dtmax_s):
        """Return a source substep for a float source state."""
        return rtc.get_timestep(
            state,
            ngamma,
            remaining_s,
            dtmax_s,
        )

    def UpdateTemperatureFromEnergy(self, state):
        """Update temperature in a float source state from specific energy."""
        return rtc.update_temperature_from_energy(state)

    def IonizationFractionImplicitUpdate(self, state, ngamma, dt_s, par):
        """Implicitly update the chemistry fraction for a float source state."""
        return rtc.ionization_fraction_implicit_update(
            state,
            ngamma,
            dt_s,
        )

    def ApplyState(self, state, fluid, par):
        """Copy a float source state back to a fluid object."""
        return rtc.apply_state(state, fluid, par)

    def GetSourceTimestepFast(self, mesh, fluid, par, remaining):
        """Return a source substep for RT-coupled heating/chemistry."""
        return rtc.get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining)

    def ApplyThermochemistryFast(self, dt, mesh, fluid, par):
        """Fast source update for RT-coupled thermo-chemistry tests."""
        return rtc.apply_thermochemistry_fast(dt, mesh, fluid, par)


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
        fields = ['rho', 'vel', 'pre']
        if hasattr(fluid, 'xHI'):
            fields.append('xHI')
        if hasattr(fluid, 'ngamma'):
            fields.append('ngamma')
        scalar_fields = [field for field in fields if field != 'vel']

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
            left_values = {
                'rho': fluid.rho[mirror_start:mirror_start+noghost][::-1],
                'vel': -fluid.vel[mirror_start:mirror_start+noghost][::-1],
                'pre': fluid.pre[mirror_start:mirror_start+noghost][::-1],
            }
            if hasattr(fluid, 'xHI'):
                left_values['xHI'] = fluid.xHI[mirror_start:mirror_start+noghost][::-1]
            if hasattr(fluid, 'ngamma'):
                left_values['ngamma'] = fluid.ngamma[mirror_start:mirror_start+noghost][::-1]
            copy_left(left_values)

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
            for attr in scalar_fields:
                quan = getattr(fluid, attr)
                quan[left_ghost] = quan[interior][:noghost][::-1]
                quan[right_ghost] = quan[interior][-noghost:][::-1]
            fluid.vel[left_ghost] = -fluid.vel[interior][:noghost][::-1]
            fluid.vel[right_ghost] = -fluid.vel[interior][-noghost:][::-1]
        elif btype == 'OpenSph':
            # spherical open boundary condition
            # open only at outer boundary
            # symmetric at the center
            # this means zero flux at r=0 
            # imply zero gradient?
            apply_spherical_inner_boundary()
            right_values = {
                'rho': fluid.rho[nolast],
                'vel': fluid.vel[nolast],
                'pre': fluid.pre[nolast],
            }
            if hasattr(fluid, 'xHI'):
                right_values['xHI'] = fluid.xHI[nolast]
            if hasattr(fluid, 'ngamma'):
                right_values['ngamma'] = fluid.ngamma[nolast]
            copy_right(right_values)
        elif btype == 'InflowSph':
            pre_inflow = ru.CalPressure(par.rho_inflow,par.temp_inflow,par.mu_inflow)
            apply_spherical_inner_boundary()
            right_values = {
                'rho': par.rho_inflow,
                'vel': par.vel_inflow,
                'pre': pre_inflow,
            }
            if hasattr(fluid, 'xHI'):
                right_values['xHI'] = getattr(par, 'hydrogen_xHI_inflow', 1.0)
            if hasattr(fluid, 'ngamma'):
                right_values['ngamma'] = photon_number_density(
                    getattr(par, 'hydrogen_ngamma_inflow', 0.0)
                ).to_value(fluid.ngamma.units)
            copy_right(right_values)
        elif btype == 'OutflowSph':
            pre_outflow = ru.CalPressure(par.rho_outflow,par.temp_outflow,par.mu_outflow)
            left_values = {
                'rho': par.rho_outflow,
                'vel': par.vel_outflow,
                'pre': pre_outflow,
            }
            if hasattr(fluid, 'xHI'):
                left_values['xHI'] = getattr(par, 'hydrogen_xHI_outflow', 1.0)
            if hasattr(fluid, 'ngamma'):
                left_values['ngamma'] = photon_number_density(
                    getattr(par, 'hydrogen_ngamma_outflow', 0.0)
                ).to_value(fluid.ngamma.units)
            copy_left(left_values)
            right_values = {
                'rho': fluid.rho[nolast],
                'vel': fluid.vel[nolast],
                'pre': fluid.pre[nolast],
            }
            if hasattr(fluid, 'xHI'):
                right_values['xHI'] = fluid.xHI[nolast]
            if hasattr(fluid, 'ngamma'):
                right_values['ngamma'] = fluid.ngamma[nolast]
            copy_right(right_values)
        else:
            raise ValueError('Boundary condition unknown: %s'%btype) 
        
    
    def GetTimeStep(self, mesh, fluid, par, CFL=None):
        """Return a CFL-limited timestep."""
        if CFL is None:
            CFL = par.CFL
        fluid.SetSoundSpeed()
        vsignal = np.absolute(fluid.vel) + fluid.cs
        xdelta = mesh.xdelta
        if xdelta.shape != vsignal.shape:
            interior = self._interior_slice(par)
            if xdelta[interior].shape == vsignal.shape:
                xdelta = xdelta[interior]
            elif vsignal[interior].shape == xdelta.shape:
                vsignal = vsignal[interior]
        dt_array = self._safe_divide(CFL * xdelta, vsignal)
        dtmax = par.dtmax.to(dt_array.units) if hasattr(par.dtmax, "to") else par.dtmax
        dt_array = np.where(vsignal != 0.0, dt_array, dtmax)
        dt = np.amin(dt_array)
        fluid.vsignal = vsignal
        self.dt = dt
        if np.isnan(np.asarray(dt)):
            print('vsignal', vsignal)
            print('fluid.vel', fluid.vel)
            print('fluid.cs', fluid.cs)
            raise Exception(" time step is nan")
        if dt < par.dtmin:
            raise ValueError(
                " time step %.2e smaller than the minimum time step %.2e"
                % (dt, par.dtmin)
            )
        if dt > par.dtmax:
            dt = par.dtmax
        return dt
