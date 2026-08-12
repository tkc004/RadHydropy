"""Finite-volume hydrodynamics solver operations."""

import radhydropy.utils as ru
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
import radhydropy.gravity as rg
from radhydropy.constants import DEFAULT_SIGMA_GAMMA
from radhydropy.units import (
    CGS_AREA_UNIT,
    CGS_MASS_DENSITY_UNIT,
    CGS_NUMBER_DENSITY_UNIT,
    CGS_PHOTON_FLUX_UNIT,
    CGS_RATE_UNIT,
    CGS_VOLUME_UNIT,
    code_unit_scales,
    _as_cgs_float,
    _code_units,
    code_quantity_to_cgs,
    photon_number_density,
)
import numpy as np
from types import SimpleNamespace
import unyt
from radhydropy.arrays import as_named_array

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
        return rtc.thermochemistry_enabled(fluid, par)

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
        code_units = _code_units(par)
        scales = code_unit_scales(code_units)
        if not hasattr(fluid, 'ngamma'):
            fluid.ngamma = np.zeros(np.shape(fluid.rho), dtype=float)
        interior = self._interior_slice(par)
        boundary = np.asarray(mesh.boundary[interior.start : interior.stop + 1], dtype=float)
        volume = np.asarray(mesh.vol[interior], dtype=float)
        submesh = SimpleNamespace(
            coordsys=getattr(mesh, 'coordsys', 'cartesian'),
            boundary=boundary * scales['length_cm'],
            vol=volume * scales['volume_cm3'],
        )
        if hasattr(mesh, 'area'):
            submesh.area = np.asarray(mesh.area[interior], dtype=float) * scales['area_cm2']
        group_edges_eV = getattr(par, 'radiation_group_edges_eV', None)
        if group_edges_eV is not None:
            sigma_groups = getattr(par, 'radiation_group_sigma_gamma', None)
            if sigma_groups is None:
                sigma_groups = getattr(par, 'hydrogen_sigma_gamma', DEFAULT_SIGMA_GAMMA)
            boundary_groups = getattr(
                par,
                'radiative_transfer_boundary_flux_groups',
                getattr(par, 'radiative_transfer_boundary_flux', 0.0),
            )
            source_groups = getattr(
                par,
                'radiative_transfer_source_photon_rate_groups',
                getattr(par, 'radiative_transfer_source_photon_rate', 0.0),
            )
            if hasattr(sigma_groups, 'to_value'):
                sigma_groups = sigma_groups.to_value(CGS_AREA_UNIT)
            else:
                sigma_groups = code_quantity_to_cgs(
                    sigma_groups,
                    code_units,
                    'area_cm2',
                )
            if hasattr(boundary_groups, 'to_value'):
                boundary_groups = boundary_groups.to_value(CGS_PHOTON_FLUX_UNIT)
            else:
                boundary_groups = code_quantity_to_cgs(
                    boundary_groups,
                    code_units,
                    'photon_flux_per_cm2_s',
                )
            if hasattr(source_groups, 'to_value'):
                source_groups = source_groups.to_value(CGS_RATE_UNIT)
            else:
                source_groups = code_quantity_to_cgs(
                    source_groups,
                    code_units,
                    'photon_rate_per_s',
                )
            result = rrt.trace_long_characteristics(
                submesh,
                np.asarray(fluid.rho[interior], dtype=float) * scales['density_g_cm3'],
                np.asarray(fluid.xHI[interior], dtype=float),
                hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
                sigma_gamma=sigma_groups,
                boundary_flux=boundary_groups,
                source_photon_rate=source_groups,
                direction=getattr(par, 'radiative_transfer_direction', 1),
                coordsys=getattr(mesh, 'coordsys', 'cartesian'),
                group_edges_eV=group_edges_eV,
            )
            fluid.ngamma[:, interior] = (
                np.asarray(result.cell_photon_density, dtype=float)
                / scales['number_density_cm3']
            )
            return result
        sigma_value = getattr(par, 'hydrogen_sigma_gamma', DEFAULT_SIGMA_GAMMA)
        boundary_value = getattr(par, 'radiative_transfer_boundary_flux', 0.0)
        source_value = getattr(par, 'radiative_transfer_source_photon_rate', 0.0)
        if hasattr(sigma_value, 'to_value'):
            sigma_gamma_cm2 = _as_cgs_float(sigma_value, CGS_AREA_UNIT)
        else:
            sigma_gamma_cm2 = code_quantity_to_cgs(
                sigma_value,
                code_units,
                'area_cm2',
            )
        if hasattr(boundary_value, 'to_value'):
            boundary_flux = _as_cgs_float(boundary_value, CGS_PHOTON_FLUX_UNIT)
        else:
            boundary_flux = code_quantity_to_cgs(
                boundary_value,
                code_units,
                'photon_flux_per_cm2_s',
            )
        if hasattr(source_value, 'to_value'):
            source_photon_rate = _as_cgs_float(source_value, CGS_RATE_UNIT)
        else:
            source_photon_rate = code_quantity_to_cgs(
                source_value,
                code_units,
                'photon_rate_per_s',
            )
        result = rrt.trace_long_characteristics(
            submesh,
            np.asarray(fluid.rho[interior], dtype=float) * scales['density_g_cm3'],
            np.asarray(fluid.xHI[interior], dtype=float),
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
            sigma_gamma=sigma_gamma_cm2,
            boundary_flux=boundary_flux,
            source_photon_rate=source_photon_rate,
            direction=getattr(par, 'radiative_transfer_direction', 1),
            coordsys=getattr(mesh, 'coordsys', 'cartesian'),
        )
        fluid.ngamma[interior] = (
            np.asarray(result.cell_photon_density, dtype=float)
            / scales['number_density_cm3']
        )
        return result

    def _spherical_center_cell_index(self, mesh):
        if getattr(mesh, 'coordsys', None) != 'spherical' or not hasattr(mesh, 'boundary'):
            return None
        boundary = np.asarray(mesh.boundary, dtype=float)
        origin_faces = np.where(boundary[:-1] == 0.0)[0]
        if len(origin_faces) > 0:
            return int(origin_faces[0])
        origin_cells = np.where(
            np.logical_and(boundary[:-1] < 0.0, boundary[1:] > 0.0)
        )[0]
        if len(origin_cells) > 0:
            return int(origin_cells[0])
        return None

    def _spherical_origin_face_index(self, mesh):
        if getattr(mesh, 'coordsys', None) != 'spherical' or not hasattr(mesh, 'boundary'):
            return None
        boundary = np.asarray(mesh.boundary, dtype=float)
        origin_faces = np.where(boundary[:-1] == 0.0)[0]
        if len(origin_faces) > 0:
            return int(origin_faces[0])
        return None

    def _zero_spherical_origin_flux(self, mesh, fluid):
        origin_face = self._spherical_origin_face_index(mesh)
        if origin_face is None:
            return
        fluid.Mass.flux[origin_face] = 0.0
        fluid.Mom.flux[origin_face] = 0.0
        fluid.Energy.flux[origin_face] = 0.0

    def _zero_spherical_center_momentum(self, mesh, fluid):
        center_cell = self._spherical_center_cell_index(mesh)
        if center_cell is None:
            return
        fluid.Mom[center_cell] = 0.0

    def _boundary_field_names(self, fluid):
        fields = ['rho', 'vel', 'pre']
        if hasattr(fluid, 'xHI'):
            fields.append('xHI')
        if hasattr(fluid, 'ngamma'):
            fields.append('ngamma')
        return fields

    def _copy_boundary_state(self, fluid, target_slice, values):
        for attr, value in values.items():
            target = getattr(fluid, attr)
            if attr == 'ngamma' and np.ndim(target) == 2:
                value_array = np.asarray(value)
                if value_array.ndim == 1:
                    value_array = value_array[:, None]
                target[:, target_slice] = value_array
            else:
                target[target_slice] = value

    def _boundary_state(
        self,
        fluid,
        source,
        include_velocity=True,
        negate_velocity=False,
        reverse=False,
    ):
        state = {
            'rho': fluid.rho[source],
            'pre': fluid.pre[source],
        }
        if include_velocity:
            velocity = fluid.vel[source]
            state['vel'] = -velocity if negate_velocity else velocity
        if hasattr(fluid, 'xHI'):
            state['xHI'] = fluid.xHI[source]
        if hasattr(fluid, 'ngamma'):
            if np.ndim(fluid.ngamma) == 2:
                state['ngamma'] = fluid.ngamma[:, source]
            else:
                state['ngamma'] = fluid.ngamma[source]
        if reverse:
            for key, value in list(state.items()):
                state[key] = value[::-1]
        return state

    def _to_code_number_density(self, value, scales):
        density = np.asarray(photon_number_density(value).to_value(unyt.cm**-3), dtype=float)
        if scales is None:
            return density
        return density / scales['number_density_cm3']

    def _apply_periodic_boundary(self, fluid, interior, left_ghost, right_ghost, noghost):
        fields = self._boundary_field_names(fluid)
        for attr in fields:
            quan = getattr(fluid, attr)
            if attr == 'ngamma' and np.ndim(quan) == 2:
                quan[:, left_ghost] = quan[:, interior][-noghost:]
                quan[:, right_ghost] = quan[:, interior][:noghost]
            else:
                quan[left_ghost] = quan[interior][-noghost:]
                quan[right_ghost] = quan[interior][:noghost]

    def _apply_open_boundary(self, fluid, first, nolast, left_ghost, right_ghost):
        fields = self._boundary_field_names(fluid)
        for attr in fields:
            quan = getattr(fluid, attr)
            if attr == 'ngamma' and np.ndim(quan) == 2:
                quan[:, left_ghost] = quan[:, first]
                quan[:, right_ghost] = quan[:, nolast]
            else:
                quan[left_ghost] = quan[first]
                quan[right_ghost] = quan[nolast]

    def _apply_reflecting_boundary(self, fluid, interior, left_ghost, right_ghost, noghost):
        for attr in ('rho', 'pre'):
            quan = getattr(fluid, attr)
            quan[left_ghost] = quan[interior][:noghost][::-1]
            quan[right_ghost] = quan[interior][-noghost:][::-1]
        fluid.vel[left_ghost] = -fluid.vel[interior][:noghost][::-1]
        fluid.vel[right_ghost] = -fluid.vel[interior][-noghost:][::-1]

    def _apply_spherical_inner_boundary(self, mesh, fluid, first, noghost):
        mirror_start = first
        if mesh is not None and hasattr(mesh, 'boundary'):
            boundary_units = getattr(mesh.boundary, 'units', None)
            origin = 0.0 * boundary_units if boundary_units is not None else 0.0
            if mesh.boundary[first] < origin and mesh.boundary[first+1] > origin:
                mirror_start = first + 1
        left_state = self._boundary_state(
            fluid,
            slice(mirror_start, mirror_start + noghost),
            negate_velocity=True,
            reverse=True,
        )
        self._copy_boundary_state(fluid, slice(0, noghost), left_state)

    def _apply_open_spherical_boundary(
        self,
        mesh,
        fluid,
        par,
        scales,
        first,
        nolast,
        left_ghost,
        right_ghost,
        noghost,
    ):
        self._apply_spherical_inner_boundary(mesh, fluid, first, noghost)
        right_state = self._boundary_state(fluid, nolast)
        self._copy_boundary_state(fluid, right_ghost, right_state)

    def _apply_inflow_spherical_boundary(
        self,
        mesh,
        fluid,
        par,
        scales,
        first,
        nolast,
        left_ghost,
        right_ghost,
        noghost,
    ):
        self._apply_spherical_inner_boundary(mesh, fluid, first, noghost)
        right_state = {
            'rho': par.rho_inflow,
            'vel': par.vel_inflow,
            'pre': fluid.eos.pressure(par.rho_inflow, par.temp_inflow, par.mu_inflow),
        }
        if hasattr(fluid, 'xHI'):
            right_state['xHI'] = getattr(par, 'hydrogen_xHI_inflow', 1.0)
        if hasattr(fluid, 'ngamma'):
            right_state['ngamma'] = self._to_code_number_density(
                getattr(par, 'hydrogen_ngamma_inflow', 0.0),
                scales,
            )
        self._copy_boundary_state(fluid, right_ghost, right_state)

    def _apply_outflow_spherical_boundary(
        self,
        mesh,
        fluid,
        par,
        scales,
        first,
        nolast,
        left_ghost,
        right_ghost,
        noghost,
    ):
        left_state = {
            'rho': par.rho_outflow,
            'vel': par.vel_outflow,
            'pre': fluid.eos.pressure(par.rho_outflow, par.temp_outflow, par.mu_outflow),
        }
        if hasattr(fluid, 'xHI'):
            left_state['xHI'] = getattr(par, 'hydrogen_xHI_outflow', 1.0)
        if hasattr(fluid, 'ngamma'):
            left_state['ngamma'] = self._to_code_number_density(
                getattr(par, 'hydrogen_ngamma_outflow', 0.0),
                scales,
            )
        self._copy_boundary_state(fluid, left_ghost, left_state)
        right_state = self._boundary_state(fluid, nolast)
        self._copy_boundary_state(fluid, right_ghost, right_state)

    def SetPrimitive(self, mesh, fluid, verbose=None):
        """Update primitive variables from conserved quantities."""
        if verbose is None:
            verbose = 0
        vol = mesh.vol
        fluid.rho = as_named_array(self._safe_divide(fluid.Mass, vol))
        fluid.vel = as_named_array(self._safe_divide(fluid.Mom, fluid.Mass))
        energy_density = as_named_array(self._safe_divide(fluid.Energy, vol))
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
    
    def SetConserved(self, mesh, fluid, verbose=None):
        """Update conserved mass, momentum, and energy from primitive variables."""
        if verbose is None:
            verbose = 0
        vol = mesh.vol
        fluid.Mass = as_named_array(fluid.rho * vol)
        fluid.Mom = as_named_array(fluid.rho * fluid.vel * vol)
        fluid.Energy = as_named_array(fluid.eos.total_energy_density(
            fluid.rho,
            fluid.vel,
            fluid.pre,
        ) * vol)
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
        if order == 0 or order == 1:
            fluid.rho.R = fluid.rho
            fluid.rho.L = ru.periodic_roll(fluid.rho, 1)
            fluid.vel.R = fluid.vel
            fluid.vel.L = ru.periodic_roll(fluid.vel, 1)
            fluid.pre.R = fluid.pre
            fluid.pre.L = ru.periodic_roll(fluid.pre, 1)
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
        
    def SetInterFaceFlux(self,mesh,fluid,boundcond, method='Rusanov',verbose=None, order=0):
        """Set interface fluxes using GLF or Rusanov numerical fluxes."""
        if verbose is None:
            verbose = 0
        if method=='GLF' or method=='Rusanov':
            if method=='GLF':
                # Global Lax Friedrich scheme
                # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)  
                # simple to implement but very diffusive
                # calculate cmax
                fluid.cmax = mesh.xdelta / np.amin(self.dt)
            elif method=='Rusanov':
                # Local Lax Friedrich schem
                # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)  
                # simple to implement but less diffusive
                fluid.cmax = np.maximum(fluid.vsignal, ru.periodic_roll(fluid.vsignal, 1))
            
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
        area = mesh.area
        df_Mass = fluid.Mass.flux * area - ru.periodic_roll(fluid.Mass.flux * area, -1)
        df_Mom = fluid.Mom.flux * area - ru.periodic_roll(fluid.Mom.flux * area, -1)
        df_Energy = fluid.Energy.flux * area - ru.periodic_roll(fluid.Energy.flux * area, -1)
        if getattr(mesh, 'coordsys', None) == 'spherical':
            # Spherical momentum needs the geometric pressure term from the
            # changing face area, not just the flux divergence.
            area_right = ru.periodic_roll(area, -1)
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
        if not (
            getattr(par, "externalgravity", False)
            or getattr(par, "selfgravity", False)
            or getattr(par, "dark_matter", None) is not None
        ):
            return None
        return rg.Gravity(
            selfgravity=getattr(par, "selfgravity", False),
            externalgravity=getattr(par, "externalgravity", False),
            potential=getattr(par, "gravity_potential", None),
            coordinate=getattr(par, "gravity_coordinate", None),
            acceleration=getattr(par, "gravity_acceleration", None),
            code_units=getattr(par, "CodeUnits", None),
            selfgravity_softening=getattr(par, "selfgravity_softening", 0.0),
            selfgravity_boundary_acceleration=getattr(
                par, "selfgravity_boundary_acceleration", 0.0
            ),
            dark_matter=getattr(par, "dark_matter", None),
        )

    def ApplyGravity(self, dt, mesh, fluid, par):
        """Apply the combined external and gas self-gravity source update."""
        gravity = self._gravity_model(par)
        if gravity is None:
            return 0
        crossing_safety_factor = getattr(par, "dark_matter_crossing_safety_factor", 0.1)
        if getattr(gravity, "dark_matter", None) is not None:
            gravity.advance_dark_matter(
                dt,
                mesh,
                fluid.rho,
                par,
                crossing_safety_factor=crossing_safety_factor,
            )
        acceleration = gravity.acceleration_on_mesh(mesh, rho=fluid.rho, par=par)
        code_units = getattr(par, "CodeUnits", None)
        if code_units is not None:
            target_unit = code_units.length_unit / code_units.time_unit**2
            if hasattr(acceleration, "to_value"):
                acceleration = np.asarray(acceleration.to_value(target_unit), dtype=float)
            else:
                acceleration = np.asarray(acceleration, dtype=float)
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
        code_units = getattr(par, 'CodeUnits', None)
        scales = code_unit_scales(code_units)
        noghost = par.noghost
        nogrid = par.nogrid
        nolast = noghost + nogrid -1
        first = noghost
        right_start = noghost + nogrid
        interior = slice(first, right_start)
        left_ghost = slice(0, noghost)
        right_ghost = slice(right_start, right_start + noghost)
        if btype == 'Periodic':
            self._apply_periodic_boundary(fluid, interior, left_ghost, right_ghost, noghost)
        elif btype == 'Open':
            # open boundary condition does not mean the gradient is zero.
            self._apply_open_boundary(fluid, first, nolast, left_ghost, right_ghost)
        elif btype == 'Reflecting':
            self._apply_reflecting_boundary(fluid, interior, left_ghost, right_ghost, noghost)
        elif btype == 'OpenSph':
            # spherical open boundary condition
            # open only at outer boundary
            # symmetric at the center
            # this means zero flux at r=0
            self._apply_open_spherical_boundary(
                mesh,
                fluid,
                par,
                scales,
                first,
                nolast,
                left_ghost,
                right_ghost,
                noghost,
            )
        elif btype == 'InflowSph':
            self._apply_inflow_spherical_boundary(
                mesh,
                fluid,
                par,
                scales,
                first,
                nolast,
                left_ghost,
                right_ghost,
                noghost,
            )
        elif btype == 'OutflowSph':
            self._apply_outflow_spherical_boundary(
                mesh,
                fluid,
                par,
                scales,
                first,
                nolast,
                left_ghost,
                right_ghost,
                noghost,
            )
        else:
            raise ValueError('Boundary condition unknown: %s'%btype) 
        
    
    def GetTimeStep(self, mesh, fluid, par, CFL=None):
        """Return a CFL-limited timestep in the active time coordinate."""
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
        dtmax = float(np.asarray(par.dtmax, dtype=float))
        dt_array = np.where(vsignal != 0.0, dt_array, dtmax)
        dt = np.amin(dt_array)
        fluid.vsignal = vsignal
        self.dt = dt
        if np.isnan(np.asarray(dt)):
            print('vsignal', vsignal)
            print('fluid.vel', fluid.vel)
            print('fluid.cs', fluid.cs)
            raise Exception(" time step is nan")
        if dt < float(np.asarray(par.dtmin, dtype=float)):
            raise ValueError(
                " time step %.2e smaller than the minimum time step %.2e"
                % (dt, par.dtmin)
            )
        if dt > dtmax:
            dt = dtmax
        if getattr(par, 'verbose', 0) >= 1:
            min_index = int(np.argmin(dt_array))
            print(
                '[hydro dt] t=%s dt=%s idx=%d xdelta=%s vel=%s cs=%s vsignal=%s dtmin=%s dtmax=%s'
                % (
                    fluid.time,
                    dt,
                    min_index,
                    xdelta[min_index],
                    fluid.vel[min_index],
                    fluid.cs[min_index],
                    vsignal[min_index],
                    par.dtmin,
                    par.dtmax,
                )
            )
        return dt
