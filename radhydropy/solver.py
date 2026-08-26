"""Finite-volume hydrodynamics solver operations."""

import radhydropy.utils as ru
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
import radhydropy.gravity as rg
from radhydropy.constants import DEFAULT_SIGMA_GAMMA, SPEED_OF_LIGHT_CGS
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

    @staticmethod
    def _hydrostatic_core_enabled(par):
        return str(getattr(par, 'gas_core_model', 'none')).lower() in (
            'hydrostatic', 'hydrostatic_fixed', 'fixed_hydrostatic',
        )

    def InitializeHydrostaticCore(self, mesh, fluid, par):
        """Initialize an optional fixed, pressure-supported central core.

        The core is a deliberately simple subgrid model: its cell-centred
        primitive state is retained as a pressure-bearing hydrostatic core,
        while the resolved halo evolves outside ``gas_core_radius``.  It is
        not a sink and does not remove gas from the calculation.
        """
        if not self._hydrostatic_core_enabled(par):
            return
        if getattr(mesh, 'coordsys', None) != 'spherical':
            raise ValueError('gas_core_model requires a spherical mesh')
        radius = getattr(par, 'gas_core_radius', None)
        if radius is None or float(radius) <= 0.0:
            raise ValueError('gas_core_radius must be positive for gas_core_model')
        first = int(par.noghost)
        last = first + int(par.nogrid)
        coordinate = np.asarray(mesh.coordinate[first:last], dtype=float)
        core_local = coordinate < float(radius)
        if not np.any(core_local) or np.all(core_local):
            raise ValueError(
                'gas_core_radius must contain at least one, but not all, '
                'resolved cells'
            )
        core = np.zeros(len(mesh.coordinate), dtype=bool)
        core[first:last] = core_local
        core_indices = np.flatnonzero(core)
        state = {
            'core_mask': core,
            'core_indices': core_indices,
            'core_last': int(core_indices[-1]),
        }
        for name in ('rho', 'vel', 'temp', 'mu', 'pre', 'xHI',
                     'xHeI', 'xHeII', 'xHeIII'):
            if hasattr(fluid, name):
                state[name] = np.asarray(getattr(fluid, name)[core], dtype=float).copy()
        fluid._hydrostatic_core = state
        par._hydrostatic_core_mask = core
        par._hydrostatic_core_face = int(core_indices[-1] + 1)

    def ApplyHydrostaticCore(self, mesh, fluid, par):
        """Restore the fixed core state before a resolved-halo update."""
        state = getattr(fluid, '_hydrostatic_core', None)
        if state is None:
            return
        core = state['core_mask']
        for name in ('rho', 'vel', 'temp', 'mu', 'pre', 'xHI',
                     'xHeI', 'xHeII', 'xHeIII'):
            if name in state and hasattr(fluid, name):
                values = np.asarray(getattr(fluid, name), dtype=float).copy()
                values[core] = state[name]
                setattr(fluid, name, as_named_array(values))
        # A fixed core is hydrostatic and has no resolved radial motion.
        fluid.vel[core] = 0.0

    def _apply_hydrostatic_core_flux(self, fluid, par):
        """Close the resolved halo with a pressure-bearing, no-mass-flux core."""
        face = getattr(par, '_hydrostatic_core_face', None)
        state = getattr(fluid, '_hydrostatic_core', None)
        if face is None or state is None:
            return
        core_last = state['core_last']
        # The core is fixed-mass: pressure acts on the halo, but gas, energy,
        # and radial momentum do not cross the core/halo interface.
        fluid.Mass.flux[face] = 0.0
        fluid.Energy.flux[face] = 0.0
        fluid.Mom.flux[face] = fluid.pre[core_last]

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
                quan[:, left_ghost] = quan[:, interior][:, -noghost:]
                quan[:, right_ghost] = quan[:, interior][:, :noghost]
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

    def SetPrimitive(self, mesh, fluid, par=None, verbose=None):
        """Update primitive variables from conserved quantities."""
        if verbose is None:
            verbose = 0
        vol = mesh.vol
        rho = np.asarray(self._safe_divide(fluid.Mass, vol), dtype=float)
        active = np.isfinite(rho) & (rho > 0.0)
        fluid.active = active
        rho = np.where(active, rho, 0.0)

        mass = np.asarray(fluid.Mass, dtype=float)
        momentum = np.asarray(fluid.Mom, dtype=float)
        vel = np.zeros_like(rho)
        valid_mass = active & np.isfinite(mass) & (mass > 0.0)
        vel[valid_mass] = momentum[valid_mass] / mass[valid_mass]

        energy_density = np.zeros_like(rho)
        valid_volume = active & np.isfinite(vol) & (vol > 0.0)
        energy_density[valid_volume] = (
            np.asarray(fluid.Energy, dtype=float)[valid_volume]
            / np.asarray(vol, dtype=float)[valid_volume]
        )
        fluid.rho = as_named_array(rho)
        fluid.vel = as_named_array(vel)
        density_floor = self._cfl_density_floor(par)
        numerical_vacuum = active & (rho <= density_floor)
        fluid.vel[numerical_vacuum] = 0.0
        energy_density = as_named_array(energy_density)
        total_pressure = fluid.eos.pressure_from_conserved(
            fluid.rho,
            fluid.vel,
            energy_density,
            temp=getattr(fluid, 'temp', None),
            mu=getattr(fluid, 'mu', None),
        )
        fluid.pre = total_pressure
        if self._dual_energy_enabled(par) and hasattr(fluid, 'InternalEnergy'):
            internal_density = np.zeros_like(rho)
            internal_density[valid_volume] = (
                np.asarray(fluid.InternalEnergy, dtype=float)[valid_volume]
                / np.asarray(vol, dtype=float)[valid_volume]
            )
            dual_pressure = (fluid.eos.gamma - 1.0) * internal_density
            total_thermal = energy_density - 0.5 * fluid.rho * fluid.vel**2
            switch = float(getattr(par, 'dual_energy_switch', 1.0e-3))
            use_dual = (
                np.isfinite(internal_density)
                & (internal_density > 0.0)
                & np.isfinite(dual_pressure)
                & (
                    total_thermal
                    <= switch * np.maximum(np.abs(energy_density), 1.0e-300)
                )
            )
            fluid.pre[use_dual] = dual_pressure[use_dual]
        fluid.rho[~active] = 0.0
        fluid.vel[~active] = 0.0
        invalid_pressure = np.logical_or(fluid.pre <= 0.0, np.isnan(fluid.pre))
        temperature_floor = getattr(par, 'hydro_temperature_floor', None)
        if temperature_floor is not None and float(temperature_floor) > 0.0:
            floor_pressure = fluid.eos.pressure(
                fluid.rho,
                float(temperature_floor),
                fluid.mu,
            )
            # Enforce the configured floor for both invalid reconstructions
            # and valid states that have cooled below the physical minimum.
            below_floor = (
                ~numerical_vacuum
                & np.logical_or(invalid_pressure, fluid.pre < floor_pressure)
            )
            fluid.pre[below_floor] = floor_pressure[below_floor]
        else:
            fluid.pre[invalid_pressure & ~numerical_vacuum] = 0.0
        fluid.pre[numerical_vacuum] = 0.0
        if verbose >= 2:
            print('fluid.rho',fluid.rho)
            print('fluid.vel',fluid.vel)
            print('fluid.pre',fluid.pre)            
    
    def SetConserved(self, mesh, fluid, verbose=None):
        """Update conserved mass, momentum, and energy from primitive variables."""
        if verbose is None:
            verbose = 0
        par = getattr(mesh, '_par', None)
        density_floor = self._cfl_density_floor(par)
        dual_energy = self._dual_energy_enabled(par)
        old_internal = (
            np.asarray(fluid.InternalEnergy, dtype=float).copy()
            if dual_energy and hasattr(fluid, 'InternalEnergy')
            else None
        )
        old_total_energy = (
            np.asarray(fluid.Energy, dtype=float).copy()
            if dual_energy and hasattr(fluid, 'Energy')
            else None
        )
        old_conserved = None
        if density_floor > 0.0 and all(
            hasattr(fluid, name) for name in ('Mass', 'Mom', 'Energy')
        ):
            density = np.asarray(fluid.rho, dtype=float)
            inactive = np.isfinite(density) & (density <= density_floor)
            old_conserved = (
                inactive,
                np.asarray(fluid.Mass, dtype=float).copy(),
                np.asarray(fluid.Mom, dtype=float).copy(),
                np.asarray(fluid.Energy, dtype=float).copy(),
            )
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
        if old_total_energy is not None:
            first = int(getattr(par, 'noghost', 0))
            count = int(getattr(par, 'nogrid', len(fluid.Energy) - first))
            fluid.Energy[first:first + count] = old_total_energy[first:first + count]
        if dual_energy and getattr(fluid.eos, 'is_polytropic', False):
            internal = np.asarray(
                fluid.eos.thermal_energy_density(fluid.pre) * vol,
                dtype=float,
            )
            if old_internal is not None:
                first = int(getattr(par, 'noghost', 0))
                count = int(getattr(par, 'nogrid', len(internal) - first))
                internal[first:first + count] = old_internal[first:first + count]
            fluid.InternalEnergy = as_named_array(np.maximum(internal, 0.0))
        if old_conserved is not None:
            inactive, old_mass, old_mom, old_energy = old_conserved
            fluid.Mass[inactive] = old_mass[inactive]
            fluid.Mom[inactive] = old_mom[inactive]
            fluid.Energy[inactive] = old_energy[inactive]
        if verbose >= 2:
            print('fluid.Mass',fluid.Mass)
            print('fluid.Mom',fluid.Mom)
            print('fluid.Energy',fluid.Energy)
        
        
    def SetGradient(self, mesh, fluid):
        """Calculate centered gradients for density, velocity, and pressure."""
        xdelta = mesh.xdelta
        fluid.rho.grad = ru.CalGradient(fluid.rho, xdelta)
        fluid.vel.grad = ru.CalGradient(fluid.vel, xdelta)
        fluid.pre.grad = ru.CalGradient(fluid.pre, xdelta)

    @staticmethod
    def _cfl_density_floor(par):
        return max(0.0, float(np.asarray(
            getattr(par, 'cfl_density_floor', 0.0), dtype=float
        )))

    @staticmethod
    def _dual_energy_enabled(par):
        return bool(getattr(par, 'dual_energy', False))

    def _apply_low_density_face_mask(self, fluid, par, order):
        """Make below-floor reconstructed states vacuum-safe.

        This is a numerical mask only: cell-centred conserved mass and density
        remain unchanged.  It prevents a tiny positive density carrying a
        large pressure from determining the CFL step or Riemann flux.
        """
        density_floor = self._cfl_density_floor(par)
        if density_floor <= 0.0:
            return
        for density, velocity, pressure in (
            (fluid.rho.R, fluid.vel.R, fluid.pre.R),
            (fluid.rho.L, fluid.vel.L, fluid.pre.L),
        ):
            inactive = ~np.isfinite(density) | (density <= density_floor)
            density[inactive] = 0.0
            velocity[inactive] = 0.0
            pressure[inactive] = 0.0
        if order == 1:
            for density, velocity, pressure in (
                (fluid.rho.R.first, fluid.vel.R.first, fluid.pre.R.first),
                (fluid.rho.L.first, fluid.vel.L.first, fluid.pre.L.first),
            ):
                inactive = ~np.isfinite(density) | (density <= density_floor)
                density[inactive] = 0.0
                velocity[inactive] = 0.0
                pressure[inactive] = 0.0
        
        
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
            # Keep face states independent from cell-centred primitives: the
            # low-density numerical-vacuum mask may modify them in place.
            fluid.rho.R = as_named_array(np.asarray(fluid.rho, dtype=float).copy())
            fluid.rho.L = ru.periodic_roll(fluid.rho, 1)
            fluid.vel.R = as_named_array(np.asarray(fluid.vel, dtype=float).copy())
            fluid.vel.L = ru.periodic_roll(fluid.vel, 1)
            fluid.pre.R = as_named_array(np.asarray(fluid.pre, dtype=float).copy())
            fluid.pre.L = ru.periodic_roll(fluid.pre, 1)
            if order == 1:
                self.SetGradient(mesh, fluid)
                fluid.rho.R.first, fluid.rho.L.first = ru.extrapolateToFace(fluid.rho, mesh.boundary, fluid.rho.grad, order=1)
                fluid.vel.R.first, fluid.vel.L.first = ru.extrapolateToFace(fluid.vel, mesh.boundary, fluid.vel.grad, order=1)
                fluid.pre.R.first, fluid.pre.L.first = ru.extrapolateToFace(fluid.pre, mesh.boundary, fluid.pre.grad, order=1)
            self._apply_low_density_face_mask(
                fluid, getattr(mesh, '_par', None), order
            )
            self._apply_cosmological_background_boundary_face(mesh, fluid, order)
        else:
            raise ValueError('order unknown: %s'%order)

    def _apply_cosmological_background_boundary_face(self, mesh, fluid, order):
        """Replace the outer face states with the homogeneous EdS state."""
        par = getattr(mesh, '_par', None)
        if par is None or not getattr(
            par, 'cosmological_background_boundary_reconstruction', False
        ):
            return
        if getattr(par, 'boundcond', None) != 'InflowSph':
            return
        first = int(par.noghost)
        outer_face = first + int(par.nogrid)
        if outer_face >= len(fluid.rho.R):
            return
        rho_background = float(par.rho_inflow)
        velocity_background = float(par.vel_inflow)
        pressure_background = float(
            fluid.eos.pressure(
                rho_background,
                float(par.temp_inflow),
                float(par.mu_inflow),
            )
        )
        for quantity, value in (
            (fluid.rho, rho_background),
            (fluid.vel, velocity_background),
            (fluid.pre, pressure_background),
        ):
            quantity.R[outer_face] = value
            quantity.L[outer_face] = value
            if order == 1:
                quantity.R.first[outer_face] = value
                quantity.L.first[outer_face] = value

    @staticmethod
    def _vacuum_safe_primitive_state(rho, vel, pre):
        """Return a finite, positive primitive state for a face Riemann solve.

        This operates on temporary face states only.  It does not alter the
        cell-centered density or conserved variables, so a vacuum cell can be
        populated by a later hydrodynamic flux update.
        """
        rho_value = np.asarray(rho, dtype=float)
        vel_value = np.asarray(vel, dtype=float)
        pre_value = np.asarray(pre, dtype=float)
        active = np.isfinite(rho_value) & (rho_value > 0.0)
        finite_velocity = np.isfinite(vel_value)
        finite_pressure = np.isfinite(pre_value) & (pre_value > 0.0)
        rho_safe = np.where(active, rho_value, 0.0)
        vel_safe = np.where(active & finite_velocity, vel_value, 0.0)
        pre_safe = np.where(active & finite_pressure, pre_value, 0.0)

        def restore_units(values, original):
            units = getattr(original, 'units', None)
            return values * units if units is not None else as_named_array(values)

        return (
            restore_units(rho_safe, rho),
            restore_units(vel_safe, vel),
            restore_units(pre_safe, pre),
        )


    @staticmethod
    def _hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma):
        """Return an HLLC Euler flux for positive, non-vacuum states.

        The caller supplies the Rusanov flux for vacuum, non-finite, or
        degenerate states.  Keeping that fallback explicit is important for
        the vacuum examples: HLLC's star-state formula is undefined when one
        side has zero density.
        """
        rho_L = np.asarray(rho_L, dtype=float)
        vel_L = np.asarray(vel_L, dtype=float)
        pre_L = np.asarray(pre_L, dtype=float)
        rho_R = np.asarray(rho_R, dtype=float)
        vel_R = np.asarray(vel_R, dtype=float)
        pre_R = np.asarray(pre_R, dtype=float)
        valid = (
            np.isfinite(rho_L) & np.isfinite(vel_L) & np.isfinite(pre_L)
            & np.isfinite(rho_R) & np.isfinite(vel_R) & np.isfinite(pre_R)
            & (rho_L > 0.0) & (rho_R > 0.0)
            & (pre_L > 0.0) & (pre_R > 0.0)
        )
        sound_L = np.zeros_like(rho_L)
        sound_R = np.zeros_like(rho_R)
        with np.errstate(divide='ignore', invalid='ignore'):
            sound_L = np.sqrt(gamma * pre_L / rho_L)
            sound_R = np.sqrt(gamma * pre_R / rho_R)
        valid &= np.isfinite(sound_L) & np.isfinite(sound_R)

        energy_L = pre_L / (gamma - 1.0) + 0.5 * rho_L * vel_L**2
        energy_R = pre_R / (gamma - 1.0) + 0.5 * rho_R * vel_R**2
        flux_L = np.stack((rho_L * vel_L,
                           rho_L * vel_L**2 + pre_L,
                           vel_L * (gamma * pre_L / (gamma - 1.0)
                                    + 0.5 * rho_L * vel_L**2)))
        flux_R = np.stack((rho_R * vel_R,
                           rho_R * vel_R**2 + pre_R,
                           vel_R * (gamma * pre_R / (gamma - 1.0)
                                    + 0.5 * rho_R * vel_R**2)))
        result = 0.5 * (flux_L + flux_R)
        with np.errstate(divide='ignore', invalid='ignore'):
            wave_L = np.minimum(vel_L - sound_L, vel_R - sound_R)
            wave_R = np.maximum(vel_L + sound_L, vel_R + sound_R)
            wave_M = (
                pre_R - pre_L
                + rho_L * vel_L * (wave_L - vel_L)
                - rho_R * vel_R * (wave_R - vel_R)
            ) / (rho_L * (wave_L - vel_L) - rho_R * (wave_R - vel_R))
            pressure_M = pre_L + rho_L * (wave_L - vel_L) * (wave_M - vel_L)
            rho_star_L = rho_L * (wave_L - vel_L) / (wave_L - wave_M)
            rho_star_R = rho_R * (wave_R - vel_R) / (wave_R - wave_M)
            energy_star_L = (
                (wave_L - vel_L) * energy_L - pre_L * vel_L
                + pressure_M * wave_M
            ) / (wave_L - wave_M)
            energy_star_R = (
                (wave_R - vel_R) * energy_R - pre_R * vel_R
                + pressure_M * wave_M
            ) / (wave_R - wave_M)
        star_L = np.stack((rho_star_L, rho_star_L * wave_M, energy_star_L))
        star_R = np.stack((rho_star_R, rho_star_R * wave_M, energy_star_R))
        flux_star_L = flux_L + wave_L * (star_L - np.stack((rho_L, rho_L * vel_L, energy_L)))
        flux_star_R = flux_R + wave_R * (star_R - np.stack((rho_R, rho_R * vel_R, energy_R)))
        left = wave_L >= 0.0
        left_star = (wave_L < 0.0) & (wave_M >= 0.0)
        right_star = (wave_M < 0.0) & (wave_R > 0.0)
        right = wave_R <= 0.0
        result = np.where(left[None, :], flux_L, result)
        result = np.where(left_star[None, :], flux_star_L, result)
        result = np.where(right_star[None, :], flux_star_R, result)
        result = np.where(right[None, :], flux_R, result)
        valid &= np.isfinite(result).all(axis=0)
        return result, valid

    def _interface_fluxes(self, fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method):
        states = fluid.eos.fluxes(rho_L, vel_L, pre_L)
        states_R = fluid.eos.fluxes(rho_R, vel_R, pre_R)
        if method != 'HLLC' or not getattr(fluid.eos, 'is_polytropic', False):
            return tuple(
                ru.CalInterFaceFluxGLF(left, right, qleft, qright, fluid.cmax)
                for left, right, qleft, qright in (
                    (states[0], states_R[0], states[1], states_R[1]),
                    (states[2], states_R[2], states[3], states_R[3]),
                    (states[4], states_R[4], states[5], states_R[5]),
                )
            )
        hllc, valid = self._hllc_flux(
            rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, fluid.eos.gamma
        )
        rusanov = np.stack(tuple(
            ru.CalInterFaceFluxGLF(left, right, qleft, qright, fluid.cmax)
            for left, right, qleft, qright in (
                (states[0], states_R[0], states[1], states_R[1]),
                (states[2], states_R[2], states[3], states_R[3]),
                (states[4], states_R[4], states[5], states_R[5]),
            )
        ))
        flux = np.where(valid[None, :], hllc, rusanov)
        return tuple(flux[index] for index in range(3))

    def SetFluxOnFace(self,fluid,boundcond,order=0,par=None,method='Rusanov'):
        """Calculate mass, momentum, and energy fluxes at interfaces."""
        rho_L, vel_L, pre_L = self._vacuum_safe_primitive_state(
            fluid.rho.L, fluid.vel.L, fluid.pre.L
        )
        rho_R, vel_R, pre_R = self._vacuum_safe_primitive_state(
            fluid.rho.R, fluid.vel.R, fluid.pre.R
        )
        Mass_flux_0, Mom_flux_0, Energy_flux_0 = self._interface_fluxes(
            fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method
        )
        if order==0:
            fluid.Mass.flux, fluid.Mom.flux, fluid.Energy.flux = Mass_flux_0, Mom_flux_0, Energy_flux_0
        elif order==1:
            rho_L, vel_L, pre_L = self._vacuum_safe_primitive_state(
                fluid.rho.L.first, fluid.vel.L.first, fluid.pre.L.first
            )
            rho_R, vel_R, pre_R = self._vacuum_safe_primitive_state(
                fluid.rho.R.first, fluid.vel.R.first, fluid.pre.R.first
            )
            Mass_flux_1, Mom_flux_1, Energy_flux_1 = self._interface_fluxes(
                fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method
            )
            self.SetConservedDensityFlux(fluid)
            limiter = getattr(par, 'flux_limiter', 'minmod') if par is not None else 'minmod'
            fluid.Mass.flux, fluid.philim_Mass = ru.ApplyFluxLimiter(
                fluid.Mass.q, Mass_flux_1, Mass_flux_0, limiter=limiter
            )
            fluid.Mom.flux, fluid.philim_Mom = ru.ApplyFluxLimiter(
                fluid.Mom.q, Mom_flux_1, Mom_flux_0, limiter=limiter
            )
            fluid.Energy.flux, fluid.philim_Energy = ru.ApplyFluxLimiter(
                fluid.Energy.q, Energy_flux_1, Energy_flux_0, limiter=limiter
            )
            # A MUSCL reconstruction is not valid across a vacuum jump.  Use
            # the positivity-safe first-order flux on gas-vacuum faces; this
            # preserves injection into vacuum while retaining order one away
            # from the front.
            floor = self._cfl_density_floor(par)
            vacuum_face = (
                np.asarray(fluid.rho.L, dtype=float) <= floor
            ) | (
                np.asarray(fluid.rho.R, dtype=float) <= floor
            )
            # The update of the gas cell immediately upstream of a vacuum
            # uses both bounding faces.  Limit that complete two-face
            # stencil, otherwise a high-order gas-gas flux can combine with
            # the gas-vacuum flux to leave a pressureless state outside the
            # invariant domain.
            vacuum_face |= np.roll(vacuum_face, -1)
            fluid.Mass.flux[vacuum_face] = Mass_flux_0[vacuum_face]
            fluid.Mom.flux[vacuum_face] = Mom_flux_0[vacuum_face]
            fluid.Energy.flux[vacuum_face] = Energy_flux_0[vacuum_face]
        else:
            raise ValueError('order unknown: %s'%order)

    def _apply_low_density_flux_mask(self, fluid, par):
        """Block hydro flux through numerical-vacuum active cells.

        The CFL and face-state masks prevent a low-density cell from setting
        the timestep, but a Riemann problem between an active neighbor and a
        vacuum state can still produce an outward flux.  Applying that flux
        would inject energy into a cell whose density remains below the
        numerical floor, creating an unphysical temperature spike.  Mask the
        two faces belonging to each below-floor active cell; cell-centred
        conserved quantities are otherwise left untouched.
        """
        density_floor = self._cfl_density_floor(par)
        if density_floor <= 0.0:
            return
        density = np.asarray(fluid.rho, dtype=float)
        first = int(getattr(par, 'noghost', 0))
        count = int(getattr(par, 'nogrid', len(density) - first))
        last = min(first + count, len(density))
        inactive = ~np.isfinite(density) | (density <= density_floor)
        face_mask = np.zeros(len(density), dtype=bool)
        # Keep gas-vacuum interfaces active: their Riemann flux is what fills
        # the vacuum.  Only a vacuum-vacuum interface should be suppressed.
        # Face i joins cell i-1 (the rolled state) to cell i.
        face_mask[first:last] = (
            inactive[first:last]
            & np.roll(inactive, 1)[first:last]
        )
        if not np.any(face_mask):
            return
        for flux in (fluid.Mass.flux, fluid.Mom.flux, fluid.Energy.flux):
            flux[face_mask] = 0.0

    @staticmethod
    def _positive_conserved_state(mass, momentum, energy, mass_floor=0.0,
                                  energy_floor=0.0, relative_tolerance=1.0e-12):
        """Return the invariant-domain admissibility mask for Euler states."""
        mass = np.asarray(mass, dtype=float)
        momentum = np.asarray(momentum, dtype=float)
        energy = np.asarray(energy, dtype=float)
        finite = np.isfinite(mass) & np.isfinite(momentum) & np.isfinite(energy)
        mass_ok = mass >= mass_floor
        internal = np.zeros_like(energy)
        positive_mass = mass > np.maximum(mass_floor, 0.0)
        internal[positive_mass] = (
            energy[positive_mass]
            - 0.5 * momentum[positive_mass]**2 / mass[positive_mass]
        )
        vacuum = ~positive_mass
        internal[vacuum] = energy[vacuum]
        kinetic = np.zeros_like(energy)
        kinetic[positive_mass] = (
            0.5 * momentum[positive_mass]**2 / mass[positive_mass]
        )
        # Cold pressureless states lie on the invariant-domain boundary.  A
        # relative tolerance prevents harmless cancellation in E-K from
        # turning that boundary state into a negative internal energy.
        tolerance = relative_tolerance * np.maximum(
            np.maximum(np.abs(energy), kinetic),
            np.maximum(np.abs(energy_floor), np.finfo(float).tiny),
        )
        return finite & mass_ok & (internal >= energy_floor - tolerance)

    def _positivity_limited_increment(self, fluid, dt, mesh, par,
                                      df_mass, df_mom, df_energy):
        """Limit a hydro increment so density and internal energy stay positive.

        A single factor is used for the complete conservative increment.  This
        is deliberately global: it preserves the finite-volume telescoping
        flux exactly while providing an invariant-domain fallback at vacuum
        interfaces.  The normal update is unchanged when it is admissible.
        """
        if not getattr(par, 'positivity_preserving', True):
            return 1.0
        mass = np.asarray(fluid.Mass, dtype=float)
        momentum = np.asarray(fluid.Mom, dtype=float)
        energy = np.asarray(fluid.Energy, dtype=float)
        dt_value = float(np.asarray(dt, dtype=float))
        mass_floor = max(
            0.0,
            float(np.asarray(getattr(par, 'positivity_density_floor', 0.0))),
        ) * np.asarray(mesh.vol, dtype=float)
        energy_floor = max(
            0.0,
            float(np.asarray(getattr(par, 'positivity_energy_floor', 0.0))),
        ) * np.asarray(mesh.vol, dtype=float)
        # A roundoff-level vacuum may carry a finite momentum after a
        # gas-vacuum Riemann solve.  Remove only that numerical debris before
        # testing the invariant domain; resolved positive-density cells are
        # never repaired here.
        vacuum_mass = (
            np.asarray(getattr(par, 'cfl_density_floor', 0.0), dtype=float)
            * np.asarray(mesh.vol, dtype=float)
        )
        vacuum = mass <= np.maximum(vacuum_mass, 0.0)
        if np.any(vacuum):
            mass = mass.copy()
            momentum = momentum.copy()
            energy = energy.copy()
            mass[vacuum] = 0.0
            momentum[vacuum] = 0.0
            energy[vacuum] = 0.0
            fluid.Mass[vacuum] = 0.0
            fluid.Mom[vacuum] = 0.0
            fluid.Energy[vacuum] = 0.0
        candidate_mass = mass + dt_value * np.asarray(df_mass, dtype=float)
        candidate_mom = momentum + dt_value * np.asarray(df_mom, dtype=float)
        candidate_energy = energy + dt_value * np.asarray(df_energy, dtype=float)
        first = int(getattr(par, 'noghost', 0))
        last = min(first + int(getattr(par, 'nogrid', len(mass) - first)), len(mass))
        physical = np.zeros(len(mass), dtype=bool)
        physical[first:last] = True
        relative_tolerance = (
            # Permit only roundoff-scale cancellation in E-K for dual-energy
            # states.  A 1e-7 relative tolerance accommodates the observed
            # accumulated synchronization error without accepting a
            # physically meaningful kinetic-energy deficit.
            1.0e-7
            if self._dual_energy_enabled(par) and hasattr(fluid, 'InternalEnergy')
            else 1.0e-12
        )

        def admissible(mass_value, momentum_value, energy_value):
            valid = self._positive_conserved_state(
                mass_value, momentum_value, energy_value,
                mass_floor=mass_floor, energy_floor=energy_floor,
                relative_tolerance=relative_tolerance,
            )
            # Ghost cells are refreshed from the boundary condition before
            # the next hydro step and must not limit a physical update.
            valid[~physical] = True
            return valid

        if np.all(admissible(
            candidate_mass, candidate_mom, candidate_energy,
        )):
            return 1.0
        if not np.all(admissible(mass, momentum, energy)):
            invalid = ~admissible(mass, momentum, energy)
            if np.any(invalid):
                index = int(np.flatnonzero(invalid)[0])
                raise ValueError(
                    'hydro state is outside positivity domain before update at '
                    'cell %d (mass=%s mom=%s energy=%s)' % (
                        index, mass[index], momentum[index], energy[index]
                    )
                )
        low, high = 0.0, 1.0
        for _ in range(48):
            factor = 0.5 * (low + high)
            valid = admissible(
                mass + factor * dt_value * np.asarray(df_mass, dtype=float),
                momentum + factor * dt_value * np.asarray(df_mom, dtype=float),
                energy + factor * dt_value * np.asarray(df_energy, dtype=float),
            )
            if np.all(valid):
                low = factor
            else:
                high = factor
        if getattr(par, 'verbose', 0) >= 1:
            print('[positivity limiter] factor=%s' % low)
        return low

    def _positivity_limited_face_fluxes(
        self, fluid, dt, mesh, par, mass_face, mom_face, energy_face,
        geometric_mom=None,
    ):
        """Apply a local invariant-domain limiter to paired face fluxes.

        A global multiplier is especially damaging for cold, nearly
        pressureless flows: one restrictive cell can suppress continuity in
        every other cell.  Instead, visit each face and scale its conservative
        correction only when the two cells sharing that face would leave the
        positivity domain.  The correction is applied with opposite signs to
        the two cells, so conservation is retained exactly (including at
        nonuniform spherical faces).
        """
        if not getattr(par, 'positivity_preserving', True):
            return 1.0
        dt_value = float(np.asarray(dt, dtype=float))
        mass = np.asarray(fluid.Mass, dtype=float).copy()
        momentum = np.asarray(fluid.Mom, dtype=float).copy()
        energy = np.asarray(fluid.Energy, dtype=float).copy()
        count = len(mass)
        first = int(getattr(par, 'noghost', 0))
        last = min(first + int(getattr(par, 'nogrid', count - first)), count)
        physical = np.zeros(count, dtype=bool)
        physical[first:last] = True
        volume = np.asarray(mesh.vol, dtype=float)
        mass_floor = max(
            0.0, float(np.asarray(getattr(par, 'positivity_density_floor', 0.0)))
        ) * volume
        energy_floor = max(
            0.0, float(np.asarray(getattr(par, 'positivity_energy_floor', 0.0)))
        ) * volume
        relative_tolerance = (
            # Keep the same roundoff allowance used by the global increment
            # limiter.  Dual energy may tolerate tiny E-K cancellation, but
            # it must still reject a substantive K > E state.
            1.0e-7
            if self._dual_energy_enabled(par) and hasattr(fluid, 'InternalEnergy')
            else 1.0e-12
        )

        # Numerical vacuum is not a resolved state and should not contribute
        # a spurious momentum/energy constraint to its neighboring face.
        vacuum_mass = (
            float(np.asarray(getattr(par, 'cfl_density_floor', 0.0))) * volume
        )
        vacuum = mass <= np.maximum(vacuum_mass, 0.0)
        mass[vacuum] = 0.0
        momentum[vacuum] = 0.0
        energy[vacuum] = 0.0
        fluid.Mass[vacuum] = 0.0
        fluid.Mom[vacuum] = 0.0
        fluid.Energy[vacuum] = 0.0

        if geometric_mom is not None:
            # This is a cell source rather than a face flux.  Apply it before
            # limiting the conservative face corrections.
            momentum += dt_value * np.asarray(geometric_mom, dtype=float)

        def valid(mass_value, momentum_value, energy_value):
            # Dual energy protects pressure reconstruction when E-K loses
            # precision, but it cannot make an inadmissible conservative
            # state valid.  Require total energy to contain at least the
            # kinetic energy (up to the small roundoff tolerance above).
            # Otherwise the limiter could pass K > E to the next primitive
            # reconstruction and let the dual field hide the violation.
            result = self._positive_conserved_state(
                mass_value, momentum_value, energy_value,
                mass_floor=mass_floor, energy_floor=energy_floor,
                relative_tolerance=relative_tolerance,
            )
            result[~physical] = True
            return result

        mass_face = np.asarray(mass_face, dtype=float)
        mom_face = np.asarray(mom_face, dtype=float)
        energy_face = np.asarray(energy_face, dtype=float)
        area = np.asarray(mesh.area, dtype=float)
        delta_mass = dt_value * mass_face * area
        delta_mom = dt_value * mom_face * area
        delta_energy = dt_value * energy_face * area
        # Accept the unlimited conservative update immediately when possible.
        # This is the overwhelmingly common path and avoids limiter overhead.
        full_mass = mass + delta_mass - ru.periodic_roll(delta_mass, -1)
        full_mom = momentum + delta_mom - ru.periodic_roll(delta_mom, -1)
        full_energy = energy + delta_energy - ru.periodic_roll(delta_energy, -1)
        if np.all(valid(full_mass, full_mom, full_energy)):
            factors = np.ones(len(mass_face), dtype=float)
            mass, momentum, energy = full_mass, full_mom, full_energy
        else:
            # Construct the limited update from a known admissible state.
            # Increasing one face coefficient changes only its two adjacent
            # cells with equal-and-opposite corrections.  Accept an increase
            # only while both cells remain admissible, so global admissibility
            # is an invariant of the construction rather than something a
            # fixed number of repair passes must recover afterward.
            factors = np.zeros(len(mass_face), dtype=float)
            total_mass = mass.copy()
            total_mom = momentum.copy()
            total_energy = energy.copy()
            if not np.all(valid(total_mass, total_mom, total_energy)):
                invalid = ~valid(total_mass, total_mom, total_energy)
                index = int(np.flatnonzero(invalid)[0])
                raise ValueError(
                    'hydro state is outside positivity domain before paired '
                    'face construction at cell %d (mass=%s mom=%s energy=%s)'
                    % (index, total_mass[index], total_mom[index],
                       total_energy[index])
                )

            def face_candidate(face, factor):
                left = (face - 1) % count
                right = face
                increment = factor - factors[face]
                trial_mass = total_mass.copy()
                trial_mom = total_mom.copy()
                trial_energy = total_energy.copy()
                trial_mass[left] -= increment * delta_mass[face]
                trial_mom[left] -= increment * delta_mom[face]
                trial_energy[left] -= increment * delta_energy[face]
                trial_mass[right] += increment * delta_mass[face]
                trial_mom[right] += increment * delta_mom[face]
                trial_energy[right] += increment * delta_energy[face]
                return trial_mass, trial_mom, trial_energy

            def adjacent_valid(face, state):
                admissible = valid(*state)
                left = (face - 1) % count
                right = face
                return all(
                    admissible[index]
                    for index in (left, right)
                    if physical[index]
                )

            # Alternate traversal direction to reduce ordering bias.  The
            # first sweep already produces a globally admissible update;
            # later sweeps only recover additional face flux monotonically.
            max_recovery_sweeps = 8
            factor_tolerance = 1.0e-13
            for sweep in range(max_recovery_sweeps):
                largest_increase = 0.0
                faces = (
                    range(len(mass_face))
                    if sweep % 2 == 0
                    else range(len(mass_face) - 1, -1, -1)
                )
                for face in faces:
                    current = factors[face]
                    if current >= 1.0 - factor_tolerance:
                        factors[face] = 1.0
                        continue
                    trial = face_candidate(face, 1.0)
                    if adjacent_valid(face, trial):
                        accepted = 1.0
                        accepted_state = trial
                    else:
                        # The current coefficient is known admissible.  Search
                        # only upward, never stepping outside the invariant
                        # domain as the previous reduce-and-repair scheme did.
                        low, high = current, 1.0
                        accepted_state = (
                            total_mass, total_mom, total_energy
                        )
                        for _ in range(48):
                            middle = 0.5 * (low + high)
                            middle_state = face_candidate(face, middle)
                            if adjacent_valid(face, middle_state):
                                low = middle
                                accepted_state = middle_state
                            else:
                                high = middle
                        accepted = low
                    largest_increase = max(
                        largest_increase, accepted - current
                    )
                    factors[face] = accepted
                    total_mass, total_mom, total_energy = accepted_state
                if np.all(factors >= 1.0 - factor_tolerance):
                    factors[...] = 1.0
                    break
                if largest_increase <= factor_tolerance:
                    break
            mass, momentum, energy = total_mass, total_mom, total_energy

        if not np.all(valid(mass, momentum, energy)):
            invalid = ~valid(mass, momentum, energy)
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                'hydro state is outside positivity domain after face update '
                'at cell %d (mass=%s mom=%s energy=%s)' %
                (index, mass[index], momentum[index], energy[index])
            )

        fluid.Mass[...] = mass
        fluid.Mom[...] = momentum
        fluid.Energy[...] = energy
        self._last_face_limiter_factors = factors
        return float(np.min(factors)) if factors.size else 1.0
        
    def SetInterFaceFlux(self,mesh,fluid,boundcond, method='Rusanov',verbose=None, order=0):
        """Set interface fluxes using GLF, Rusanov, or HLLC fluxes."""
        if verbose is None:
            verbose = 0
        if method in ('GLF', 'Rusanov', 'HLLC'):
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
            else:  # HLLC uses Rusanov speeds for CFL and vacuum fallback.
                fluid.cmax = np.maximum(fluid.vsignal, ru.periodic_roll(fluid.vsignal, 1))
            
            self.SetFaceLR(mesh,fluid, boundcond, order=order)
            self.SetFluxOnFace(
                fluid, boundcond, order=order, par=getattr(mesh, '_par', None), method=method
            )
            self._apply_low_density_flux_mask(
                fluid, getattr(mesh, '_par', None)
            )
            self._apply_hydrostatic_core_flux(fluid, getattr(mesh, '_par', None))
            self._zero_spherical_origin_flux(mesh, fluid)
        else:
            raise ValueError("Interface flux method unknown: %s"%method) 
        if (verbose>=2):
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

        dual_energy = (
            self._dual_energy_enabled(getattr(mesh, '_par', None))
            and hasattr(fluid, 'InternalEnergy')
            and getattr(fluid.eos, 'is_polytropic', False)
        )
        df_InternalEnergy = None
        if dual_energy:
            gamma_minus_one = fluid.eos.gamma - 1.0
            velocity_left = np.asarray(fluid.vel.L, dtype=float)
            velocity_right = np.asarray(fluid.vel.R, dtype=float)
            pressure_left = np.asarray(fluid.pre.L, dtype=float)
            pressure_right = np.asarray(fluid.pre.R, dtype=float)
            internal_left = pressure_left / gamma_minus_one
            internal_right = pressure_right / gamma_minus_one
            face_velocity = np.where(
                0.5 * (velocity_left + velocity_right) >= 0.0,
                velocity_left,
                velocity_right,
            )
            face_internal = np.where(
                face_velocity >= 0.0,
                internal_left,
                internal_right,
            )
            internal_flux = face_velocity * face_internal
            origin_face = self._spherical_origin_face_index(mesh)
            if origin_face is not None:
                internal_flux[origin_face] = 0.0
            df_InternalEnergy = (
                internal_flux * area
                - ru.periodic_roll(internal_flux * area, -1)
            )
            if getattr(mesh, 'coordsys', None) == 'spherical':
                df_InternalEnergy -= fluid.pre * (
                    ru.periodic_roll(face_velocity * area, -1)
                    - face_velocity * area
                )

        par = getattr(mesh, '_par', None)
        if par is None:
            # Standalone/unit-test meshes have no physical-cell metadata.  In
            # particular, retain the exact spherical pressure cancellation in
            # this legacy path; configured simulations use the paired-face
            # limiter below.
            positivity_factor = self._positivity_limited_increment(
                fluid, dt, mesh, par, df_Mass, df_Mom, df_Energy
            )
            fluid.Mass += positivity_factor * df_Mass * dt
            fluid.Mom += positivity_factor * df_Mom * dt
            fluid.Energy += positivity_factor * df_Energy * dt
            fluid.time += dt
            return
        geometric_mom = None
        if getattr(mesh, 'coordsys', None) == 'spherical':
            area_right = ru.periodic_roll(area, -1)
            geometric_mom = fluid.pre * (area_right - area)
        positivity_factor = self._positivity_limited_face_fluxes(
            fluid, dt, mesh, par,
            fluid.Mass.flux, fluid.Mom.flux, fluid.Energy.flux,
            geometric_mom=geometric_mom,
        )
        if df_InternalEnergy is not None:
            # Couple the dual-energy advection to the same face coefficients
            # used by the conservative update.  Applying the minimum face
            # coefficient globally defeats the purpose of the local limiter.
            factors = np.asarray(
                getattr(self, '_last_face_limiter_factors',
                        np.ones(len(fluid.Mass.flux))),
                dtype=float,
            )
            limited_internal_flux = internal_flux * factors
            limited_df_internal = (
                limited_internal_flux * area
                - ru.periodic_roll(limited_internal_flux * area, -1)
            )
            if getattr(mesh, 'coordsys', None) == 'spherical':
                limited_df_internal -= fluid.pre * (
                    ru.periodic_roll(
                        factors * face_velocity * area, -1
                    ) - factors * face_velocity * area
                )
            fluid.InternalEnergy += limited_df_internal * dt
            fluid.InternalEnergy = np.maximum(fluid.InternalEnergy, 0.0)
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
            or getattr(par, "cosmological_gravity", False)
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
            cosmological=getattr(par, "cosmological_gravity", False),
            cosmology=getattr(par, "cosmology", None),
        )

    def ApplyGravity(self, dt, mesh, fluid, par):
        """Apply the combined external and gas self-gravity source update."""
        gravity = self._gravity_model(par)
        if gravity is None:
            return 0
        if getattr(gravity, "cosmological", False):
            par.fluid_time = fluid.time
        crossing_safety_factor = getattr(par, "dark_matter_crossing_safety_factor", 0.1)
        if getattr(gravity, "dark_matter", None) is not None:
            gravity.advance_dark_matter(
                dt,
                mesh,
                fluid.rho,
                par,
                crossing_safety_factor=crossing_safety_factor,
            )
        # ApplyGravity follows the conservative hydro flux update and precedes
        # the primitive-state refresh.  Therefore fluid.rho and fluid.vel can
        # still describe the pre-hydro state, while Mass and Mom already
        # describe the post-hydro state.  Derive both quantities from the
        # current conserved fields so the gravity momentum and work updates
        # use the same state.
        volume = np.asarray(mesh.vol, dtype=float)
        mass = np.asarray(fluid.Mass, dtype=float)
        momentum = np.asarray(fluid.Mom, dtype=float)
        current_rho = np.zeros_like(mass)
        np.divide(mass, volume, out=current_rho, where=volume > 0.0)
        current_vel = np.zeros_like(momentum)
        np.divide(momentum, mass, out=current_vel, where=mass > 0.0)
        acceleration = gravity.acceleration_on_mesh(
            mesh, rho=current_rho, par=par
        )
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
        # Advance momentum with constant acceleration over this source step.
        # The corresponding kinetic-energy change is
        #   rho * [v a dt + 1/2 (a dt)^2].
        # Using only rho*v*a*dt can leave E < K in strongly collapsing cells,
        # which is especially visible with HLLC and causes the positivity
        # check to reject the next hydro update.
        acceleration_dt = acceleration * float(np.asarray(dt, dtype=float))
        gravity_work = np.asarray(
            current_rho
            * (current_vel * acceleration_dt + 0.5 * acceleration_dt**2)
            * volume,
            dtype=float,
        )
        fluid.Mom += current_rho * acceleration * volume * dt
        fluid.Energy += gravity_work
        self.last_gravity_work = float(
            np.sum(gravity_work[self._interior_slice(par)])
        )
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

    def ApplyThermochemistryFast(self, dt, mesh, fluid, par, transport_result=None):
        """Fast source update for RT-coupled thermo-chemistry tests."""
        return rtc.apply_thermochemistry_fast(
            dt,
            mesh,
            fluid,
            par,
            transport_result=transport_result,
        )

    def ApplyRadiationPressure(self, dt, mesh, fluid, par, source_result):
        """Apply momentum from photons consumed by the thermo-chemistry step."""
        if not getattr(par, "radiation_pressure", False):
            return 0
        if not source_result or source_result.get("absorbed_photon_rate") is None:
            return 0

        code_units = _code_units(par)
        scales = code_unit_scales(code_units)
        interior = self._interior_slice(par)
        absorbed = np.asarray(source_result["absorbed_photon_rate"], dtype=float)
        energies = np.asarray(source_result["photon_energy_erg"], dtype=float)
        if absorbed.ndim == 1:
            absorbed = absorbed[None, :]
        if energies.ndim == 0:
            energies = energies[None]
        if absorbed.shape[0] != energies.size:
            raise ValueError("absorbed photon groups and photon energies disagree")
        if absorbed.shape[1] != par.nogrid:
            raise ValueError("absorbed photon rate must contain physical cells only")

        rho_cgs = np.asarray(fluid.rho[interior], dtype=float) * scales["density_g_cm3"]
        momentum_rate_density = (
            float(source_result.get("direction", 1))
            * np.sum(absorbed * energies[:, None], axis=0)
            / SPEED_OF_LIGHT_CGS
        )
        efficiency = float(getattr(par, "radiation_pressure_efficiency", 1.0))
        valid = rho_cgs > 0.0
        if not np.any(valid):
            return 0
        acceleration_cgs = np.zeros_like(momentum_rate_density)
        acceleration_cgs[valid] = (
            efficiency * momentum_rate_density[valid] / rho_cgs[valid]
        )
        acceleration = acceleration_cgs / scales["acceleration_cm_s2"]
        volume = np.asarray(mesh.vol[interior], dtype=float)
        momentum = fluid.Mom[interior]
        energy = fluid.Energy[interior]
        rho = fluid.rho[interior]
        velocity = fluid.vel[interior]
        momentum[valid] += rho[valid] * acceleration[valid] * volume[valid] * dt
        energy[valid] += (
            rho[valid]
            * velocity[valid]
            * acceleration[valid]
            * volume[valid]
            * dt
        )
        return 1


    def SetBoundary(self, mesh, fluid, par):
        """Fill ghost cells according to the selected boundary condition."""
        self.ApplyHydrostaticCore(mesh, fluid, par)
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
        density = np.asarray(fluid.rho, dtype=float)
        if xdelta.shape != vsignal.shape:
            interior = self._interior_slice(par)
            if xdelta[interior].shape == vsignal.shape:
                xdelta = xdelta[interior]
                density = density[interior]
            elif vsignal[interior].shape == xdelta.shape:
                vsignal = vsignal[interior]
                density = density[interior]

        # Ghost zones are needed by the Riemann solve but must not determine
        # the CFL step.  In particular, reflecting/outflow boundary updates
        # can leave a ghost velocity temporarily very large while the active
        # solution remains valid.  Keep the full signal-speed array for later
        # flux work, and reduce only over the active cells here.
        active_xdelta = xdelta
        active_density = density
        active_vsignal = vsignal
        first = int(getattr(par, 'noghost', 0))
        active_count = int(getattr(par, 'nogrid', len(vsignal)))
        active_slice = slice(first, first + active_count)
        if (
            xdelta.ndim == 1
            and vsignal.ndim == 1
            and len(vsignal) >= first + active_count + first
        ):
            active_xdelta = xdelta[active_slice]
            active_density = density[active_slice]
            active_vsignal = vsignal[active_slice]

        core_mask = getattr(par, '_hydrostatic_core_mask', None)
        if core_mask is not None:
            core_active = np.asarray(core_mask[active_slice], dtype=bool)
            active_vsignal = np.asarray(active_vsignal, dtype=float).copy()
            active_vsignal[core_active] = 0.0

        # A vacuum cell has no characteristic speed for the CFL constraint.
        # EOS sound-speed evaluation can produce ``inf`` for rho == 0 because
        # pressure/rho is undefined; exclude such cells from the minimum and
        # keep their interface signal speed neutral for the next flux update.
        density_floor = max(
            0.0, float(np.asarray(getattr(par, 'cfl_density_floor', 0.0)))
        )
        zero_density = active_density <= density_floor
        if np.any(zero_density):
            active_vsignal = np.asarray(active_vsignal, dtype=float).copy()
            active_vsignal[zero_density] = 0.0
        dt_array = self._safe_divide(CFL * active_xdelta, active_vsignal)
        dtmax = float(np.asarray(par.dtmax, dtype=float))
        dt_array = np.where(active_vsignal != 0.0, dt_array, dtmax)
        dt = np.amin(dt_array)
        fluid.vsignal = np.asarray(vsignal, dtype=float)
        if len(fluid.vsignal) == len(active_vsignal):
            fluid.vsignal[zero_density] = 0.0
        else:
            fluid.vsignal[active_slice] = active_vsignal
        self.dt = dt
        if np.isnan(np.asarray(dt)):
            print('vsignal', vsignal)
            print('fluid.vel', fluid.vel)
            print('fluid.cs', fluid.cs)
            raise Exception(" time step is nan")
        if dt < float(np.asarray(par.dtmin, dtype=float)):
            active_index = int(np.argmin(dt_array))
            min_index = active_index + first
            if len(np.asarray(fluid.vel)) == len(active_vsignal):
                diagnostic_index = active_index
            else:
                diagnostic_index = min_index
            raise ValueError(
                " time step %.2e smaller than the minimum time step %.2e "
                "at cell %d (rho=%.2e, vel=%.2e, cs=%.2e, dx=%.2e)"
                % (
                    dt,
                    par.dtmin,
                    min_index,
                    active_density[active_index],
                    fluid.vel[diagnostic_index],
                    fluid.cs[diagnostic_index],
                    active_xdelta[active_index],
                )
            )
        if dt > dtmax:
            dt = dtmax
        if (
            getattr(par, 'verbose', 0) >= 1
            # Keep routine CFL reductions quiet; report only a timestep that
            # has fallen at least four decades below the configured maximum.
            and dt <= 1.0e-4 * float(np.asarray(par.dtmax, dtype=float))
        ):
            min_index = int(np.argmin(dt_array))
            if len(np.asarray(fluid.vel)) == len(active_vsignal):
                diagnostic_index = min_index
            else:
                diagnostic_index = min_index + first
            print(
                '[hydro dt] t=%s dt=%s idx=%d radius=%s rho=%s vel=%s '
                'cs=%s vsignal=%s dx=%s pre=%s dtmin=%s dtmax=%s'
                % (
                    fluid.time,
                    dt,
                    diagnostic_index,
                    np.asarray(mesh.coordinate)[diagnostic_index],
                    np.asarray(fluid.rho)[diagnostic_index],
                    np.asarray(fluid.vel)[diagnostic_index],
                    np.asarray(fluid.cs)[diagnostic_index],
                    np.asarray(vsignal)[diagnostic_index],
                    np.asarray(mesh.xdelta)[diagnostic_index],
                    np.asarray(fluid.pre)[diagnostic_index],
                    par.dtmin,
                    par.dtmax,
                )
            )
            cell_volume = np.asarray(mesh.vol)[diagnostic_index]
            cell_rho = np.asarray(fluid.rho)[diagnostic_index]
            cell_vel = np.asarray(fluid.vel)[diagnostic_index]
            cell_energy_density = (
                np.asarray(fluid.Energy)[diagnostic_index] / cell_volume
            )
            cell_kinetic_density = 0.5 * cell_rho * cell_vel**2
            cell_thermal_density = cell_energy_density - cell_kinetic_density
            cell_specific_thermal = (
                cell_thermal_density / cell_rho
                if cell_rho > 0.0 else 0.0
            )
            print(
                '[hydro dt energy] idx=%d energy_density=%s '
                'kinetic_density=%s thermal_density=%s '
                'specific_thermal=%s'
                % (
                    diagnostic_index,
                    cell_energy_density,
                    cell_kinetic_density,
                    cell_thermal_density,
                    cell_specific_thermal,
                )
            )
            print(
                '[hydro dt mask] cfl_density_floor=%s masked=%s' % (
                    density_floor,
                    int(np.count_nonzero(zero_density)),
                )
            )
            neighbor_start = max(first, diagnostic_index - 2)
            neighbor_stop = min(
                first + int(getattr(par, 'nogrid', len(np.asarray(fluid.rho)))),
                diagnostic_index + 3,
            )
            print('[hydro dt neighbors] idx radius rho vel cs pre')
            for neighbor in range(neighbor_start, neighbor_stop):
                print(
                    '[hydro dt neighbors] %d %s %s %s %s %s'
                    % (
                        neighbor,
                        np.asarray(mesh.coordinate)[neighbor],
                        np.asarray(fluid.rho)[neighbor],
                        np.asarray(fluid.vel)[neighbor],
                        np.asarray(fluid.cs)[neighbor],
                        np.asarray(fluid.pre)[neighbor],
                    )
                )
        return dt
