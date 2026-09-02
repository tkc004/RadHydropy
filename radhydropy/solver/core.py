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
        self.dual_energy_pressure_fallback_count = 0
        self.dual_energy_synchronization_count = 0
        self.dual_energy_floor_count = 0
        self.dual_energy_floor_injected_energy = 0.0
        self.dual_energy_entropy_limiter_count = 0
        # Last pressure-reconstruction diagnostics.  These are deliberately
        # arrays rather than counters: resolution comparisons need to locate
        # the cells where E-K and the independently evolved thermal state
        # disagree.
        self.dual_energy_total_thermal = None
        self.dual_energy_internal_density = None
        self.dual_energy_total_pressure = None
        self.dual_energy_dual_pressure = None
        self.dual_energy_total_valid = None
        self.dual_energy_dual_valid = None
        self.dual_energy_pressure_selection_code = None
        self.last_centrifugal_work = 0.0
        self.last_centrifugal_work_by_cell = None
        self.last_centrifugal_source_factors = None
        self.centrifugal_source_limited_count = 0

    def _safe_divide(self, numerator, denominator):
        return ru.SafeDivide(numerator, denominator)

    def _interior_slice(self, par):
        first = int(par.mesh.ghost_cells)
        return slice(
            first,
            first + int(par.mesh.grid_cells),
        )

    def _thermochemistry_enabled(self, fluid, par):
        return rtc.thermochemistry_enabled(fluid, par)

    def _thermochemistry_radiation_enabled(self, fluid, par):
        return (
            self._thermochemistry_enabled(fluid, par)
            and (
                getattr(par, 'hydrogen_radiation_field', False)
                or getattr(par, 'radiative_transfer', False)
            )
            and hasattr(fluid, 'ngamma_code')
        )

    def ApplyRadiativeTransfer(self, mesh, fluid, par):
        """Refresh photon density from the shared radiative-transfer solver."""
        if not getattr(par, 'radiative_transfer', False):
            return None
        code_units = _code_units(par)
        scales = code_unit_scales(code_units)
        if not hasattr(fluid, 'ngamma_code'):
            fluid.ngamma_code = np.zeros(np.shape(fluid.rho_code), dtype=float)
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
                np.asarray(fluid.rho_code[interior], dtype=float) * scales['density_g_cm3'],
                np.asarray(fluid.xHI[interior], dtype=float),
                hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
                sigma_gamma=sigma_groups,
                boundary_flux=boundary_groups,
                source_photon_rate=source_groups,
                direction=rrt._parameter_value(par, 'radiative_transfer_direction', 1),
                coordsys=getattr(mesh, 'coordsys', 'cartesian'),
                group_edges_eV=group_edges_eV,
            )
            fluid.ngamma_code[:, interior] = (
                np.asarray(result.cell_photon_density, dtype=float)
                / scales['number_density_cm3']
            )
            return result
        sigma_value = getattr(par, 'hydrogen_sigma_gamma', DEFAULT_SIGMA_GAMMA)
        boundary_value = rrt._parameter_value(par, 'radiative_transfer_boundary_flux', 0.0)
        source_value = rrt._parameter_value(par, 'radiative_transfer_source_photon_rate', 0.0)
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
            np.asarray(fluid.rho_code[interior], dtype=float) * scales['density_g_cm3'],
            np.asarray(fluid.xHI[interior], dtype=float),
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
            sigma_gamma=sigma_gamma_cm2,
            boundary_flux=boundary_flux,
            source_photon_rate=source_photon_rate,
            direction=rrt._parameter_value(par, 'radiative_transfer_direction', 1),
            coordsys=getattr(mesh, 'coordsys', 'cartesian'),
        )
        fluid.ngamma_code[interior] = (
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
        fluid.Mass_code.flux[origin_face] = 0.0
        fluid.Mom_code.flux[origin_face] = 0.0
        fluid.Energy_code.flux[origin_face] = 0.0
        if hasattr(fluid, 'AngularMomentum_code') and hasattr(fluid.AngularMomentum_code, 'flux'):
            fluid.AngularMomentum_code.flux[origin_face] = 0.0
        if hasattr(fluid, 'rotational_energy_flux'):
            fluid.rotational_energy_flux[origin_face] = 0.0

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
        first = int(par.mesh.ghost_cells)
        last = first + int(par.mesh.grid_cells)
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
        for name in ('rho_code', 'vel_code', 'temp_code', 'mu', 'pre_code', 'xHI',
                     'xHeI', 'xHeII', 'xHeIII',
                     'specific_angular_momentum_code'):
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
        for name in ('rho_code', 'vel_code', 'temp_code', 'mu', 'pre_code', 'xHI',
                     'xHeI', 'xHeII', 'xHeIII',
                     'specific_angular_momentum_code'):
            if name in state and hasattr(fluid, name):
                values = np.asarray(getattr(fluid, name), dtype=float).copy()
                values[core] = state[name]
                setattr(fluid, name, as_named_array(values))
        # A fixed core is hydrostatic and has no resolved radial motion.
        fluid.vel_code[core] = 0.0

    def _apply_hydrostatic_core_flux(self, fluid, par):
        """Close the resolved halo with a pressure-bearing, no-mass-flux core."""
        face = getattr(par, '_hydrostatic_core_face', None)
        state = getattr(fluid, '_hydrostatic_core', None)
        if face is None or state is None:
            return
        core_last = state['core_last']
        # The core is fixed-mass: pressure acts on the halo, but gas, energy,
        # and radial momentum do not cross the core/halo interface.
        fluid.Mass_code.flux[face] = 0.0
        fluid.Energy_code.flux[face] = 0.0
        fluid.Mom_code.flux[face] = fluid.pre_code[core_last]

    def _boundary_field_names(self, *args, **kwargs):
        from .boundary_conditions import _boundary_field_names

        return _boundary_field_names(self, *args, **kwargs)

    def _copy_boundary_state(self, *args, **kwargs):
        from .boundary_conditions import _copy_boundary_state

        return _copy_boundary_state(self, *args, **kwargs)

    def _boundary_state(self, *args, **kwargs):
        from .boundary_conditions import _boundary_state

        return _boundary_state(self, *args, **kwargs)

    def _to_code_number_density(self, *args, **kwargs):
        from .boundary_conditions import _to_code_number_density

        return _to_code_number_density(self, *args, **kwargs)

    def _apply_periodic_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_periodic_boundary

        return _apply_periodic_boundary(self, *args, **kwargs)

    def _apply_open_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_open_boundary

        return _apply_open_boundary(self, *args, **kwargs)

    def _apply_reflecting_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_reflecting_boundary

        return _apply_reflecting_boundary(self, *args, **kwargs)

    def _apply_spherical_inner_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_spherical_inner_boundary

        return _apply_spherical_inner_boundary(self, *args, **kwargs)

    def _apply_open_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_open_spherical_boundary

        return _apply_open_spherical_boundary(self, *args, **kwargs)

    def _apply_inflow_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_inflow_spherical_boundary

        return _apply_inflow_spherical_boundary(self, *args, **kwargs)

    def _apply_outflow_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_outflow_spherical_boundary

        return _apply_outflow_spherical_boundary(self, *args, **kwargs)

    def _apply_wind_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_wind_spherical_boundary

        return _apply_wind_spherical_boundary(self, *args, **kwargs)

    def SetPrimitive(self, mesh, fluid, par=None, verbose=None):
        """Update primitive variables from conserved quantities."""
        if verbose is None:
            verbose = 0
        vol = mesh.vol
        rho = np.asarray(self._safe_divide(fluid.Mass_code, vol), dtype=float)
        active = np.isfinite(rho) & (rho > 0.0)
        fluid.active = active
        rho = np.where(active, rho, 0.0)

        mass = np.asarray(fluid.Mass_code, dtype=float)
        momentum = np.asarray(fluid.Mom_code, dtype=float)
        vel = np.zeros_like(rho)
        valid_mass = active & np.isfinite(mass) & (mass > 0.0)
        vel[valid_mass] = momentum[valid_mass] / mass[valid_mass]

        energy_density = np.zeros_like(rho)
        valid_volume = active & np.isfinite(vol) & (vol > 0.0)
        energy_density[valid_volume] = (
            np.asarray(fluid.Energy_code, dtype=float)[valid_volume]
            / np.asarray(vol, dtype=float)[valid_volume]
        )
        fluid.rho_code = as_named_array(rho)
        fluid.vel_code = as_named_array(vel)
        if hasattr(fluid, 'AngularMomentum_code'):
            specific_angular_momentum = np.zeros_like(rho)
            np.divide(
                np.asarray(fluid.AngularMomentum_code, dtype=float),
                mass,
                out=specific_angular_momentum,
                where=valid_mass,
            )
            fluid.specific_angular_momentum_code = as_named_array(
                specific_angular_momentum
            )
        rotational_energy_density = self._rotational_energy_density(
            mesh, fluid, par
        )
        density_floor = self._cfl_density_floor(par)
        numerical_vacuum = active & (rho <= density_floor)
        fluid.vel_code[numerical_vacuum] = 0.0
        # Conserved Energy contains rotational kinetic energy when the opt-in
        # model is enabled; pressure sees only thermal plus radial kinetic
        # energy at this stage.
        energy_density = as_named_array(
            energy_density - rotational_energy_density
        )
        pressure_args = (fluid.rho_code, fluid.vel_code, energy_density)
        if getattr(fluid.eos, 'is_isothermal', False):
            total_pressure = fluid.eos.pressure_from_conserved(
                *pressure_args,
                temp=getattr(fluid, 'temp_code', None),
                mu=getattr(fluid, 'mu', None),
            )
        else:
            total_pressure = fluid.eos.pressure_from_conserved(*pressure_args)
        fluid.pre_code = total_pressure
        if self._dual_energy_enabled(par) and hasattr(fluid, 'InternalEnergy_code'):
            internal_density = np.zeros_like(rho)
            internal_density[valid_volume] = (
                np.asarray(fluid.InternalEnergy_code, dtype=float)[valid_volume]
                / np.asarray(vol, dtype=float)[valid_volume]
            )
            dual_pressure = (fluid.eos.gamma - 1.0) * internal_density
            total_thermal = energy_density - 0.5 * fluid.rho_code * fluid.vel_code**2
            eta1 = self._dual_energy_eta(par, 'dual_energy_eta1', 1.0e-3)
            total_valid = (
                active & ~numerical_vacuum
                & np.isfinite(total_thermal) & (total_thermal > 0.0)
                & np.isfinite(total_pressure) & (total_pressure > 0.0)
            )
            dual_valid = (
                active & ~numerical_vacuum
                & np.isfinite(internal_density) & (internal_density > 0.0)
                & np.isfinite(dual_pressure) & (dual_pressure > 0.0)
            )
            thermal_fraction = np.divide(
                total_thermal,
                np.maximum(np.abs(energy_density), 1.0e-300),
                out=np.zeros_like(total_thermal),
                where=np.isfinite(total_thermal),
            )
            consistency_factor = max(0.0, float(np.asarray(getattr(
                par, 'dual_energy_consistency_factor', 1.0e-1), dtype=float)))
            dual_to_total = np.divide(
                internal_density,
                np.maximum(total_thermal, 1.0e-300),
                out=np.full_like(internal_density, np.inf),
                where=total_valid,
            )
            # If dual thermal energy is much larger than conservative E-K,
            # cancellation has made E-K unusably small even though it remains
            # formally positive.  Prefer the independently evolved dual
            # estimate in that case.  Conversely, if dual energy is much
            # smaller than E-K, retain the conservative estimate.  Rejecting
            # the dual estimate in both directions was the source of the
            # artificial near-zero temperatures outside strong shocks.
            upper_consistency_factor = (
                1.0 / consistency_factor if consistency_factor > 0.0 else np.inf
            )
            dual_preferred = dual_valid & (
                ~total_valid | (dual_to_total > upper_consistency_factor)
            )
            use_total = total_valid & ~dual_preferred & (
                (thermal_fraction > eta1)
                | ~dual_valid
                | (dual_to_total < consistency_factor)
            )
            use_dual = dual_valid & ~use_total
            pressure_selection = str(getattr(
                par, 'dual_energy_pressure_selection', 'switch'
            )).lower()
            if pressure_selection in ('conservative', 'e-k', 'ek'):
                use_total = total_valid
                use_dual = np.zeros_like(use_total, dtype=bool)
            elif pressure_selection in (
                'internal', 'internal-energy', 'dual', 'dual-energy'
            ):
                # Use the independently evolved internal-energy equation for
                # primitive pressure reconstruction.  Total Energy remains
                # the conservative flux variable and is retained for audits.
                use_total = np.zeros_like(use_total, dtype=bool)
                use_dual = dual_valid

            # Preserve the exact same-state quantities used below for
            # pressure selection.  Codes: -1 inactive, 0 conservative E-K,
            # 1 dual-energy internal state, 2 pressure-floor reconstruction.
            selection_code = np.full(rho.shape, -1, dtype=np.int8)
            selection_code[use_total] = 0
            selection_code[use_dual] = 1
            self.dual_energy_total_thermal = total_thermal.copy()
            self.dual_energy_internal_density = internal_density.copy()
            self.dual_energy_total_pressure = np.asarray(total_pressure, dtype=float).copy()
            self.dual_energy_dual_pressure = np.asarray(dual_pressure, dtype=float).copy()
            self.dual_energy_total_valid = total_valid.copy()
            self.dual_energy_dual_valid = dual_valid.copy()
            self.dual_energy_pressure_selection_code = selection_code
            fluid.pre_code[use_dual] = dual_pressure[use_dual]
            fluid.pre_code[use_total] = total_pressure[use_total]

            # If the separately advected field has failed but E-K is still a
            # valid conservative estimate, use E-K and count the fallback.
            fallback = (
                active & ~numerical_vacuum & ~dual_valid & total_valid
            )
            self.dual_energy_pressure_fallback_count += int(
                np.count_nonzero(fallback)
            )

            # Neither estimate is usable.  Add only the configured small
            # positive thermal energy to the conservative state, retain
            # total-energy accounting separately, and use its pressure.  In
            # particular, do not use a large dual estimate when E-K is
            # inadmissible.
            # The pressure floor is a last resort only when both the
            # conservative E-K estimate and the independently evolved
            # InternalEnergy are invalid.  In particular, an inadmissible
            # E-K residual is the normal reason for selecting dual energy in
            # a cold converging flow; it must not overwrite a valid dual
            # state with the configured floor.
            both_invalid = (
                active & ~numerical_vacuum & ~total_valid & ~dual_valid
            )
            if np.any(both_invalid):
                floor_pressure_value = max(
                    0.0, float(np.asarray(
                        getattr(par, 'dual_energy_pressure_floor', 1.0e-20),
                        dtype=float,
                    ))
                )
                if floor_pressure_value <= 0.0:
                    floor_pressure_value = 1.0e-20
                floor_pressure = np.full_like(rho, floor_pressure_value)
                floor_internal_density = floor_pressure / (fluid.eos.gamma - 1.0)
                current_internal_density = np.maximum(total_thermal, 0.0)
                injected_density = np.maximum(
                    floor_internal_density - current_internal_density, 0.0
                )
                injected_energy = injected_density * np.asarray(vol, dtype=float)
                fluid.Energy_code[both_invalid] += injected_energy[both_invalid]
                fluid.pre_code[both_invalid] = floor_pressure[both_invalid]
                internal_density[both_invalid] = floor_internal_density[both_invalid]
                fluid.InternalEnergy_code[both_invalid] = injected_energy[both_invalid] + (
                    current_internal_density[both_invalid]
                    * np.asarray(vol, dtype=float)[both_invalid]
                )
                self.dual_energy_floor_count += int(
                    np.count_nonzero(both_invalid)
                )
                self.dual_energy_floor_injected_energy += float(
                    np.sum(injected_energy[both_invalid])
                )
                selection_code[both_invalid] = 2
            # Keep the conservative fallback pressure for cells where the
            # dual field is invalid but E-K is admissible.
            fluid.pre_code[fallback] = total_pressure[fallback]
        fluid.rho_code[~active] = 0.0
        fluid.vel_code[~active] = 0.0
        invalid_pressure = np.logical_or(fluid.pre_code <= 0.0, np.isnan(fluid.pre_code))
        temperature_floor = getattr(par, 'hydro_temperature_floor', None)
        if temperature_floor is not None and float(temperature_floor) > 0.0:
            floor_pressure = fluid.eos.pressure(
                fluid.rho_code,
                float(temperature_floor),
                fluid.mu,
            )
            # Enforce the configured floor for both invalid reconstructions
            # and valid states that have cooled below the physical minimum.
            below_floor = (
                ~numerical_vacuum
                & np.logical_or(invalid_pressure, fluid.pre_code < floor_pressure)
            )
            fluid.pre_code[below_floor] = floor_pressure[below_floor]
        else:
            fluid.pre_code[invalid_pressure & ~numerical_vacuum] = 0.0
        fluid.pre_code[numerical_vacuum] = 0.0
        if verbose >= 2:
            print('fluid.rho_code',fluid.rho_code)
            print('fluid.vel_code',fluid.vel_code)
            print('fluid.pre_code', fluid.pre_code)
    
    def SetConserved(self, mesh, fluid, verbose=None):
        """Update conserved mass, momentum, and energy from primitive variables."""
        if verbose is None:
            verbose = 0
        par = getattr(mesh, '_par', None)
        density_floor = self._cfl_density_floor(par)
        dual_energy = self._dual_energy_enabled(par)
        old_internal = (
            np.asarray(fluid.InternalEnergy_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, 'InternalEnergy_code')
            else None
        )
        old_total_energy = (
            np.asarray(fluid.Energy_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, 'Energy_code')
            else None
        )
        old_total_mass = (
            np.asarray(fluid.Mass_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, 'Mass_code')
            else None
        )
        old_total_momentum = (
            np.asarray(fluid.Mom_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, 'Mom_code')
            else None
        )
        old_angular_momentum = (
            np.asarray(fluid.AngularMomentum_code, dtype=float).copy()
            if hasattr(fluid, 'AngularMomentum_code') else None
        )
        old_potential_energy = (
            np.asarray(fluid.GravitationalPotentialEnergy_code, dtype=float).copy()
            if (
                hasattr(fluid, 'GravitationalPotentialEnergy_code')
                and (
                    getattr(fluid, '_gravity_potential_energy_initialized', False)
                    or np.any(np.asarray(fluid.GravitationalPotentialEnergy_code, dtype=float) != 0.0)
                )
            ) else None
        )
        old_conserved = None
        if density_floor > 0.0 and all(
            hasattr(fluid, name) for name in ('Mass_code', 'Mom_code', 'Energy_code')
        ):
            density = np.asarray(fluid.rho_code, dtype=float)
            inactive = np.isfinite(density) & (density <= density_floor)
            old_conserved = (
                inactive,
                np.asarray(fluid.Mass_code, dtype=float).copy(),
                np.asarray(fluid.Mom_code, dtype=float).copy(),
                np.asarray(fluid.Energy_code, dtype=float).copy(),
            )
        vol = mesh.vol
        fluid.Mass_code = as_named_array(fluid.rho_code * vol)
        fluid.Mom_code = as_named_array(fluid.rho_code * fluid.vel_code * vol)
        if hasattr(fluid, 'specific_angular_momentum_code') or old_angular_momentum is not None:
            specific_angular_momentum = np.asarray(
                getattr(fluid, 'specific_angular_momentum_code',
                        np.zeros_like(fluid.rho_code)),
                dtype=float,
            )
            fluid.AngularMomentum_code = as_named_array(
                fluid.rho_code * specific_angular_momentum * vol
            )
        rotational_energy_density = self._rotational_energy_density(
            mesh, fluid, par
        )
        fluid.Energy_code = as_named_array(
            (
                fluid.eos.total_energy_density(fluid.rho_code, fluid.vel_code, fluid.pre_code)
                + rotational_energy_density
            ) * vol
        )
        potential = self._gravity_potential(mesh, par)
        if potential is not None:
            fluid.GravitationalPotentialEnergy_code = as_named_array(
                fluid.Mass_code * potential
                if old_potential_energy is None else old_potential_energy
            )
            fluid._gravity_potential_energy_initialized = True
        fluid.Mass_code[np.logical_or(fluid.Mass_code<0.0, np.isnan(fluid.Mass_code))] = 0.0
        fluid.Energy_code[np.logical_or(fluid.Energy_code<0.0, np.isnan(fluid.Energy_code))] = 0.0
        if old_total_energy is not None:
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            fluid.Energy_code[first:first + count] = old_total_energy[first:first + count]
        if old_total_mass is not None and old_total_momentum is not None:
            # In dual-energy mode Mass/Mom are the authoritative conservative
            # state.  Rebuilding them as rho*vol and rho*vel*vol after
            # SetPrimitive introduces a division/multiplication round trip;
            # in a kinetic-dominated cell that roundoff can make K exceed the
            # preserved total Energy.  Keep the conserved hydro quantities
            # exact and synchronize only the primitive/thermal quantities.
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            fluid.Mass_code[first:first + count] = old_total_mass[first:first + count]
            fluid.Mom_code[first:first + count] = old_total_momentum[first:first + count]
        if old_angular_momentum is not None:
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            fluid.AngularMomentum_code[first:first + count] = (
                old_angular_momentum[first:first + count]
            )
        if dual_energy and getattr(fluid.eos, 'is_polytropic', False):
            internal = np.asarray(
                fluid.eos.thermal_energy_density(fluid.pre_code) * vol,
                dtype=float,
            )
            if old_internal is not None:
                first = int(par.mesh.ghost_cells)
                count = int(par.mesh.grid_cells)
                internal[first:first + count] = old_internal[first:first + count]
            fluid.InternalEnergy_code = as_named_array(np.maximum(internal, 0.0))
        if old_conserved is not None:
            inactive, old_mass, old_mom, old_energy = old_conserved
            fluid.Mass_code[inactive] = old_mass[inactive]
            fluid.Mom_code[inactive] = old_mom[inactive]
            fluid.Energy_code[inactive] = old_energy[inactive]
        if (
            dual_energy and old_internal is not None
            and getattr(fluid.eos, 'is_polytropic', False)
        ):
            eta2 = self._dual_energy_eta(par, 'dual_energy_eta2', 1.0e-1)
            conserved_mass = np.asarray(fluid.Mass_code, dtype=float)
            conserved_momentum = np.asarray(fluid.Mom_code, dtype=float)
            conserved_energy = np.asarray(fluid.Energy_code, dtype=float)
            conserved_kinetic = np.zeros_like(conserved_energy)
            np.divide(
                0.5 * conserved_momentum**2,
                conserved_mass,
                out=conserved_kinetic,
                where=conserved_mass > 0.0,
            )
            total_thermal = (
                conserved_energy - conserved_kinetic
                - self._rotational_energy_from_conserved(mesh, fluid, par)
            )
            total_fraction = np.divide(
                total_thermal,
                np.maximum(np.abs(conserved_energy), 1.0e-300),
                out=np.zeros_like(total_thermal),
                where=np.isfinite(total_thermal),
            )
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            physical = np.zeros(len(total_thermal), dtype=bool)
            physical[first:first + count] = True
            sync = (
                physical & np.isfinite(total_thermal)
                & (total_thermal > 0.0) & (total_fraction > eta2)
            )
            fluid.InternalEnergy_code[sync] = total_thermal[sync]
            self.dual_energy_synchronization_count += int(np.count_nonzero(sync))
        if verbose >= 2:
            print('fluid.Mass_code',fluid.Mass_code)
            print('fluid.Mom_code',fluid.Mom_code)
            print('fluid.Energy_code',fluid.Energy_code)
        
        
    def SetGradient(self, mesh, fluid):
        """Calculate centered gradients for density, velocity, and pressure."""
        xdelta = mesh.xdelta
        fluid.rho_code.grad = ru.CalGradient(fluid.rho_code, xdelta)
        fluid.vel_code.grad = ru.CalGradient(fluid.vel_code, xdelta)
        fluid.pre_code.grad = ru.CalGradient(fluid.pre_code, xdelta)
        if hasattr(fluid, 'specific_angular_momentum_code'):
            fluid.specific_angular_momentum_code.grad = ru.CalGradient(
                fluid.specific_angular_momentum_code, xdelta
            )

    @staticmethod
    def _positivity_limited_internal_flux(old_internal, flux, area, dt,
                                          physical):
        from .positivity import limit_internal_flux

        return limit_internal_flux(old_internal, flux, area, dt, physical)

    @staticmethod
    @staticmethod
    def _cfl_density_floor( *args, **kwargs):
        from .dual_energy import _cfl_density_floor

        return _cfl_density_floor(*args, **kwargs)

    @staticmethod
    @staticmethod
    def _dual_energy_enabled( *args, **kwargs):
        from .dual_energy import _dual_energy_enabled

        return _dual_energy_enabled(*args, **kwargs)

    @staticmethod
    @staticmethod
    def _rotational_energy_enabled( *args, **kwargs):
        from .dual_energy import _rotational_energy_enabled

        return _rotational_energy_enabled(*args, **kwargs)

    @staticmethod
    @staticmethod
    def _gravity_potential_energy_enabled( *args, **kwargs):
        from .dual_energy import _gravity_potential_energy_enabled

        return _gravity_potential_energy_enabled(*args, **kwargs)

    def _gravity_potential(self, *args, **kwargs):
        from .dual_energy import _gravity_potential

        return _gravity_potential(self, *args, **kwargs)

    def _gravity_potential_faces(self, *args, **kwargs):
        from .dual_energy import _gravity_potential_faces

        return _gravity_potential_faces(self, *args, **kwargs)

    def _rotational_energy_density(self, *args, **kwargs):
        from .dual_energy import _rotational_energy_density

        return _rotational_energy_density(self, *args, **kwargs)

    def _rotational_energy_from_conserved(self, *args, **kwargs):
        from .dual_energy import _rotational_energy_from_conserved

        return _rotational_energy_from_conserved(self, *args, **kwargs)

    @staticmethod
    @staticmethod
    def _dual_energy_eta( *args, **kwargs):
        from .dual_energy import _dual_energy_eta

        return _dual_energy_eta(*args, **kwargs)

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
            (fluid.rho_code.R, fluid.vel_code.R, fluid.pre_code.R),
            (fluid.rho_code.L, fluid.vel_code.L, fluid.pre_code.L),
        ):
            inactive = ~np.isfinite(density) | (density <= density_floor)
            density[inactive] = 0.0
            velocity[inactive] = 0.0
            pressure[inactive] = 0.0
        if order == 1:
            for density, velocity, pressure in (
                (fluid.rho_code.R.first, fluid.vel_code.R.first, fluid.pre_code.R.first),
                (fluid.rho_code.L.first, fluid.vel_code.L.first, fluid.pre_code.L.first),
            ):
                inactive = ~np.isfinite(density) | (density <= density_floor)
                density[inactive] = 0.0
                velocity[inactive] = 0.0
                pressure[inactive] = 0.0
        
        
    def SetConservedDensityFlux(self, fluid):
        """Store Euler fluxes and conserved densities on fluid arrays."""
        (
            fluid.Mass_code.F,
            fluid.Mass_code.q,
            fluid.Mom_code.F,
            fluid.Mom_code.q,
            fluid.Energy_code.F,
            fluid.Energy_code.q,
        ) = fluid.eos.fluxes(fluid.rho_code, fluid.vel_code, fluid.pre_code)

    @staticmethod
    @staticmethod
    def _set_angular_momentum_flux( *args, **kwargs):
        from .angular_momentum import _set_angular_momentum_flux

        return _set_angular_momentum_flux(*args, **kwargs)

    def _limit_angular_momentum_flux(self, *args, **kwargs):
        from .angular_momentum import _limit_angular_momentum_flux

        return _limit_angular_momentum_flux(self, *args, **kwargs)

    def _set_rotational_energy_flux(self, *args, **kwargs):
        from .angular_momentum import _set_rotational_energy_flux

        return _set_rotational_energy_flux(self, *args, **kwargs)

    def _apply_local_angular_energy_fallback(self, *args, **kwargs):
        from .angular_momentum import _apply_local_angular_energy_fallback

        return _apply_local_angular_energy_fallback(self, *args, **kwargs)
        
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
            fluid.rho_code.R = as_named_array(np.asarray(fluid.rho_code, dtype=float).copy())
            fluid.rho_code.L = ru.periodic_roll(fluid.rho_code, 1)
            fluid.vel_code.R = as_named_array(np.asarray(fluid.vel_code, dtype=float).copy())
            fluid.vel_code.L = ru.periodic_roll(fluid.vel_code, 1)
            fluid.pre_code.R = as_named_array(np.asarray(fluid.pre_code, dtype=float).copy())
            fluid.pre_code.L = ru.periodic_roll(fluid.pre_code, 1)
            if hasattr(fluid, 'specific_angular_momentum_code'):
                fluid.specific_angular_momentum_code.R = as_named_array(
                    np.asarray(fluid.specific_angular_momentum_code, dtype=float).copy()
                )
                fluid.specific_angular_momentum_code.L = ru.periodic_roll(
                    fluid.specific_angular_momentum_code, 1
                )
            if order == 1:
                self.SetGradient(mesh, fluid)
                fluid.rho_code.R.first, fluid.rho_code.L.first = ru.extrapolateToFace(fluid.rho_code, mesh.boundary, fluid.rho_code.grad, order=1)
                fluid.vel_code.R.first, fluid.vel_code.L.first = ru.extrapolateToFace(fluid.vel_code, mesh.boundary, fluid.vel_code.grad, order=1)
                fluid.pre_code.R.first, fluid.pre_code.L.first = ru.extrapolateToFace(fluid.pre_code, mesh.boundary, fluid.pre_code.grad, order=1)
                if hasattr(fluid, 'specific_angular_momentum_code'):
                    (
                        fluid.specific_angular_momentum_code.R.first,
                        fluid.specific_angular_momentum_code.L.first,
                    ) = ru.extrapolateToFace(
                        fluid.specific_angular_momentum_code,
                        mesh.boundary,
                        fluid.specific_angular_momentum_code.grad,
                        order=1,
                    )
                    # MUSCL reconstruction of j is a passive-scalar
                    # reconstruction, but j also enters the rotational-energy
                    # admissibility condition.  Keep both states at each face
                    # inside the local cell-average range so an antidiffusive
                    # gradient cannot create a new angular-momentum extremum.
                    j_right_cell = np.asarray(
                        fluid.specific_angular_momentum_code.R, dtype=float
                    )
                    j_left_cell = np.asarray(
                        fluid.specific_angular_momentum_code.L, dtype=float
                    )
                    j_min = np.minimum(j_left_cell, j_right_cell)
                    j_max = np.maximum(j_left_cell, j_right_cell)
                    fluid.specific_angular_momentum_code.R.first = as_named_array(
                        np.clip(
                            np.asarray(
                                fluid.specific_angular_momentum_code.R.first,
                                dtype=float,
                            ),
                            j_min,
                            j_max,
                        )
                    )
                    fluid.specific_angular_momentum_code.L.first = as_named_array(
                        np.clip(
                            np.asarray(
                                fluid.specific_angular_momentum_code.L.first,
                                dtype=float,
                            ),
                            j_min,
                            j_max,
                        )
                    )
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
        if par.boundary.condition != 'InflowSph':
            return
        first = int(par.mesh.ghost_cells)
        outer_face = first + int(par.mesh.grid_cells)
        if outer_face >= len(fluid.rho_code.R):
            return
        rho_background = float(
            par.boundary.inflow_density
        )
        velocity_background = float(
            par.boundary.inflow_velocity
        )
        pressure_background = float(
            fluid.eos.pressure(
                rho_background,
                float(
                    par.boundary.inflow_temperature
                ),
                float(par.boundary.inflow_mu),
            )
        )
        for quantity, value in (
            (fluid.rho_code, rho_background),
            (fluid.vel_code, velocity_background),
            (fluid.pre_code, pressure_background),
        ):
            quantity.R[outer_face] = value
            quantity.L[outer_face] = value
            if order == 1:
                quantity.R.first[outer_face] = value
                quantity.L.first[outer_face] = value
        if hasattr(fluid, 'specific_angular_momentum_code'):
            angular_momentum = float(getattr(
                par, 'specific_angular_momentum_inflow', 0.0
            ))
            fluid.specific_angular_momentum_code.R[outer_face] = angular_momentum
            fluid.specific_angular_momentum_code.L[outer_face] = angular_momentum
            if order == 1:
                fluid.specific_angular_momentum_code.R.first[outer_face] = angular_momentum
                fluid.specific_angular_momentum_code.L.first[outer_face] = angular_momentum

    @staticmethod
    def _vacuum_safe_primitive_state(rho, vel, pre):
        from .fluxes import vacuum_safe_primitive_state

        return vacuum_safe_primitive_state(rho, vel, pre)

    @staticmethod
    def _hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma):
        from .fluxes import hllc_flux

        return hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma)

    def _interface_fluxes(self, fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method):
        from .fluxes import interface_fluxes

        return interface_fluxes(
            fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method
        )

    def SetFluxOnFace(self,fluid,boundcond,order=0,par=None,method='Rusanov'):
        """Calculate mass, momentum, and energy fluxes at interfaces."""
        rho_L, vel_L, pre_L = self._vacuum_safe_primitive_state(
            fluid.rho_code.L, fluid.vel_code.L, fluid.pre_code.L
        )
        rho_R, vel_R, pre_R = self._vacuum_safe_primitive_state(
            fluid.rho_code.R, fluid.vel_code.R, fluid.pre_code.R
        )
        Mass_flux_0, Mom_flux_0, Energy_flux_0 = self._interface_fluxes(
            fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method
        )
        if order==0:
            fluid.Mass_code.flux, fluid.Mom_code.flux, fluid.Energy_code.flux = Mass_flux_0, Mom_flux_0, Energy_flux_0
            fluid.angular_momentum_mass_flux_low = as_named_array(Mass_flux_0.copy())
            fluid.angular_momentum_mom_flux_low = as_named_array(Mom_flux_0.copy())
            fluid.angular_momentum_energy_flux_low = as_named_array(Energy_flux_0.copy())
        elif order==1:
            rho_L, vel_L, pre_L = self._vacuum_safe_primitive_state(
                fluid.rho_code.L.first, fluid.vel_code.L.first, fluid.pre_code.L.first
            )
            rho_R, vel_R, pre_R = self._vacuum_safe_primitive_state(
                fluid.rho_code.R.first, fluid.vel_code.R.first, fluid.pre_code.R.first
            )
            Mass_flux_1, Mom_flux_1, Energy_flux_1 = self._interface_fluxes(
                fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method
            )
            self.SetConservedDensityFlux(fluid)
            limiter = getattr(par, 'flux_limiter', 'minmod') if par is not None else 'minmod'
            fluid.Mass_code.flux, fluid.philim_Mass_code = ru.ApplyFluxLimiter(
                fluid.Mass_code.q, Mass_flux_1, Mass_flux_0, limiter=limiter
            )
            fluid.Mom_code.flux, fluid.philim_Mom_code = ru.ApplyFluxLimiter(
                fluid.Mom_code.q, Mom_flux_1, Mom_flux_0, limiter=limiter
            )
            fluid.Energy_code.flux, fluid.philim_Energy_code = ru.ApplyFluxLimiter(
                fluid.Energy_code.q, Energy_flux_1, Energy_flux_0, limiter=limiter
            )
            fluid.angular_momentum_mass_flux_low = as_named_array(Mass_flux_0.copy())
            fluid.angular_momentum_mom_flux_low = as_named_array(Mom_flux_0.copy())
            fluid.angular_momentum_energy_flux_low = as_named_array(Energy_flux_0.copy())
            # A MUSCL reconstruction is not valid across a vacuum jump.  Use
            # the positivity-safe first-order flux on gas-vacuum faces; this
            # preserves injection into vacuum while retaining order one away
            # from the front.
            floor = self._cfl_density_floor(par)
            # Check the reconstructed states themselves.  A centered
            # reconstruction can overshoot across the imposed spherical wind
            # jump even when both cell-centered states are positive.  Testing
            # only ``rho_code.L/R`` therefore lets an inadmissible high-order
            # flux through when the positivity limiter is disabled.
            reconstructed_density = (
                np.asarray(fluid.rho_code.L.first, dtype=float),
                np.asarray(fluid.rho_code.R.first, dtype=float),
            )
            reconstructed_pressure = (
                np.asarray(fluid.pre_code.L.first, dtype=float),
                np.asarray(fluid.pre_code.R.first, dtype=float),
            )
            vacuum_face = np.zeros_like(
                reconstructed_density[0], dtype=bool
            )
            for state_density, state_pressure in zip(
                reconstructed_density, reconstructed_pressure
            ):
                vacuum_face |= (
                    ~np.isfinite(state_density)
                    | (state_density <= floor)
                    | ~np.isfinite(state_pressure)
                    | (state_pressure <= 0.0)
                )
            # A reconstructed face depends on neighboring cell gradients,
            # and each cell update depends on its two bounding faces.  Once
            # one reconstructed state is invalid, retain the complete local
            # stencil at first order; reverting only that face still allows
            # an adjacent high-order flux to combine with it and overshoot.
            vacuum_face |= np.roll(vacuum_face, -1)
            vacuum_face |= np.roll(vacuum_face, 1)
            fluid.Mass_code.flux[vacuum_face] = Mass_flux_0[vacuum_face]
            fluid.Mom_code.flux[vacuum_face] = Mom_flux_0[vacuum_face]
            fluid.Energy_code.flux[vacuum_face] = Energy_flux_0[vacuum_face]
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
        density = np.asarray(fluid.rho_code, dtype=float)
        first = int(par.mesh.ghost_cells)
        count = int(par.mesh.grid_cells)
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
        for flux in (fluid.Mass_code.flux, fluid.Mom_code.flux, fluid.Energy_code.flux):
            flux[face_mask] = 0.0

    @staticmethod
    def _positive_conserved_state(mass, momentum, energy, mass_floor=0.0,
                                  energy_floor=0.0, relative_tolerance=1.0e-12,
                                  angular_momentum=None, radius=None):
        from .positivity import positive_conserved_state

        return positive_conserved_state(
            mass, momentum, energy, mass_floor=mass_floor,
            energy_floor=energy_floor, relative_tolerance=relative_tolerance,
            angular_momentum=angular_momentum, radius=radius,
        )

    def _positivity_limited_face_fluxes(
        self, fluid, dt, mesh, par, mass_face, mom_face, energy_face,
        geometric_mom=None, angular_face=None,
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
            dt_value = float(np.asarray(dt, dtype=float))
            area = np.asarray(mesh.area, dtype=float)
            fluid.Mass_code += dt_value * (
                np.asarray(mass_face, dtype=float) * area
                - ru.periodic_roll(np.asarray(mass_face, dtype=float) * area, -1)
            )
            fluid.Mom_code += dt_value * (
                np.asarray(mom_face, dtype=float) * area
                - ru.periodic_roll(np.asarray(mom_face, dtype=float) * area, -1)
                + (np.asarray(geometric_mom, dtype=float)
                   if geometric_mom is not None else 0.0)
            )
            fluid.Energy_code += dt_value * (
                np.asarray(energy_face, dtype=float) * area
                - ru.periodic_roll(np.asarray(energy_face, dtype=float) * area, -1)
            )
            if angular_face is not None and hasattr(fluid, 'AngularMomentum_code'):
                angular_area = np.asarray(angular_face, dtype=float) * area
                fluid.AngularMomentum_code += dt_value * (
                    angular_area - ru.periodic_roll(angular_area, -1)
                )
            self._last_face_limiter_factors = np.ones_like(
                np.asarray(mass_face, dtype=float)
            )
            return 1.0
        dt_value = float(np.asarray(dt, dtype=float))
        mass = np.asarray(fluid.Mass_code, dtype=float).copy()
        momentum = np.asarray(fluid.Mom_code, dtype=float).copy()
        energy = np.asarray(fluid.Energy_code, dtype=float).copy()
        angular = (np.asarray(fluid.AngularMomentum_code, dtype=float).copy()
                   if angular_face is not None else None)
        radius = (
            np.abs(np.asarray(mesh.coordinate, dtype=float))
            if angular is not None and hasattr(mesh, 'coordinate')
            else None
        )
        count = len(mass)
        if par is None:
            first, last = 0, count
        else:
            first = int(par.mesh.ghost_cells)
            last = min(first + int(par.mesh.grid_cells), count)
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
            1.0e-6
            if self._dual_energy_enabled(par) and hasattr(fluid, 'InternalEnergy_code')
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
        fluid.Mass_code[vacuum] = 0.0
        fluid.Mom_code[vacuum] = 0.0
        fluid.Energy_code[vacuum] = 0.0
        if angular is not None:
            angular[vacuum] = 0.0

        def valid(mass_value, momentum_value, energy_value,
                  angular_value=None):
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
                angular_momentum=angular_value, radius=radius,
            )
            result[~physical] = True
            return result

        def cell_valid(index, mass_value, momentum_value, energy_value,
                       angular_value=None):
            """Check one trial cell without NumPy allocation."""
            if not (
                np.isfinite(mass_value)
                and np.isfinite(momentum_value)
                and np.isfinite(energy_value)
                and mass_value >= mass_floor[index]
            ):
                return False
            if mass_value <= max(mass_floor[index], 0.0):
                return energy_value >= energy_floor[index]
            kinetic_value = 0.5 * momentum_value**2 / mass_value
            rotational_value = 0.0
            if (
                angular_value is not None
                and radius is not None
                and radius[index] > 0.0
            ):
                rotational_value = 0.5 * angular_value**2 / (
                    mass_value * radius[index]**2
                )
            internal_value = energy_value - kinetic_value - rotational_value
            tolerance = relative_tolerance * max(
                abs(energy_value),
                kinetic_value,
                abs(energy_floor[index]),
                np.finfo(float).tiny,
            )
            # Leave a tiny margin for the different evaluation order used by
            # the final vectorized admissibility check.  This is only an
            # internal limiter margin; the public roundoff tolerance remains
            # exactly ``relative_tolerance``.
            tolerance *= 1.0 - 1.0e-8
            return internal_value >= energy_floor[index] - tolerance

        geometry_increment = np.zeros_like(momentum)
        geometry_fraction = np.ones(len(momentum), dtype=float)
        if geometric_mom is not None:
            # The spherical pressure geometry term is a momentum source while
            # total energy remains governed by the conservative energy flux.
            # In a nearly pressureless cell, applying the full source can
            # make K exceed E even though the pre-source state is admissible.
            # Limit only this local momentum increment to the first admissible
            # boundary; do not add compensating energy.
            geometry_increment = (
                dt_value * np.asarray(geometric_mom, dtype=float)
            )
            base_valid = valid(mass, momentum, energy, angular)
            full_geometry_momentum = momentum + geometry_increment
            full_valid = valid(mass, full_geometry_momentum, energy, angular)
            affected = physical & base_valid & ~full_valid
            for index in np.flatnonzero(affected):
                low, high = 0.0, 1.0
                for _ in range(48):
                    middle = 0.5 * (low + high)
                    trial_momentum = (
                        momentum[index] + middle * geometry_increment[index]
                    )
                    trial_valid = cell_valid(
                        index, mass[index], trial_momentum, energy[index],
                        angular[index] if angular is not None else None,
                    )
                    if trial_valid:
                        low = middle
                    else:
                        high = middle
                geometry_fraction[index] = low

        mass_face = np.asarray(mass_face, dtype=float)
        mom_face = np.asarray(mom_face, dtype=float)
        energy_face = np.asarray(energy_face, dtype=float)
        area = np.asarray(mesh.area, dtype=float)
        delta_mass = dt_value * mass_face * area
        delta_mom = dt_value * mom_face * area
        delta_energy = dt_value * energy_face * area
        delta_angular = (dt_value * np.asarray(angular_face, dtype=float) * area
                         if angular_face is not None else None)
        # Accept the unlimited conservative update immediately when possible.
        # This is the overwhelmingly common path and avoids limiter overhead.
        full_mass = mass + delta_mass - ru.periodic_roll(delta_mass, -1)
        full_mom = (
            momentum + geometry_increment
            + delta_mom - ru.periodic_roll(delta_mom, -1)
        )
        full_energy = energy + delta_energy - ru.periodic_roll(delta_energy, -1)
        full_angular = (
            angular + delta_angular - ru.periodic_roll(delta_angular, -1)
            if angular is not None else None
        )
        if np.all(valid(full_mass, full_mom, full_energy, full_angular)):
            factors = np.ones(len(mass_face), dtype=float)
            mass, momentum, energy = full_mass, full_mom, full_energy
            total_angular = full_angular
        else:
            # Construct the limited update from a known admissible state.
            # Increasing one face coefficient changes only its two adjacent
            # cells with equal-and-opposite corrections.  Accept an increase
            # only while both cells remain admissible, so global admissibility
            # is an invariant of the construction rather than something a
            # fixed number of repair passes must recover afterward.
            factors = np.zeros(len(mass_face), dtype=float)
            momentum = momentum + geometry_fraction * geometry_increment
            total_mass = mass.copy()
            total_mom = momentum.copy()
            total_energy = energy.copy()
            total_angular = angular.copy() if angular is not None else None
            if not np.all(valid(total_mass, total_mom, total_energy,
                                total_angular)):
                invalid = ~valid(total_mass, total_mom, total_energy,
                                  total_angular)
                index = int(np.flatnonzero(invalid)[0])
                mass_value = float(total_mass[index])
                momentum_value = float(total_mom[index])
                energy_value = float(total_energy[index])
                kinetic_value = (
                    0.5 * momentum_value**2 / mass_value
                    if mass_value > 0.0 else 0.0
                )
                angular_value = (
                    float(total_angular[index])
                    if total_angular is not None else 0.0
                )
                radius_value = (
                    float(radius[index])
                    if radius is not None else float("nan")
                )
                rotational_value = (
                    0.5 * angular_value**2
                    / (mass_value * radius_value**2)
                    if mass_value > 0.0 and radius_value > 0.0 else 0.0
                )
                internal_value = (
                    energy_value - kinetic_value - rotational_value
                )
                specific_value = (
                    angular_value / mass_value if mass_value > 0.0 else 0.0
                )
                raise ValueError(
                    'hydro state is outside positivity domain before paired '
                    'face construction at cell %d '
                    '(mass=%s mom=%s energy=%s radius=%s J=%s j=%s '
                    'kinetic=%s rotational=%s internal=%s)'
                    % (index, mass_value, momentum_value, energy_value,
                       radius_value, angular_value, specific_value,
                       kinetic_value, rotational_value, internal_value)
                )

            def adjacent_valid(face, factor):
                """Check only the two cells changed by one face trial."""
                left = (face - 1) % count
                right = face
                increment = factor - factors[face]
                trial_mass_left = total_mass[left] - increment * delta_mass[face]
                trial_mom_left = total_mom[left] - increment * delta_mom[face]
                trial_energy_left = (
                    total_energy[left] - increment * delta_energy[face]
                )
                trial_mass_right = total_mass[right] + increment * delta_mass[face]
                trial_mom_right = total_mom[right] + increment * delta_mom[face]
                trial_energy_right = (
                    total_energy[right] + increment * delta_energy[face]
                )
                trial_angular_left = (
                    total_angular[left] - increment * delta_angular[face]
                    if total_angular is not None else None
                )
                trial_angular_right = (
                    total_angular[right] + increment * delta_angular[face]
                    if total_angular is not None else None
                )
                indices = [index for index in (left, right) if physical[index]]
                if not indices:
                    return True
                for index in indices:
                    trial_mass_value = (
                        trial_mass_left if index == left else trial_mass_right
                    )
                    trial_mom_value = (
                        trial_mom_left if index == left else trial_mom_right
                    )
                    trial_energy_value = (
                        trial_energy_left
                        if index == left else trial_energy_right
                    )
                    trial_angular_value = (
                        trial_angular_left
                        if index == left else trial_angular_right
                    )
                    if not cell_valid(
                        index,
                        trial_mass_value,
                        trial_mom_value,
                        trial_energy_value,
                        trial_angular_value,
                    ):
                        return False
                return True

            # Alternate traversal direction to reduce ordering bias.  The
            # first sweep already produces a globally admissible update;
            # later sweeps only recover additional face flux monotonically.
            # Keep the recovery policy independent of geometry and energy
            # formulation; reducing it for cold spherical dual-energy runs
            # would be a performance heuristic rather than a numerical
            # criterion and could leave more ordering-dependent limiting.
            max_recovery_sweeps = 8
            recovery_iterations = 48
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
                    if adjacent_valid(face, 1.0):
                        accepted = 1.0
                    else:
                        # The current coefficient is known admissible.  Search
                        # only upward, never stepping outside the invariant
                        # domain as the previous reduce-and-repair scheme did.
                        low, high = current, 1.0
                        for _ in range(recovery_iterations):
                            middle = 0.5 * (low + high)
                            if adjacent_valid(face, middle):
                                low = middle
                            else:
                                high = middle
                        accepted = low
                    largest_increase = max(
                        largest_increase, accepted - current
                    )
                    increment = accepted - current
                    left = (face - 1) % count
                    right = face
                    total_mass[left] -= increment * delta_mass[face]
                    total_mom[left] -= increment * delta_mom[face]
                    total_energy[left] -= increment * delta_energy[face]
                    total_mass[right] += increment * delta_mass[face]
                    total_mom[right] += increment * delta_mom[face]
                    total_energy[right] += increment * delta_energy[face]
                    if total_angular is not None:
                        total_angular[left] -= increment * delta_angular[face]
                        total_angular[right] += increment * delta_angular[face]
                    factors[face] = accepted
                if np.all(factors >= 1.0 - factor_tolerance):
                    factors[...] = 1.0
                    break
                if largest_increase <= factor_tolerance:
                    break
            mass, momentum, energy = total_mass, total_mom, total_energy

        # A WindSph density floor is a wind reservoir, not an independent
        # thermal-energy repair.  If a limited update leaves a physical cell
        # below the requested floor, replenish the missing mass with the
        # same specific momentum and total energy as the imposed wind.
        # Applying all three increments together preserves the meaning of the
        # floor and keeps the conservative state check below authoritative.
        boundary = getattr(par, 'boundary', None)
        if boundary is not None and getattr(boundary, 'condition', None) == 'WindSph':
            wind_density = float(np.asarray(boundary.outflow_density))
            wind_velocity = float(np.asarray(boundary.outflow_velocity))
            wind_pressure = float(np.asarray(fluid.eos.pressure(
                boundary.outflow_density,
                boundary.outflow_temperature,
                boundary.outflow_mu,
            )))
            wind_internal = (
                wind_pressure / wind_density / (fluid.eos.gamma - 1.0)
                if wind_density > 0.0 and not fluid.eos.is_isothermal
                else 0.0
            )
            wind_specific_energy = 0.5 * wind_velocity**2 + wind_internal

            missing_mass = np.maximum(mass_floor - mass, 0.0)
            reservoir_cells = physical & (missing_mass > 0.0)
            if np.any(reservoir_cells):
                momentum[reservoir_cells] += (
                    missing_mass[reservoir_cells] * wind_velocity
                )
                energy[reservoir_cells] += (
                    missing_mass[reservoir_cells] * wind_specific_energy
                )
                mass[reservoir_cells] += missing_mass[reservoir_cells]
                self._last_wind_reservoir_mass = float(
                    np.sum(missing_mass[reservoir_cells])
                )
            else:
                self._last_wind_reservoir_mass = 0.0

            # A spherical pressure update can leave a cell exactly on the
            # density floor with a small kinetic-energy deficit.  In that
            # case add the smallest further parcel of the same wind state
            # that restores the conservative invariant.  This is still a
            # coupled reservoir operation; it is not a thermal-energy patch.
            kinetic = np.divide(
                0.5 * momentum**2,
                mass,
                out=np.zeros_like(mass),
                where=mass > 0.0,
            )
            floor_edge = physical & (mass <= mass_floor * (1.0 + 1.0e-12))
            energy_deficit = floor_edge & (
                energy < kinetic + energy_floor
            )
            for index in np.flatnonzero(energy_deficit):
                base_mass = mass[index]
                base_momentum = momentum[index]
                base_energy = energy[index]
                required_energy = energy_floor[index]

                def reservoir_valid(delta_mass):
                    trial_mass = base_mass + delta_mass
                    trial_momentum = base_momentum + delta_mass * wind_velocity
                    trial_energy = base_energy + delta_mass * wind_specific_energy
                    return trial_energy - (
                        0.5 * trial_momentum**2 / trial_mass
                    ) >= required_energy

                low = 0.0
                high = max(base_mass, mass_floor[index], 1.0) * 1.0e-12
                for _ in range(96):
                    if reservoir_valid(high):
                        break
                    high *= 2.0
                if not reservoir_valid(high):
                    raise ValueError(
                        'wind reservoir could not restore conservative '
                        'energy admissibility at cell %d' % index
                    )
                for _ in range(64):
                    middle = 0.5 * (low + high)
                    if reservoir_valid(middle):
                        high = middle
                    else:
                        low = middle
                mass[index] += high
                momentum[index] += high * wind_velocity
                energy[index] += high * wind_specific_energy
                self._last_wind_reservoir_mass += high

        if not np.all(valid(mass, momentum, energy, total_angular)):
            invalid = ~valid(mass, momentum, energy, total_angular)
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                'hydro state is outside positivity domain after face update '
                'at cell %d (mass=%s mom=%s energy=%s)' %
                (index, mass[index], momentum[index], energy[index])
            )

        fluid.Mass_code[...] = mass
        fluid.Mom_code[...] = momentum
        fluid.Energy_code[...] = energy
        if total_angular is not None:
            fluid.AngularMomentum_code[...] = total_angular
        self._last_face_limiter_factors = factors
        return float(np.min(factors)) if factors.size else 1.0

    def _apply_wind_reservoir_flux(self, dt, mesh, fluid, par):
        """Restore rejected WindSph boundary flux as one coupled parcel."""
        if (
            par is None
            or getattr(getattr(par, 'boundary', None), 'condition', None)
            != 'WindSph'
        ):
            return 0.0
        factors = np.asarray(
            getattr(self, '_last_face_limiter_factors', np.ones(0)),
            dtype=float,
        )
        first = int(par.mesh.ghost_cells)
        if first >= len(factors) or first >= len(mesh.area):
            return 0.0
        rejected_fraction = max(0.0, 1.0 - float(factors[first]))
        if rejected_fraction <= 0.0:
            return 0.0

        boundary = par.boundary
        rho_wind = float(np.asarray(boundary.outflow_density))
        velocity_wind = float(np.asarray(boundary.outflow_velocity))
        pressure_wind = float(np.asarray(fluid.eos.pressure(
            boundary.outflow_density,
            boundary.outflow_temperature,
            boundary.outflow_mu,
        )))
        wind_internal = (
            pressure_wind / rho_wind / (fluid.eos.gamma - 1.0)
            if rho_wind > 0.0 and not fluid.eos.is_isothermal
            else 0.0
        )
        wind_specific_energy = 0.5 * velocity_wind**2 + wind_internal
        area = float(np.asarray(mesh.area[first]))
        dt_value = float(np.asarray(dt))
        mass_rate = rho_wind * velocity_wind * area
        momentum_rate = (rho_wind * velocity_wind**2 + pressure_wind) * area
        energy_rate = velocity_wind * (
            0.5 * rho_wind * velocity_wind**2
            + fluid.eos.gamma * pressure_wind / (fluid.eos.gamma - 1.0)
        ) * area
        correction_mass = rejected_fraction * dt_value * mass_rate
        correction_momentum = rejected_fraction * dt_value * momentum_rate
        correction_energy = rejected_fraction * dt_value * energy_rate
        fluid.Mass_code[first] += correction_mass
        fluid.Mom_code[first] += correction_momentum
        fluid.Energy_code[first] += correction_energy
        if hasattr(fluid, 'InternalEnergy_code'):
            correction_internal = correction_energy - (
                velocity_wind * correction_momentum
                - 0.5 * velocity_wind**2 * correction_mass
            )
            fluid.InternalEnergy_code[first] += correction_internal
        mass = np.asarray(fluid.Mass_code, dtype=float)
        momentum = np.asarray(fluid.Mom_code, dtype=float)
        energy = np.asarray(fluid.Energy_code, dtype=float)
        kinetic = 0.5 * momentum[first]**2 / mass[first]
        if energy[first] < kinetic:
            # A rejected parcel can still be too fast for the receiving cell
            # when its velocity differs substantially from the wind.  Add
            # only the minimum further parcel of the same wind state needed
            # to make the combined conservative state admissible.
            base_mass = mass[first]
            base_momentum = momentum[first]
            base_energy = energy[first]

            def reservoir_valid(delta_mass):
                trial_mass = base_mass + delta_mass
                trial_momentum = base_momentum + delta_mass * velocity_wind
                trial_energy = base_energy + delta_mass * wind_specific_energy
                return trial_energy >= (
                    0.5 * trial_momentum**2 / trial_mass
                )

            low = 0.0
            high = max(base_mass, correction_mass, 1.0) * 1.0e-12
            for _ in range(96):
                if reservoir_valid(high):
                    break
                high *= 2.0
            if not reservoir_valid(high):
                raise ValueError(
                    'WindSph reservoir correction could not restore '
                    'conservative energy at cell %d' % first
                )
            for _ in range(64):
                middle = 0.5 * (low + high)
                if reservoir_valid(middle):
                    high = middle
                else:
                    low = middle
            mass[first] += high
            momentum[first] += high * velocity_wind
            energy[first] += high * wind_specific_energy
            if hasattr(fluid, 'InternalEnergy_code'):
                fluid.InternalEnergy_code[first] += high * (
                    wind_specific_energy
                    - velocity_wind**2
                    + 0.5 * velocity_wind**2
                )
            self._last_wind_reservoir_mass += high
        kinetic = 0.5 * momentum[first]**2 / mass[first]
        if not (
            np.isfinite(mass[first])
            and np.isfinite(momentum[first])
            and np.isfinite(energy[first])
            and mass[first] > 0.0
            and energy[first] >= kinetic
        ):
            raise ValueError(
                'WindSph reservoir correction produced an inadmissible '
                'conserved state at cell %d' % first
            )
        self._last_wind_reservoir_mass = (
            getattr(self, '_last_wind_reservoir_mass', 0.0) + correction_mass
        )
        return correction_mass
        
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
                fluid.cmax = np.maximum(fluid.vsignal_code, ru.periodic_roll(fluid.vsignal_code, 1))
            else:  # HLLC uses Rusanov speeds for CFL and vacuum fallback.
                fluid.cmax = np.maximum(fluid.vsignal_code, ru.periodic_roll(fluid.vsignal_code, 1))
            
            self.SetFaceLR(mesh,fluid, boundcond, order=order)
            self.SetFluxOnFace(
                fluid, boundcond, order=order, par=getattr(mesh, '_par', None), method=method
            )
            self._apply_low_density_flux_mask(
                fluid, getattr(mesh, '_par', None)
            )
            self._apply_hydrostatic_core_flux(fluid, getattr(mesh, '_par', None))
            self._zero_spherical_origin_flux(mesh, fluid)
            self._apply_local_angular_energy_fallback(
                mesh, fluid, getattr(mesh, '_par', None)
            )
            angular_momentum_face = self._set_angular_momentum_flux(
                fluid, order=order
            )
            self._set_rotational_energy_flux(
                mesh,
                fluid,
                getattr(mesh, '_par', None),
                j_face=angular_momentum_face,
            )
            # Optional fluxes are constructed after the primary hydro fluxes;
            # enforce the exact-origin condition once more at the end so a
            # later reconstruction cannot repopulate that face.
            self._zero_spherical_origin_flux(mesh, fluid)
        else:
            raise ValueError("Interface flux method unknown: %s"%method) 
        if (verbose>=2):
            print('fluid.Mass_code.flux',fluid.Mass_code.flux)
            print('fluid.Mom_code.flux',fluid.Mom_code.flux)
            print('fluid.Energy_code.flux',fluid.Energy_code.flux)
            
            
    def AddFluxes(self, dt: float, mesh, fluid, boundcond):
        """Apply interface fluxes to conserved quantities and advance time."""
        old_mass_for_internal = np.asarray(fluid.Mass_code, dtype=float).copy()
        self._limit_angular_momentum_flux(
            dt, mesh, fluid, getattr(mesh, '_par', None)
        )
        # Shift the face fluxes so each cell receives the net in-flow minus
        # out-flow through its two bounding faces.
        area = mesh.area
        df_Mass_code = fluid.Mass_code.flux * area - ru.periodic_roll(fluid.Mass_code.flux * area, -1)
        df_Mom_code = fluid.Mom_code.flux * area - ru.periodic_roll(fluid.Mom_code.flux * area, -1)
        df_Energy_code = fluid.Energy_code.flux * area - ru.periodic_roll(fluid.Energy_code.flux * area, -1)
        df_AngularMomentum = None
        if hasattr(fluid, 'AngularMomentum_code'):
            angular_flux_area = fluid.AngularMomentum_code.flux * area
            df_AngularMomentum = (
                angular_flux_area - ru.periodic_roll(angular_flux_area, -1)
            )
        potential_face = self._gravity_potential_faces(mesh, getattr(mesh, '_par', None))
        df_potential = None
        if potential_face is not None:
            potential_flux_area = potential_face * fluid.Mass_code.flux * area
            df_potential = (
                potential_flux_area
                - ru.periodic_roll(potential_flux_area, -1)
            )
        if getattr(mesh, 'coordsys', None) == 'spherical':
            # Spherical momentum needs the geometric pressure term from the
            # changing face area, not just the flux divergence.
            area_right = ru.periodic_roll(area, -1)
            df_Mom_code += fluid.pre_code * (area_right - area)

        dual_energy = (
            self._dual_energy_enabled(getattr(mesh, '_par', None))
            and hasattr(fluid, 'InternalEnergy_code')
            and getattr(fluid.eos, 'is_polytropic', False)
        )
        df_InternalEnergy = None
        if dual_energy:
            velocity_left = np.asarray(fluid.vel_code.L, dtype=float)
            velocity_right = np.asarray(fluid.vel_code.R, dtype=float)
            face_velocity = np.where(
                0.5 * (velocity_left + velocity_right) >= 0.0,
                velocity_left,
                velocity_right,
            )
            # Decompose the already computed Riemann total-energy flux into
            # internal and kinetic parts.  This carries the shock information
            # in the Riemann flux into the dual internal-energy update instead
            # of using a separately reconstructed upwind pressure flux.
            mass_flux = np.asarray(fluid.Mass_code.flux, dtype=float)
            momentum_flux = np.asarray(fluid.Mom_code.flux, dtype=float)
            total_energy_flux = np.asarray(fluid.Energy_code.flux, dtype=float)
            if hasattr(fluid, 'rotational_energy_flux'):
                total_energy_flux -= np.asarray(
                    fluid.rotational_energy_flux, dtype=float
                )
            internal_flux = (
                total_energy_flux
                - face_velocity * momentum_flux
                + 0.5 * face_velocity**2 * mass_flux
            )
            face_pressure = momentum_flux - face_velocity * mass_flux
            origin_face = self._spherical_origin_face_index(mesh)
            if origin_face is not None:
                internal_flux[origin_face] = 0.0
            df_InternalEnergy = (
                internal_flux * area
                - ru.periodic_roll(internal_flux * area, -1)
            )
            if getattr(mesh, 'coordsys', None) == 'spherical':
                # Account for spherical pressure work using the same
                # interface pressure implied by the Riemann momentum flux.
                df_InternalEnergy -= (
                    ru.periodic_roll(face_pressure * face_velocity * area, -1)
                    - face_pressure * face_velocity * area
                )

        par = getattr(mesh, '_par', None)
        geometric_mom = None
        if getattr(mesh, 'coordsys', None) == 'spherical':
            area_right = ru.periodic_roll(area, -1)
            geometric_mom = fluid.pre_code * (area_right - area)
        positivity_factor = self._positivity_limited_face_fluxes(
            fluid, dt, mesh, par,
            fluid.Mass_code.flux, fluid.Mom_code.flux, fluid.Energy_code.flux,
            geometric_mom=geometric_mom,
            angular_face=(fluid.AngularMomentum_code.flux
                          if df_AngularMomentum is not None else None),
        )
        # A positivity reduction at the prescribed wind face is a numerical
        # rejection of reservoir material, not a physical reduction of the
        # stellar-wind luminosity.  Reinsert the rejected parcel with its
        # matching mass, momentum, and energy before synchronizing primitives.
        self._apply_wind_reservoir_flux(
            dt, mesh, fluid, getattr(mesh, '_par', None)
        )
        if df_InternalEnergy is not None:
            # Couple the dual-energy advection to the same face coefficients
            # used by the conservative update.  Applying the minimum face
            # coefficient globally defeats the purpose of the local limiter.
            factors = np.asarray(
                getattr(self, '_last_face_limiter_factors',
                        np.ones(len(fluid.Mass_code.flux))),
                dtype=float,
            )
            limited_internal_flux = np.asarray(internal_flux, dtype=float) * factors
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            physical = np.zeros(len(fluid.InternalEnergy_code), dtype=bool)
            physical[first:first + count] = True
            internal_factors = self._positivity_limited_internal_flux(
                fluid.InternalEnergy_code,
                limited_internal_flux,
                area,
                dt,
                physical,
            )
            limited_internal_flux *= internal_factors
            limited_df_internal = (
                limited_internal_flux * area
                - ru.periodic_roll(limited_internal_flux * area, -1)
            )
            if getattr(mesh, 'coordsys', None) == 'spherical':
                # Retain the established spherical pressure-work
                # discretization.  The positivity limiter acts on the
                # Riemann internal-energy flux above; changing the geometric
                # source and limiting it as a scalar face flux simultaneously
                # can over-limit cold expanding cells.
                limited_df_internal -= fluid.pre_code * (
                    ru.periodic_roll(
                        factors * face_velocity * area, -1
                    ) - factors * face_velocity * area
                )
            candidate_internal = (
                np.asarray(fluid.InternalEnergy_code, dtype=float)
                + limited_df_internal * dt
            )
            previous_internal = np.asarray(fluid.InternalEnergy_code, dtype=float)

            # Do not silently turn an unsuccessful dual-energy update into a
            # pressureless cell.  The conservative update has already been
            # positivity-limited, so recover its thermal energy whenever
            # E-K is a strictly positive, finite estimate.  This is the same
            # fallback used by SetPrimitive, but doing it here prevents a
            # zero InternalEnergy value from surviving until the next
            # synchronization and generating a deep entropy spike.
            mass = np.asarray(fluid.Mass_code, dtype=float)
            momentum = np.asarray(fluid.Mom_code, dtype=float)
            total_energy = np.asarray(fluid.Energy_code, dtype=float)
            conservative_internal = np.zeros_like(total_energy)
            np.divide(
                0.5 * momentum**2,
                mass,
                out=conservative_internal,
                where=mass > 0.0,
            )
            conservative_internal = total_energy - conservative_internal
            conservative_internal -= self._rotational_energy_from_conserved(
                mesh, fluid, getattr(mesh, '_par', None)
            )
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            physical = np.zeros(len(candidate_internal), dtype=bool)
            physical[first:first + count] = True
            fallback = (
                physical
                & (~np.isfinite(candidate_internal) | (candidate_internal <= 0.0))
                & np.isfinite(conservative_internal)
                & (conservative_internal > 0.0)
            )
            if np.any(fallback):
                candidate_internal[fallback] = conservative_internal[fallback]
                self.dual_energy_pressure_fallback_count += int(
                    np.count_nonzero(fallback)
                )
            # A failed pressure-work update does not make the previous dual
            # state unphysical.  If E-K is also temporarily unusable, retain
            # that previous positive estimate for this step.  This avoids
            # injecting the pressure floor merely because both *post-update*
            # estimates crossed zero during a highly converging HLLC step.
            # The total-energy field remains authoritative and unchanged.
            retain_previous = (
                physical
                & (~np.isfinite(candidate_internal) | (candidate_internal <= 0.0))
                & ~fallback
                & np.isfinite(previous_internal)
                & (previous_internal > 0.0)
            )
            if np.any(retain_previous):
                candidate_internal[retain_previous] = previous_internal[retain_previous]
                self.dual_energy_pressure_fallback_count += int(
                    np.count_nonzero(retain_previous)
                )

            # A positivity limiter alone can still leave a tiny positive
            # value after a large cancellation in the spherical pressure-work
            # update.  Treat an abrupt loss below the configured consistency
            # fraction as a failed dual estimate as well.  Prefer E-K when it
            # is admissible; otherwise keep the previous positive dual state.
            consistency_factor = max(0.0, float(np.asarray(getattr(
                par, 'dual_energy_consistency_factor', 1.0e-1), dtype=float)))
            far_below_previous = (
                physical & np.isfinite(candidate_internal)
                & np.isfinite(previous_internal)
                & (previous_internal > 0.0)
                & (candidate_internal < consistency_factor * previous_internal)
            )
            conservative_recovery = (
                far_below_previous
                & np.isfinite(conservative_internal)
                & (conservative_internal > 0.0)
            )
            if np.any(conservative_recovery):
                candidate_internal[conservative_recovery] = (
                    conservative_internal[conservative_recovery]
                )
                self.dual_energy_pressure_fallback_count += int(
                    np.count_nonzero(conservative_recovery)
                )
            retain_consistent = far_below_previous & ~conservative_recovery
            if np.any(retain_consistent):
                candidate_internal[retain_consistent] = (
                    previous_internal[retain_consistent]
                )
                self.dual_energy_pressure_fallback_count += int(
                    np.count_nonzero(retain_consistent)
                )

            # Entropy-stable dual-energy correction for smooth cells.  For an
            # adiabatic ideal gas, the cell entropy proxy is proportional to
            # e/rho**gamma.  The Riemann internal-energy update may lose this
            # quantity through cancellation in the spherical pressure-work
            # term, even while remaining positive.  Preserve the previous
            # entropy only for moderate density changes; strong compression,
            # expansion, and near-vacuum cells are left to the conservative
            # consistency/fallback logic above.
            if (
                getattr(par, 'dual_energy_entropy_limiter', True)
                and not self._thermochemistry_enabled(fluid, par)
            ):
                volume = np.asarray(mesh.vol, dtype=float)
                old_density = np.divide(
                    old_mass_for_internal,
                    volume,
                    out=np.zeros_like(old_mass_for_internal),
                    where=volume > 0.0,
                )
                new_density = np.divide(
                    np.asarray(fluid.Mass_code, dtype=float),
                    volume,
                    out=np.zeros_like(old_density),
                    where=volume > 0.0,
                )
                density_ratio = np.divide(
                    new_density,
                    old_density,
                    out=np.ones_like(old_density),
                    where=old_density > 0.0,
                )
                moderate_density_change = (
                    physical & (density_ratio >= 0.5) & (density_ratio <= 2.0)
                )
                isentropic_internal = previous_internal * np.maximum(
                    density_ratio, 0.0
                ) ** float(fluid.eos.gamma)
                entropy_limited = (
                    moderate_density_change
                    & np.isfinite(previous_internal)
                    & (previous_internal > 0.0)
                    & np.isfinite(isentropic_internal)
                    & (isentropic_internal > 0.0)
                    & (candidate_internal < isentropic_internal)
                )
                if np.any(entropy_limited):
                    candidate_internal[entropy_limited] = (
                        isentropic_internal[entropy_limited]
                    )
                    self.dual_energy_entropy_limiter_count += int(
                        np.count_nonzero(entropy_limited)
                    )

            # Leave unresolved cells at zero only when the conservative state
            # is also non-positive.  SetPrimitive will then apply the
            # configured positive floor and record that injected energy.
            fluid.InternalEnergy_code = as_named_array(
                np.maximum(candidate_internal, 0.0)
            )
        if df_potential is not None:
            factors = np.asarray(
                getattr(self, '_last_face_limiter_factors',
                        np.ones(len(fluid.Mass_code.flux))),
                dtype=float,
            )
            limited_potential_flux_area = (
                potential_face * fluid.Mass_code.flux * factors * area
            )
            fluid.GravitationalPotentialEnergy_code += dt * (
                limited_potential_flux_area
                - ru.periodic_roll(limited_potential_flux_area, -1)
            )
        # advance time
        fluid.time += dt

    def _gravity_model(self, *args, **kwargs):
        from .gravity_sources import _gravity_model

        return _gravity_model(self, *args, **kwargs)

    def ApplyGravity(self, *args, **kwargs):
        from .gravity_sources import ApplyGravity

        return ApplyGravity(self, *args, **kwargs)

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
        grid_cells = int(par.mesh.grid_cells)
        if absorbed.shape[1] != grid_cells:
            raise ValueError("absorbed photon rate must contain physical cells only")

        rho_cgs = np.asarray(fluid.rho_code[interior], dtype=float) * scales["density_g_cm3"]
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
        momentum = fluid.Mom_code[interior]
        energy = fluid.Energy_code[interior]
        rho_code = fluid.rho_code[interior]
        velocity = fluid.vel_code[interior]
        momentum[valid] += rho_code[valid] * acceleration[valid] * volume[valid] * dt
        energy[valid] += (
            rho_code[valid]
            * velocity[valid]
            * acceleration[valid]
            * volume[valid]
            * dt
        )
        return 1


    def SetBoundary(self, mesh, fluid, par):
        """Fill ghost cells according to the selected boundary condition."""
        self.ApplyHydrostaticCore(mesh, fluid, par)
        btype = par.boundary.condition
        code_units = getattr(par, 'CodeUnits', None)
        scales = code_unit_scales(code_units)
        noghost = int(par.mesh.ghost_cells)
        nogrid = int(par.mesh.grid_cells)
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
        elif btype == 'WindSph':
            self._apply_wind_spherical_boundary(
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
        from .timestep import get_time_step

        return get_time_step(self, mesh, fluid, par, CFL=CFL)
