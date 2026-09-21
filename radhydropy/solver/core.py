# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Finite-volume hydrodynamics solver operations."""

import logging
import math
from types import SimpleNamespace

import numpy as np

import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
import radhydropy.utils as ru
from radhydropy.arrays import as_named_array
from radhydropy.constants import (
    DEFAULT_SIGMA_GAMMA_CGS_CM2,
    PROTON_MASS_CGS,
)
from radhydropy.diagnostic_logging import log_diagnostic
from radhydropy.runtime_fields import (
    select_fluid_primitive_arrays,
    select_mesh_geometry_arrays,
)
from radhydropy.units import (
    CGS_AREA_UNIT,
    CGS_PHOTON_FLUX_UNIT,
    CGS_RATE_UNIT,
    _as_cgs_float,
    _code_units,
    code_quantity_to_cgs,
    code_unit_scales,
)

# The paired-face positivity recovery path calls ``cell_valid`` many times.
# Keep this scalar constant out of that hot loop; constructing a NumPy finfo
# object for every trial is unnecessary overhead.
_FLOAT_TINY = np.finfo(float).tiny


def _positive_quadratic_root(a, b, c):
    """Return the smallest nonnegative root of ``a*x²+b*x+c``."""
    if abs(a) <= _FLOAT_TINY:
        if b >= 0.0:
            return None
        root = -c / b
    else:
        discriminant = b**2 - 4.0 * a * c
        if discriminant < 0.0:
            return None
        roots = [
            candidate
            for candidate in (
                (-b - math.sqrt(discriminant)) / (2.0 * a),
                (-b + math.sqrt(discriminant)) / (2.0 * a),
            )
            if candidate >= 0.0
        ]
        if not roots:
            return None
        root = min(roots)
    return root if math.isfinite(root) else None


def _analytical_face_increment_limit(
    faces,
    face,
    max_increment,
    total_mass,
    total_mom,
    total_energy,
    delta_mass,
    delta_mom,
    delta_energy,
    physical,
    mass_floor,
    energy_floor,
    total_angular,
    radius,
    delta_angular,
):
    largest = max_increment
    for index, sign in zip(faces, (-1.0, 1.0), strict=True):
        if not physical[index]:
            continue
        mass0 = float(total_mass[index])
        mom0 = float(total_mom[index])
        energy0 = float(total_energy[index])
        mass_limit = float(mass_floor[index])
        energy_limit = float(energy_floor[index])
        if not all(math.isfinite(value) for value in (mass0, mom0, energy0)) or mass0 < mass_limit:
            return None
        dm = sign * float(delta_mass[face])
        dp = sign * float(delta_mom[face])
        de = sign * float(delta_energy[face])
        mass_bound = _bounded_face_mass_increment(
            dm,
            max_increment,
            mass0,
            mass_limit,
        )
        if mass_bound < 0.0:
            return None
        angular0 = angular_delta = radius_value = 0.0
        if total_angular is not None:
            radius_value = float(radius[index])
            if radius_value <= 0.0 or not math.isfinite(radius_value):
                return None
            angular0 = float(total_angular[index])
            angular_delta = sign * float(delta_angular[face])
        rotational_factor = 1.0 / radius_value**2 if total_angular is not None else 0.0
        energy0 -= energy_limit
        a = dm * de - 0.5 * dp**2 - 0.5 * rotational_factor * angular_delta**2
        b = mass0 * de + dm * energy0 - mom0 * dp - rotational_factor * angular0 * angular_delta
        c = mass0 * energy0 - 0.5 * mom0**2 - 0.5 * rotational_factor * angular0**2
        if c < 0.0:
            return None
        if (a * mass_bound + b) * mass_bound + c >= 0.0:
            continue
        root = _positive_quadratic_root(a, b, c)
        if root is None:
            return None
        largest = min(largest, root)
    return largest


def _bounded_face_mass_increment(dm, max_increment, mass0, mass_limit):
    if dm < 0.0:
        return min(max_increment, (mass0 - mass_limit) / -dm)
    return max_increment


def _store_radiative_transfer_result(result, fluid, density_runtime_code, interior, scales):
    photon_density_code = (
        np.asarray(result.cell_photon_density, dtype=float) / scales["number_density_cgs_cm3"]
    )
    if np.ndim(photon_density_code) != 2:  # noqa: PLR2004
        raise ValueError("radiative-transfer result must have shape (ngroup, ncell)")
    expected_shape = (photon_density_code.shape[0], len(density_runtime_code))
    if np.shape(fluid.ngamma_code) != expected_shape:
        fluid.ngamma_code = np.zeros(expected_shape, dtype=float)
    fluid.ngamma_code[:, interior] = photon_density_code
    return result


def _trace_radiative_transfer(
    par,
    mesh,
    fluid,
    code_units,
    scales,
    density_runtime_code,
    interior,
    submesh,
):
    absorber = {
        "HI": (
            getattr(par, "hydrogen_mass_fraction", 1.0)
            * np.asarray(density_runtime_code[interior], dtype=float)
            * scales["density_cgs_g_cm3"]
            / PROTON_MASS_CGS
            * np.clip(np.asarray(fluid.xHI[interior], dtype=float), 0.0, 1.0)
        ),
    }
    group_edges_eV = getattr(par, "radiation_group_edges_eV", None)
    if group_edges_eV is not None:
        result = _trace_multigroup(
            par,
            mesh,
            code_units,
            submesh,
            absorber,
            group_edges_eV,
        )
    else:
        result = _trace_single_group(par, mesh, code_units, submesh, absorber)
    return _store_radiative_transfer_result(
        result,
        fluid,
        density_runtime_code,
        interior,
        scales,
    )


def _trace_multigroup(par, mesh, code_units, submesh, absorber, group_edges_eV):
    sigma_groups = getattr(par, "radiation_group_sigma_gamma", None)
    if sigma_groups is None:
        sigma_groups = getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA_CGS_CM2)
    boundary_groups = getattr(
        par,
        "radiative_transfer_boundary_flux_groups",
        getattr(par, "radiative_transfer_boundary_flux", 0.0),
    )
    source_groups = getattr(
        par,
        "source_photon_rate_groups",
        getattr(par, "source_photon_rate", 0.0),
    )
    sigma_groups = _radiative_value(sigma_groups, code_units, CGS_AREA_UNIT, "area_cgs_cm2")
    boundary_groups = _radiative_value(
        boundary_groups,
        code_units,
        CGS_PHOTON_FLUX_UNIT,
        "photon_flux_per_cgs_cm2_s",
    )
    source_groups = _radiative_value(
        source_groups,
        code_units,
        CGS_RATE_UNIT,
        "photon_rate_per_s",
    )
    return rrt.trace_long_characteristics(
        submesh,
        absorber_densities=absorber,
        cross_sections_cgs_cm2={"HI": sigma_groups},
        boundary_flux=boundary_groups,
        source_photon_rate=source_groups,
        direction=rrt.parameter_value(par, "radiative_transfer_direction", 1),
        coordsys=getattr(mesh, "coordsys", "cartesian"),
        group_edges_eV=group_edges_eV,
    )


def _trace_single_group(par, mesh, code_units, submesh, absorber):
    sigma_value = getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA_CGS_CM2)
    boundary_value = rrt.parameter_value(par, "radiative_transfer_boundary_flux", 0.0)
    source_value = rrt.parameter_value(par, "source_photon_rate", 0.0)
    sigma_value = _radiative_value(sigma_value, code_units, CGS_AREA_UNIT, "area_cgs_cm2")
    boundary_value = _radiative_value(
        boundary_value,
        code_units,
        CGS_PHOTON_FLUX_UNIT,
        "photon_flux_per_cgs_cm2_s",
    )
    source_value = _radiative_value(source_value, code_units, CGS_RATE_UNIT, "photon_rate_per_s")
    return rrt.trace_long_characteristics(
        submesh,
        absorber_densities=absorber,
        cross_sections_cgs_cm2={"HI": sigma_value},
        boundary_flux=boundary_value,
        source_photon_rate=source_value,
        direction=rrt.parameter_value(par, "radiative_transfer_direction", 1),
        coordsys=getattr(mesh, "coordsys", "cartesian"),
    )


def _radiative_value(value, code_units, target_unit, code_unit_name):
    if hasattr(value, "to_value"):
        return _as_cgs_float(value, target_unit)
    return code_quantity_to_cgs(value, code_units, code_unit_name)


class Solver:
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

    def safe_divide(self, numerator, denominator):
        """Safely divide solver arrays for extracted solver components."""
        return self._safe_divide(numerator, denominator)

    def _geometry_state(self, mesh, par):
        """Return the representation-selected typed mesh geometry."""
        if par is None:
            par = getattr(mesh, "par", getattr(mesh, "_par", None))
        geometry = getattr(mesh, "geometry_state", None)
        if geometry is None:
            raise ValueError("solver requires typed mesh geometry state")
        coordinate, boundary, width, area, volume = select_mesh_geometry_arrays(
            geometry,
            par,
        )
        return SimpleNamespace(
            coordinate_runtime_code=coordinate,
            boundary_runtime_code=boundary,
            width_runtime_code=width,
            area_runtime_code=area,
            volume_runtime_code=volume,
        )

    def geometry_state(self, mesh, par):
        """Return the selected mesh geometry for solver components."""
        return self._geometry_state(mesh, par)

    def _fluid_primitive_state(self, fluid, par):
        """Return the typed representation-selected primitive arrays."""
        runtime_state = getattr(fluid, "runtime_state", None)
        if runtime_state is None:
            raise ValueError("solver requires typed fluid runtime state")
        return runtime_state

    def _active_primitive_arrays(self, fluid, par):
        """Return the active representation's primitive runtime arrays.

        The Euler kernels are representation-neutral.  This is the single
        solver boundary where their density, velocity, pressure, and
        temperature inputs are selected from the typed runtime state.
        """
        primitive = self._fluid_primitive_state(fluid, par)
        return select_fluid_primitive_arrays(primitive, par)[:4]

    def active_primitive_arrays(self, fluid, par):
        """Return the active primitive arrays for solver components."""
        return self._active_primitive_arrays(fluid, par)

    def _interior_slice(self, par):
        first = int(par.mesh.ghost_cells)
        return slice(
            first,
            first + int(par.mesh.grid_cells),
        )

    def interior_slice(self, par):
        """Return the active-cell slice for solver components."""
        return self._interior_slice(par)

    def _thermochemistry_enabled(self, fluid, par):
        return rtc.thermochemistry_enabled(fluid, par)

    def _thermochemistry_radiation_enabled(self, fluid, par):
        return (
            self._thermochemistry_enabled(fluid, par)
            and (
                getattr(par, "hydrogen_radiation_field", False)
                or getattr(par, "radiative_transfer", False)
            )
            and hasattr(fluid, "ngamma_code")
        )

    def ApplyRadiativeTransfer(self, mesh, fluid, par):  # noqa: N802
        """Refresh photon density from the shared radiative-transfer solver."""
        if not getattr(par, "radiative_transfer", False):
            return None
        code_units = _code_units(par)
        scales = code_unit_scales(code_units)
        density_runtime_code = self._active_primitive_arrays(fluid, par)[0]
        if not hasattr(fluid, "ngamma_code"):
            fluid.ngamma_code = np.zeros(
                np.shape(density_runtime_code),
                dtype=float,
            )
        interior = self._interior_slice(par)
        geometry = getattr(mesh, "geometry_state", None)
        if geometry is None:
            raise ValueError("radiative transfer requires typed mesh geometry state")
        geometry_state = self._geometry_state(mesh, par)
        boundary_field = geometry_state.boundary_runtime_code
        volume_field = geometry_state.volume_runtime_code
        area_field = geometry_state.area_runtime_code
        boundary = np.asarray(
            boundary_field[interior.start : interior.stop + 1],
            dtype=float,
        )
        volume = np.asarray(volume_field[interior], dtype=float)
        submesh = SimpleNamespace(
            coordsys=getattr(mesh, "coordsys", "cartesian"),
            boundary_cgs_cm=boundary * scales["length_cgs_cm"],
            width_cgs_cm=np.diff(boundary) * scales["length_cgs_cm"],
            volume_cgs_cm3=volume * scales["volume_cgs_cm3"],
        )
        if area_field is not None:
            submesh.face_area_cgs_cm2 = (
                np.asarray(area_field[interior], dtype=float) * scales["area_cgs_cm2"]
            )
        return _trace_radiative_transfer(
            par,
            mesh,
            fluid,
            code_units,
            scales,
            density_runtime_code,
            interior,
            submesh,
        )

    def _spherical_origin_face_index(self, mesh):
        if getattr(mesh, "coordsys", None) != "spherical":
            return None
        boundary = np.asarray(
            self._geometry_state(
                mesh,
                getattr(mesh, "par", getattr(mesh, "_par", None)),
            ).boundary_runtime_code,
            dtype=float,
        )
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
        if hasattr(fluid, "AngularMomentum_code") and hasattr(fluid.AngularMomentum_code, "flux"):
            fluid.AngularMomentum_code.flux[origin_face] = 0.0
        if hasattr(fluid, "rotational_energy_flux"):
            fluid.rotational_energy_flux[origin_face] = 0.0

    def zero_spherical_origin_flux(self, mesh, fluid):
        """Zero the flux through a spherical origin face."""
        return self._zero_spherical_origin_flux(mesh, fluid)

    @staticmethod
    def _hydrostatic_core_enabled(par):
        from .hydrostatic import hydrostatic_core_enabled  # noqa: PLC0415

        return hydrostatic_core_enabled(par)

    def InitializeHydrostaticCore(self, mesh, fluid, par):  # noqa: N802
        from .hydrostatic import initialize_hydrostatic_core  # noqa: PLC0415

        return initialize_hydrostatic_core(self, mesh, fluid, par)

    def ApplyHydrostaticCore(self, mesh, fluid, par):  # noqa: N802
        from .hydrostatic import apply_hydrostatic_core  # noqa: PLC0415

        return apply_hydrostatic_core(self, mesh, fluid, par)

    def _apply_hydrostatic_core_flux(self, fluid, par):
        from .hydrostatic import apply_hydrostatic_core_flux  # noqa: PLC0415

        return apply_hydrostatic_core_flux(self, fluid, par)

    def apply_hydrostatic_core_flux(self, fluid, par):
        """Apply hydrostatic-core flux corrections."""
        return self._apply_hydrostatic_core_flux(fluid, par)

    def _boundary_field_names(self, *args, **kwargs):
        from .boundary_conditions import _boundary_field_names  # noqa: PLC0415

        return _boundary_field_names(self, *args, **kwargs)

    def boundary_field_names(self, *args, **kwargs):
        """Return fields participating in boundary updates."""
        return self._boundary_field_names(*args, **kwargs)

    def _copy_boundary_state(self, *args, **kwargs):
        from .boundary_conditions import _copy_boundary_state  # noqa: PLC0415

        return _copy_boundary_state(self, *args, **kwargs)

    def copy_boundary_state(self, *args, **kwargs):
        """Copy a prepared boundary state into ghost cells."""
        return self._copy_boundary_state(*args, **kwargs)

    def _boundary_state(self, *args, **kwargs):
        from .boundary_conditions import _boundary_state  # noqa: PLC0415

        return _boundary_state(self, *args, **kwargs)

    def boundary_state(self, *args, **kwargs):
        """Build a boundary state from a fluid slice."""
        return self._boundary_state(*args, **kwargs)

    def _to_code_number_density(self, *args, **kwargs):
        from .boundary_conditions import _to_code_number_density  # noqa: PLC0415

        return _to_code_number_density(self, *args, **kwargs)

    def to_code_number_density(self, *args, **kwargs):
        """Convert an injected photon number density to code units."""
        return self._to_code_number_density(*args, **kwargs)

    def _apply_periodic_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_periodic_boundary  # noqa: PLC0415

        return _apply_periodic_boundary(self, *args, **kwargs)

    def apply_periodic_boundary(self, *args, **kwargs):
        """Apply periodic ghost-cell values."""
        return self._apply_periodic_boundary(*args, **kwargs)

    def _apply_open_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_open_boundary  # noqa: PLC0415

        return _apply_open_boundary(self, *args, **kwargs)

    def apply_open_boundary(self, *args, **kwargs):
        """Apply open ghost-cell values."""
        return self._apply_open_boundary(*args, **kwargs)

    def _apply_reflecting_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_reflecting_boundary  # noqa: PLC0415

        return _apply_reflecting_boundary(self, *args, **kwargs)

    def apply_reflecting_boundary(self, *args, **kwargs):
        """Apply reflecting ghost-cell values."""
        return self._apply_reflecting_boundary(*args, **kwargs)

    def _apply_spherical_inner_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_spherical_inner_boundary  # noqa: PLC0415

        return _apply_spherical_inner_boundary(self, *args, **kwargs)

    def apply_spherical_inner_boundary(self, *args, **kwargs):
        """Apply the inner spherical ghost-cell boundary."""
        return self._apply_spherical_inner_boundary(*args, **kwargs)

    def _apply_open_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_open_spherical_boundary  # noqa: PLC0415

        return _apply_open_spherical_boundary(self, *args, **kwargs)

    def apply_open_spherical_boundary(self, *args, **kwargs):
        """Apply an open spherical boundary."""
        return self._apply_open_spherical_boundary(*args, **kwargs)

    def _apply_inflow_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_inflow_spherical_boundary  # noqa: PLC0415

        return _apply_inflow_spherical_boundary(self, *args, **kwargs)

    def apply_inflow_spherical_boundary(self, *args, **kwargs):
        """Apply an inflow spherical boundary."""
        return self._apply_inflow_spherical_boundary(*args, **kwargs)

    def _apply_outflow_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_outflow_spherical_boundary  # noqa: PLC0415

        return _apply_outflow_spherical_boundary(self, *args, **kwargs)

    def apply_outflow_spherical_boundary(self, *args, **kwargs):
        """Apply an outflow spherical boundary."""
        return self._apply_outflow_spherical_boundary(*args, **kwargs)

    def _apply_wind_spherical_boundary(self, *args, **kwargs):
        from .boundary_conditions import _apply_wind_spherical_boundary  # noqa: PLC0415

        return _apply_wind_spherical_boundary(self, *args, **kwargs)

    def apply_wind_spherical_boundary(self, *args, **kwargs):
        """Apply a wind spherical boundary."""
        return self._apply_wind_spherical_boundary(*args, **kwargs)

    def SetPrimitive(self, mesh, fluid, par=None, verbose=None):  # noqa: N802
        """Update primitive variables from conserved quantities."""
        if verbose is None:
            verbose = 0
        par = par or getattr(mesh, "par", getattr(mesh, "_par", None))
        self._validate_dual_energy_compatibility(
            fluid,
            par or getattr(mesh, "par", getattr(mesh, "_par", None)),
        )
        (
            rho_runtime_code,
            vel_runtime_code,
            pre_runtime_code,
            temp_runtime_code,
        ) = self._active_primitive_arrays(fluid, par)
        vol = self._geometry_state(mesh, par).volume_runtime_code
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
        rho_runtime_code[...] = as_named_array(rho)
        vel_runtime_code[...] = as_named_array(vel)
        if hasattr(fluid, "AngularMomentum_code"):
            specific_angular_momentum = np.zeros_like(rho)
            np.divide(
                np.asarray(fluid.AngularMomentum_code, dtype=float),
                mass,
                out=specific_angular_momentum,
                where=valid_mass,
            )
            fluid.specific_angular_momentum_code = as_named_array(
                specific_angular_momentum,
            )
        rotational_energy_density = self._rotational_energy_density(
            mesh,
            fluid,
            par,
        )
        density_floor = self._cfl_density_floor(par)
        numerical_vacuum = active & (rho <= density_floor)
        vel_runtime_code[numerical_vacuum] = 0.0
        # Conserved Energy contains rotational kinetic energy when the opt-in
        # model is enabled; pressure sees only thermal plus radial kinetic
        # energy at this stage.
        energy_density = as_named_array(
            energy_density - rotational_energy_density,
        )
        pressure_args = (rho_runtime_code, vel_runtime_code, energy_density)
        if getattr(fluid.eos, "is_isothermal", False):
            total_pressure = fluid.eos.pressure_from_conserved(
                *pressure_args,
                temp=temp_runtime_code,
                mu=getattr(fluid, "mu", None),
            )
        else:
            total_pressure = fluid.eos.pressure_from_conserved(*pressure_args)
        pre_runtime_code[...] = total_pressure
        if self._dual_energy_enabled(par) and hasattr(fluid, "InternalEnergy_code"):
            internal_density = np.zeros_like(rho)
            internal_density[valid_volume] = (
                np.asarray(fluid.InternalEnergy_code, dtype=float)[valid_volume]
                / np.asarray(vol, dtype=float)[valid_volume]
            )
            dual_pressure = (fluid.eos.gamma - 1.0) * internal_density
            total_thermal = energy_density - 0.5 * rho_runtime_code * vel_runtime_code**2
            eta1 = self._dual_energy_eta(par, "dual_energy_eta1")
            total_valid = (
                active
                & ~numerical_vacuum
                & np.isfinite(total_thermal)
                & (total_thermal > 0.0)
                & np.isfinite(total_pressure)
                & (total_pressure > 0.0)
            )
            dual_valid = (
                active
                & ~numerical_vacuum
                & np.isfinite(internal_density)
                & (internal_density > 0.0)
                & np.isfinite(dual_pressure)
                & (dual_pressure > 0.0)
            )
            thermal_fraction = np.divide(
                total_thermal,
                np.maximum(np.abs(energy_density), 1.0e-300),
                out=np.zeros_like(total_thermal),
                where=np.isfinite(total_thermal),
            )
            consistency_factor = max(
                0.0,
                float(
                    np.asarray(getattr(par, "dual_energy_consistency_factor", 1.0e-1), dtype=float),
                ),
            )
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
            use_total = (
                total_valid
                & ~dual_preferred
                & ((thermal_fraction > eta1) | ~dual_valid | (dual_to_total < consistency_factor))
            )
            use_dual = dual_valid & ~use_total
            pressure_selection = str(
                getattr(
                    par,
                    "dual_energy_pressure_selection",
                    "switch",
                ),
            ).lower()
            if pressure_selection in ("conservative", "e-k", "ek"):
                use_total = total_valid
                use_dual = np.zeros_like(use_total, dtype=bool)
            elif pressure_selection in (
                "internal",
                "internal-energy",
                "dual",
                "dual-energy",
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
            pre_runtime_code[use_dual] = dual_pressure[use_dual]
            pre_runtime_code[use_total] = total_pressure[use_total]

            # If the separately advected field has failed but E-K is still a
            # valid conservative estimate, use E-K and count the fallback.
            fallback = active & ~numerical_vacuum & ~dual_valid & total_valid
            self.dual_energy_pressure_fallback_count += int(
                np.count_nonzero(fallback),
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
            both_invalid = active & ~numerical_vacuum & ~total_valid & ~dual_valid
            if np.any(both_invalid):
                floor_pressure_value = max(
                    0.0,
                    float(
                        np.asarray(
                            getattr(par, "dual_energy_pressure_floor", 1.0e-20),
                            dtype=float,
                        ),
                    ),
                )
                if floor_pressure_value <= 0.0:
                    floor_pressure_value = 1.0e-20
                floor_pressure = np.full_like(rho, floor_pressure_value)
                floor_internal_density = floor_pressure / (fluid.eos.gamma - 1.0)
                current_internal_density = np.maximum(total_thermal, 0.0)
                injected_density = np.maximum(
                    floor_internal_density - current_internal_density,
                    0.0,
                )
                injected_energy = injected_density * np.asarray(vol, dtype=float)
                fluid.Energy_code[both_invalid] += injected_energy[both_invalid]
                pre_runtime_code[both_invalid] = floor_pressure[both_invalid]
                internal_density[both_invalid] = floor_internal_density[both_invalid]
                fluid.InternalEnergy_code[both_invalid] = injected_energy[both_invalid] + (
                    current_internal_density[both_invalid]
                    * np.asarray(vol, dtype=float)[both_invalid]
                )
                self.dual_energy_floor_count += int(
                    np.count_nonzero(both_invalid),
                )
                self.dual_energy_floor_injected_energy += float(
                    np.sum(injected_energy[both_invalid]),
                )
                selection_code[both_invalid] = 2
            # Keep the conservative fallback pressure for cells where the
            # dual field is invalid but E-K is admissible.
            pre_runtime_code[fallback] = total_pressure[fallback]
        rho_runtime_code[~active] = 0.0
        vel_runtime_code[~active] = 0.0
        self._apply_primitive_pressure_floor(
            pre_runtime_code,
            rho_runtime_code,
            numerical_vacuum,
            fluid,
            par,
        )
        if verbose >= 2:  # noqa: PLR2004
            log_diagnostic(
                logging.DEBUG,
                "primitive_state_reconstructed",
                rho_runtime_code=rho_runtime_code,
                velocity_runtime_code=vel_runtime_code,
                pressure_runtime_code=pre_runtime_code,
            )

    @staticmethod
    def _apply_primitive_pressure_floor(
        pressure_runtime_code,
        density_runtime_code,
        numerical_vacuum,
        fluid,
        par,
    ):
        invalid_pressure = np.logical_or(
            pressure_runtime_code <= 0.0,
            np.isnan(pressure_runtime_code),
        )
        temperature_floor = getattr(par, "hydro_temperature_floor", None)
        if temperature_floor is not None and float(temperature_floor) > 0.0:
            floor_pressure = np.asarray(
                fluid.eos.pressure(
                    density_runtime_code,
                    float(temperature_floor),
                    fluid.mu,
                ),
                dtype=float,
            )
            below_floor = ~numerical_vacuum & np.logical_or(
                invalid_pressure,
                pressure_runtime_code < floor_pressure,
            )
            pressure_runtime_code[below_floor] = floor_pressure[below_floor]
        else:
            pressure_runtime_code[invalid_pressure & ~numerical_vacuum] = 0.0
        pressure_runtime_code[numerical_vacuum] = 0.0

    def SetConserved(self, mesh, fluid, verbose=None):  # noqa: N802
        """Update conserved mass, momentum, and energy from primitive variables."""
        if verbose is None:
            verbose = 0
        par = getattr(mesh, "par", getattr(mesh, "_par", None))
        self._validate_dual_energy_compatibility(fluid, par)
        (
            rho_runtime_code,
            vel_runtime_code,
            pre_runtime_code,
            _,
        ) = self._active_primitive_arrays(fluid, par)
        density_floor = self._cfl_density_floor(par)
        dual_energy = self._dual_energy_enabled(par)
        (
            old_internal,
            old_total_energy,
            old_total_mass,
            old_total_momentum,
            old_angular_momentum,
            old_potential_energy,
            old_conserved,
        ) = self._capture_conserved_state(
            fluid,
            rho_runtime_code,
            density_floor,
            dual_energy,
        )
        vol = self._geometry_state(
            mesh,
            getattr(mesh, "par", getattr(mesh, "_par", None)),
        ).volume_runtime_code
        fluid.Mass_code = as_named_array(rho_runtime_code * vol)
        fluid.Mom_code = as_named_array(rho_runtime_code * vel_runtime_code * vol)
        if hasattr(fluid, "specific_angular_momentum_code") or old_angular_momentum is not None:
            specific_angular_momentum = np.asarray(
                getattr(fluid, "specific_angular_momentum_code", np.zeros_like(rho_runtime_code)),
                dtype=float,
            )
            fluid.AngularMomentum_code = as_named_array(
                rho_runtime_code * specific_angular_momentum * vol,
            )
        rotational_energy_density = self._rotational_energy_density(
            mesh,
            fluid,
            par,
        )
        fluid.Energy_code = as_named_array(
            (
                fluid.eos.total_energy_density(rho_runtime_code, vel_runtime_code, pre_runtime_code)
                + rotational_energy_density
            )
            * vol,
        )
        potential = self._gravity_potential(mesh, par)
        if potential is not None:
            fluid.GravitationalPotentialEnergy_code = as_named_array(
                fluid.Mass_code * potential
                if old_potential_energy is None
                else old_potential_energy,
            )
            fluid.gravity_potential_energy_initialized = True
        fluid.Mass_code[np.logical_or(fluid.Mass_code < 0.0, np.isnan(fluid.Mass_code))] = 0.0
        fluid.Energy_code[np.logical_or(fluid.Energy_code < 0.0, np.isnan(fluid.Energy_code))] = 0.0
        self._restore_conserved_totals(
            fluid,
            par,
            old_total_energy,
            old_total_mass,
            old_total_momentum,
            old_angular_momentum,
        )
        if dual_energy and getattr(fluid.eos, "is_polytropic", False):
            internal = np.asarray(
                fluid.eos.thermal_energy_density(pre_runtime_code) * vol,
                dtype=float,
            )
            if old_internal is not None:
                first = int(par.mesh.ghost_cells)
                count = int(par.mesh.grid_cells)
                preserved_internal = (
                    old_internal
                    if old_internal.size == count
                    else old_internal[first : first + count]
                )
                internal[first : first + count] = preserved_internal
            fluid.InternalEnergy_code = as_named_array(np.maximum(internal, 0.0))
        if old_conserved is not None:
            inactive, old_mass, old_mom, old_energy = old_conserved
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            if old_mass.size == count:
                inactive_active = inactive[first : first + count]
                fluid.Mass_code[first : first + count][inactive_active] = old_mass[inactive_active]
                fluid.Mom_code[first : first + count][inactive_active] = old_mom[inactive_active]
                fluid.Energy_code[first : first + count][inactive_active] = old_energy[
                    inactive_active
                ]
            else:
                fluid.Mass_code[inactive] = old_mass[inactive]
                fluid.Mom_code[inactive] = old_mom[inactive]
                fluid.Energy_code[inactive] = old_energy[inactive]
        self._synchronize_dual_internal_energy(
            mesh,
            fluid,
            par,
            old_internal,
            dual_energy,
        )
        if verbose >= 2:  # noqa: PLR2004
            log_diagnostic(
                logging.DEBUG,
                "conserved_state_synchronized",
                mass_code=fluid.Mass_code,
                momentum_code=fluid.Mom_code,
                energy_code=fluid.Energy_code,
            )
        if hasattr(fluid, "refresh_runtime_state"):
            fluid.refresh_runtime_state()

    @staticmethod
    def _restore_conserved_totals(
        fluid,
        par,
        old_total_energy,
        old_total_mass,
        old_total_momentum,
        old_angular_momentum,
    ):
        if (
            old_total_energy is None
            and old_total_mass is None
            and old_total_momentum is None
            and old_angular_momentum is None
        ):
            return
        first = int(par.mesh.ghost_cells)
        count = int(par.mesh.grid_cells)

        def active_values(values):
            return values if values.size == count else values[first : first + count]

        if old_total_energy is not None:
            fluid.Energy_code[first : first + count] = active_values(old_total_energy)
        if old_total_mass is not None and old_total_momentum is not None:
            fluid.Mass_code[first : first + count] = active_values(old_total_mass)
            fluid.Mom_code[first : first + count] = active_values(old_total_momentum)
        if old_angular_momentum is not None:
            fluid.AngularMomentum_code[first : first + count] = active_values(
                old_angular_momentum,
            )

    def _synchronize_dual_internal_energy(
        self,
        mesh,
        fluid,
        par,
        old_internal,
        dual_energy,
    ):
        if not (
            dual_energy and old_internal is not None and getattr(fluid.eos, "is_polytropic", False)
        ):
            return
        eta2 = self._dual_energy_eta(par, "dual_energy_eta2")
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
            conserved_energy
            - conserved_kinetic
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
        physical[first : first + count] = True
        sync = (
            physical & np.isfinite(total_thermal) & (total_thermal > 0.0) & (total_fraction > eta2)
        )
        fluid.InternalEnergy_code[sync] = total_thermal[sync]
        self.dual_energy_synchronization_count += int(np.count_nonzero(sync))

    @staticmethod
    def _capture_conserved_state(fluid, rho_runtime_code, density_floor, dual_energy):
        old_internal = (
            np.asarray(fluid.InternalEnergy_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, "InternalEnergy_code")
            else None
        )
        old_total_energy = (
            np.asarray(fluid.Energy_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, "Energy_code")
            else None
        )
        old_total_mass = (
            np.asarray(fluid.Mass_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, "Mass_code")
            else None
        )
        old_total_momentum = (
            np.asarray(fluid.Mom_code, dtype=float).copy()
            if dual_energy and hasattr(fluid, "Mom_code")
            else None
        )
        old_angular_momentum = (
            np.asarray(fluid.AngularMomentum_code, dtype=float).copy()
            if hasattr(fluid, "AngularMomentum_code")
            else None
        )
        old_potential_energy = None
        if hasattr(fluid, "GravitationalPotentialEnergy_code") and (
            getattr(fluid, "gravity_potential_energy_initialized", False)
            or np.any(np.asarray(fluid.GravitationalPotentialEnergy_code, dtype=float) != 0.0)
        ):
            old_potential_energy = np.asarray(
                fluid.GravitationalPotentialEnergy_code,
                dtype=float,
            ).copy()
        old_conserved = None
        if density_floor > 0.0 and all(
            hasattr(fluid, name) for name in ("Mass_code", "Mom_code", "Energy_code")
        ):
            inactive = np.isfinite(rho_runtime_code) & (rho_runtime_code <= density_floor)
            old_conserved = (
                inactive,
                np.asarray(fluid.Mass_code, dtype=float).copy(),
                np.asarray(fluid.Mom_code, dtype=float).copy(),
                np.asarray(fluid.Energy_code, dtype=float).copy(),
            )
        return (
            old_internal,
            old_total_energy,
            old_total_mass,
            old_total_momentum,
            old_angular_momentum,
            old_potential_energy,
            old_conserved,
        )

    def SetGradient(self, mesh, fluid):  # noqa: N802
        """Calculate centered gradients for density, velocity, and pressure."""
        par = getattr(mesh, "par", getattr(mesh, "_par", None))
        width_runtime_code = self._geometry_state(mesh, par).width_runtime_code
        self._fluid_primitive_state(fluid, par)
        periodic = (
            par is not None and hasattr(par, "boundary") and par.boundary.condition == "Periodic"
        )
        density_code, velocity_code, pressure_code, _ = self._active_primitive_arrays(fluid, par)
        if periodic:
            first = int(par.mesh.ghost_cells)
            count = int(par.mesh.grid_cells)
            for quantity in (density_code, velocity_code, pressure_code):
                quantity.grad = self._periodic_physical_gradient(
                    quantity,
                    width_runtime_code,
                    first,
                    count,
                )
        else:
            density_code.grad = ru.CalGradient(density_code, width_runtime_code)
            velocity_code.grad = ru.CalGradient(velocity_code, width_runtime_code)
            pressure_code.grad = ru.CalGradient(pressure_code, width_runtime_code)
        if hasattr(fluid, "specific_angular_momentum_code"):
            if periodic:
                fluid.specific_angular_momentum_code.grad = self._periodic_physical_gradient(
                    fluid.specific_angular_momentum_code,
                    width_runtime_code,
                    first,
                    count,
                )
            else:
                fluid.specific_angular_momentum_code.grad = ru.CalGradient(
                    fluid.specific_angular_momentum_code,
                    width_runtime_code,
                )

    @staticmethod
    def _periodic_physical_gradient(quantity, width_runtime_code, first, count):
        """Build periodic gradients from physical cells and mirror ghosts."""
        values = np.asarray(quantity, dtype=float)
        last = first + count
        gradient = np.zeros_like(values)
        physical_values = values[first:last]
        physical_width_runtime_code = np.asarray(
            width_runtime_code[first:last],
            dtype=float,
        )
        gradient[first:last] = (
            ru.periodic_roll(physical_values, -1) - ru.periodic_roll(physical_values, 1)
        ) / (2.0 * physical_width_runtime_code)
        if first:
            gradient[:first] = gradient[last - first : last]
        if last < len(gradient):
            gradient[last:] = gradient[first : first + len(gradient) - last]
        return as_named_array(gradient)

    @staticmethod
    def _positivity_limited_internal_flux(old_internal, flux, area, dt, physical):
        from .positivity import limit_internal_flux  # noqa: PLC0415

        return limit_internal_flux(old_internal, flux, area, dt, physical)

    @staticmethod
    @staticmethod
    def _cfl_density_floor(*args, **kwargs):
        from .dual_energy import _cfl_density_floor  # noqa: PLC0415

        return _cfl_density_floor(*args, **kwargs)

    @staticmethod
    def cfl_density_floor(*args, **kwargs):
        """Return the configured density floor for solver components."""
        return Solver._cfl_density_floor(*args, **kwargs)

    @staticmethod
    @staticmethod
    def _dual_energy_enabled(*args, **kwargs):
        from .dual_energy import _dual_energy_enabled  # noqa: PLC0415

        return _dual_energy_enabled(*args, **kwargs)

    @staticmethod
    def dual_energy_enabled(*args, **kwargs):
        """Return whether dual-energy evolution is enabled."""
        return Solver._dual_energy_enabled(*args, **kwargs)

    def _validate_dual_energy_compatibility(self, fluid, par):
        if (
            par is not None
            and self._dual_energy_enabled(par)
            and getattr(getattr(fluid, "eos", None), "is_isothermal", False)
        ):
            raise ValueError(
                "dual energy is not supported with an isothermal EOS; "
                "disable dual_energy or use a polytropic EOS",
            )

    @staticmethod
    @staticmethod
    def _rotational_energy_enabled(*args, **kwargs):
        from .dual_energy import _rotational_energy_enabled  # noqa: PLC0415

        return _rotational_energy_enabled(*args, **kwargs)

    @staticmethod
    def rotational_energy_enabled(*args, **kwargs):
        """Return whether rotational energy is enabled."""
        return Solver._rotational_energy_enabled(*args, **kwargs)

    @staticmethod
    @staticmethod
    def _gravity_potential_energy_enabled(*args, **kwargs):
        from .dual_energy import _gravity_potential_energy_enabled  # noqa: PLC0415

        return _gravity_potential_energy_enabled(*args, **kwargs)

    @staticmethod
    def gravity_potential_energy_enabled(*args, **kwargs):
        """Return whether gravitational potential energy is enabled."""
        return Solver._gravity_potential_energy_enabled(*args, **kwargs)

    def _gravity_potential(self, *args, **kwargs):
        from .dual_energy import _gravity_potential  # noqa: PLC0415

        return _gravity_potential(self, *args, **kwargs)

    def gravity_potential(self, *args, **kwargs):
        """Return cell-centered gravitational potential values."""
        return self._gravity_potential(*args, **kwargs)

    def _gravity_potential_faces(self, *args, **kwargs):
        from .dual_energy import _gravity_potential_faces  # noqa: PLC0415

        return _gravity_potential_faces(self, *args, **kwargs)

    def gravity_potential_faces(self, *args, **kwargs):
        """Return face-centered gravitational potential values."""
        return self._gravity_potential_faces(*args, **kwargs)

    def _rotational_energy_density(self, *args, **kwargs):
        from .dual_energy import _rotational_energy_density  # noqa: PLC0415

        return _rotational_energy_density(self, *args, **kwargs)

    def rotational_energy_density(self, *args, **kwargs):
        """Return the rotational kinetic-energy density."""
        return self._rotational_energy_density(*args, **kwargs)

    def _rotational_energy_from_conserved(self, *args, **kwargs):
        from .dual_energy import _rotational_energy_from_conserved  # noqa: PLC0415

        return _rotational_energy_from_conserved(self, *args, **kwargs)

    def rotational_energy_from_conserved(self, *args, **kwargs):
        """Return rotational energy reconstructed from conserved state."""
        return self._rotational_energy_from_conserved(*args, **kwargs)

    @staticmethod
    @staticmethod
    def _dual_energy_eta(*args, **kwargs):
        from .dual_energy import _dual_energy_eta  # noqa: PLC0415

        return _dual_energy_eta(*args, **kwargs)

    @staticmethod
    def dual_energy_eta(*args, **kwargs):
        """Return a dual-energy limiter parameter."""
        return Solver._dual_energy_eta(*args, **kwargs)

    def _apply_low_density_face_mask(self, fluid, par, order):
        """Make below-floor reconstructed states vacuum-safe.

        This is a numerical mask only: cell-centred conserved mass and density
        remain unchanged.  It prevents a tiny positive density carrying a
        large pressure from determining the CFL step or Riemann flux.
        """
        density_floor = self._cfl_density_floor(par)
        if density_floor <= 0.0:
            return
        density_code, velocity_code, pressure_code, _ = self._active_primitive_arrays(fluid, par)
        for density, velocity, pressure in (
            (density_code.R, velocity_code.R, pressure_code.R),
            (density_code.L, velocity_code.L, pressure_code.L),
        ):
            inactive = ~np.isfinite(density) | (density <= density_floor)
            density[inactive] = 0.0
            velocity[inactive] = 0.0
            pressure[inactive] = 0.0
        if order == 1:
            for density, velocity, pressure in (
                (density_code.R.first, velocity_code.R.first, pressure_code.R.first),
                (density_code.L.first, velocity_code.L.first, pressure_code.L.first),
            ):
                inactive = ~np.isfinite(density) | (density <= density_floor)
                density[inactive] = 0.0
                velocity[inactive] = 0.0
                pressure[inactive] = 0.0

    def apply_low_density_face_mask(self, fluid, par, order):
        """Apply vacuum-safe values to reconstructed low-density faces."""
        return self._apply_low_density_face_mask(fluid, par, order)

    def SetConservedDensityFlux(self, fluid, par=None):  # noqa: N802
        """Store Euler fluxes and conserved densities on fluid arrays."""
        density_code, velocity_code, pressure_code, _ = self._active_primitive_arrays(fluid, par)
        (
            fluid.Mass_code.F,
            fluid.Mass_code.q,
            fluid.Mom_code.F,
            fluid.Mom_code.q,
            fluid.Energy_code.F,
            fluid.Energy_code.q,
        ) = fluid.eos.fluxes(density_code, velocity_code, pressure_code)

    @staticmethod
    @staticmethod
    def _set_angular_momentum_flux(*args, **kwargs):
        from .angular_momentum import _set_angular_momentum_flux  # noqa: PLC0415

        return _set_angular_momentum_flux(*args, **kwargs)

    def set_angular_momentum_flux(self, *args, **kwargs):
        """Construct the optional angular-momentum face flux."""
        return self._set_angular_momentum_flux(*args, **kwargs)

    def _limit_angular_momentum_flux(self, *args, **kwargs):
        from .angular_momentum import _limit_angular_momentum_flux  # noqa: PLC0415

        return _limit_angular_momentum_flux(self, *args, **kwargs)

    def _set_rotational_energy_flux(self, *args, **kwargs):
        from .angular_momentum import _set_rotational_energy_flux  # noqa: PLC0415

        return _set_rotational_energy_flux(self, *args, **kwargs)

    def set_rotational_energy_flux(self, *args, **kwargs):
        """Construct the optional rotational-energy face flux."""
        return self._set_rotational_energy_flux(*args, **kwargs)

    def _apply_local_angular_energy_fallback(self, *args, **kwargs):
        from .angular_momentum import _apply_local_angular_energy_fallback  # noqa: PLC0415

        return _apply_local_angular_energy_fallback(self, *args, **kwargs)

    def apply_local_angular_energy_fallback(self, *args, **kwargs):
        """Apply the local angular-energy admissibility fallback."""
        return self._apply_local_angular_energy_fallback(*args, **kwargs)

    def SetFaceLR(self, mesh, fluid, boundcond, order=0):  # noqa: N802
        """Construct left and right states at cell faces.

        ``order=0`` uses piecewise constant states. ``order=1`` applies a
        gradient reconstruction before limiting the fluxes.
        """
        from .fluxes import set_face_lr  # noqa: PLC0415

        return set_face_lr(self, mesh, fluid, order=order)

    def _apply_cosmological_background_boundary_face(self, mesh, fluid, order):
        """Replace the outer face states with the homogeneous EdS state."""
        par = getattr(mesh, "par", getattr(mesh, "_par", None))
        if par is None or not getattr(
            par,
            "cosmological_background_boundary_reconstruction",
            False,
        ):
            return
        if par.boundary.condition != "InflowSph":
            return
        density_code, velocity_code, pressure_code, _ = self._active_primitive_arrays(fluid, par)
        first = int(par.mesh.ghost_cells)
        outer_face = first + int(par.mesh.grid_cells)
        if outer_face >= len(density_code.R):
            return
        rho_background = float(
            par.boundary.rho_inflow_proper,
        )
        velocity_background = float(
            par.boundary.vel_inflow_proper,
        )
        pressure_background = float(
            fluid.eos.pressure(
                rho_background,
                float(
                    par.boundary.temperature_inflow_proper,
                ),
                float(par.boundary.inflow_mu),
            ),
        )
        for quantity, value in (
            (density_code, rho_background),
            (velocity_code, velocity_background),
            (pressure_code, pressure_background),
        ):
            quantity.R[outer_face] = value
            quantity.L[outer_face] = value
            if order == 1:
                quantity.R.first[outer_face] = value
                quantity.L.first[outer_face] = value
        if hasattr(fluid, "specific_angular_momentum_code"):
            angular_momentum = float(
                getattr(
                    par,
                    "specific_angular_momentum_inflow",
                    0.0,
                ),
            )
            fluid.specific_angular_momentum_code.R[outer_face] = angular_momentum
            fluid.specific_angular_momentum_code.L[outer_face] = angular_momentum
            if order == 1:
                fluid.specific_angular_momentum_code.R.first[outer_face] = angular_momentum
                fluid.specific_angular_momentum_code.L.first[outer_face] = angular_momentum

    def apply_cosmological_background_boundary_face(self, mesh, fluid, order):
        """Apply the cosmological background state to the outer face."""
        return self._apply_cosmological_background_boundary_face(mesh, fluid, order)

    @staticmethod
    def _vacuum_safe_primitive_state(rho, vel, pre):
        from .fluxes import vacuum_safe_primitive_state  # noqa: PLC0415

        return vacuum_safe_primitive_state(rho, vel, pre)

    @staticmethod
    def vacuum_safe_primitive_state(rho, vel, pre):
        """Return vacuum-safe primitive states for a Riemann solve."""
        return Solver._vacuum_safe_primitive_state(rho, vel, pre)

    @staticmethod
    def _hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma):  # noqa: N803
        from .fluxes import hllc_flux  # noqa: PLC0415

        return hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma)

    @staticmethod
    def hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma):  # noqa: N803
        """Compute the HLLC interface flux."""
        return Solver._hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma)

    def _interface_fluxes(self, fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method):  # noqa: N803
        from .fluxes import interface_fluxes  # noqa: PLC0415

        return interface_fluxes(
            fluid,
            rho_L,
            vel_L,
            pre_L,
            rho_R,
            vel_R,
            pre_R,
            method,
        )

    def SetFluxOnFace(self, fluid, boundcond, order=0, par=None, method="Rusanov"):  # noqa: N802
        """Calculate mass, momentum, and energy fluxes at interfaces."""
        from .fluxes import set_flux_on_face  # noqa: PLC0415

        return set_flux_on_face(self, fluid, par=par, order=order, method=method)

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
        density_runtime_code = self._active_primitive_arrays(fluid, par)[0]
        density = np.asarray(density_runtime_code, dtype=float)
        first = int(par.mesh.ghost_cells)
        count = int(par.mesh.grid_cells)
        last = min(first + count, len(density))
        inactive = ~np.isfinite(density) | (density <= density_floor)
        face_mask = np.zeros(len(density), dtype=bool)
        # Keep gas-vacuum interfaces active: their Riemann flux is what fills
        # the vacuum.  Only a vacuum-vacuum interface should be suppressed.
        # Face i joins cell i-1 (the rolled state) to cell i.
        face_mask[first:last] = inactive[first:last] & np.roll(inactive, 1)[first:last]
        if not np.any(face_mask):
            return
        for flux in (fluid.Mass_code.flux, fluid.Mom_code.flux, fluid.Energy_code.flux):
            flux[face_mask] = 0.0

    def apply_low_density_flux_mask(self, fluid, par):
        """Block flux through cells below the configured density floor."""
        return self._apply_low_density_flux_mask(fluid, par)

    @staticmethod
    def _positive_conserved_state(
        mass,
        momentum,
        energy,
        mass_floor=0.0,
        energy_floor=0.0,
        relative_tolerance=1.0e-12,
        angular_momentum=None,
        radius=None,
    ):
        from .positivity import positive_conserved_state  # noqa: PLC0415

        return positive_conserved_state(
            mass,
            momentum,
            energy,
            mass_floor=mass_floor,
            energy_floor=energy_floor,
            relative_tolerance=relative_tolerance,
            angular_momentum=angular_momentum,
            radius=radius,
        )

    def _apply_unlimited_face_fluxes(
        self,
        fluid,
        dt,
        mesh,
        par,
        mass_face,
        mom_face,
        energy_face,
        geometric_mom=None,
        angular_face=None,
    ):
        """Apply face corrections without the positivity limiter."""
        dt_value = float(np.asarray(dt, dtype=float))
        geometry = self._geometry_state(mesh, par)
        area = np.asarray(geometry.area_runtime_code, dtype=float)
        mass_update = np.asarray(mass_face, dtype=float) * area
        mom_update = np.asarray(mom_face, dtype=float) * area
        energy_update = np.asarray(energy_face, dtype=float) * area
        fluid.Mass_code += dt_value * (mass_update - ru.periodic_roll(mass_update, -1))
        fluid.Mom_code += dt_value * (
            mom_update
            - ru.periodic_roll(mom_update, -1)
            + (np.asarray(geometric_mom, dtype=float) if geometric_mom is not None else 0.0)
        )
        fluid.Energy_code += dt_value * (energy_update - ru.periodic_roll(energy_update, -1))
        if angular_face is not None and hasattr(fluid, "AngularMomentum_code"):
            angular_update = np.asarray(angular_face, dtype=float) * area
            fluid.AngularMomentum_code += dt_value * (
                angular_update - ru.periodic_roll(angular_update, -1)
            )
        self.last_face_limiter_factors = np.ones_like(np.asarray(mass_face, dtype=float))
        return 1.0

    def _positivity_context(self, fluid, mesh, par, dt):
        """Build arrays and predicates shared by face-flux recovery."""
        dt_value = float(np.asarray(dt, dtype=float))
        mass = np.asarray(fluid.Mass_code, dtype=float).copy()
        momentum = np.asarray(fluid.Mom_code, dtype=float).copy()
        energy = np.asarray(fluid.Energy_code, dtype=float).copy()
        angular = (
            np.asarray(fluid.AngularMomentum_code, dtype=float).copy()
            if hasattr(fluid, "AngularMomentum_code")
            else None
        )
        geometry = self._geometry_state(mesh, par)
        radius = (
            np.abs(np.asarray(geometry.coordinate_runtime_code, dtype=float))
            if angular is not None
            else None
        )
        count = len(mass)
        first = int(par.mesh.ghost_cells) if par is not None else 0
        last = min(first + int(par.mesh.grid_cells), count) if par is not None else count
        physical = np.zeros(count, dtype=bool)
        physical[first:last] = True
        volume = np.asarray(geometry.volume_runtime_code, dtype=float)
        mass_floor = (
            max(0.0, float(np.asarray(getattr(par, "positivity_density_floor", 0.0)))) * volume
        )
        energy_floor = (
            max(0.0, float(np.asarray(getattr(par, "positivity_energy_floor", 0.0)))) * volume
        )
        relative_tolerance = (
            1.0e-6
            if self._dual_energy_enabled(par) and hasattr(fluid, "InternalEnergy_code")
            else 1.0e-12
        )
        vacuum_mass = float(np.asarray(getattr(par, "cfl_density_floor", 0.0))) * volume
        vacuum = mass <= np.maximum(vacuum_mass, 0.0)
        mass[vacuum] = 0.0
        momentum[vacuum] = 0.0
        energy[vacuum] = 0.0
        fluid.Mass_code[vacuum] = 0.0
        fluid.Mom_code[vacuum] = 0.0
        fluid.Energy_code[vacuum] = 0.0
        if angular is not None:
            angular[vacuum] = 0.0

        def valid(mass_value, momentum_value, energy_value, angular_value=None):
            result = self._positive_conserved_state(
                mass_value,
                momentum_value,
                energy_value,
                mass_floor=mass_floor,
                energy_floor=energy_floor,
                relative_tolerance=relative_tolerance,
                angular_momentum=angular_value,
                radius=radius,
            )
            result[~physical] = True
            return result

        def cell_valid(index, mass_value, momentum_value, energy_value, angular_value=None):
            if not (
                math.isfinite(mass_value)
                and math.isfinite(momentum_value)
                and math.isfinite(energy_value)
                and mass_value >= mass_floor[index]
            ):
                return False
            if mass_value <= mass_floor[index]:
                return energy_value >= energy_floor[index]
            kinetic_value = 0.5 * momentum_value**2 / mass_value
            rotational_value = 0.0
            if angular_value is not None and radius is not None and radius[index] > 0.0:
                rotational_value = 0.5 * angular_value**2 / (mass_value * radius[index] ** 2)
            internal_value = energy_value - kinetic_value - rotational_value
            tolerance = relative_tolerance * max(
                abs(energy_value),
                kinetic_value,
                abs(energy_floor[index]),
                _FLOAT_TINY,
            )
            return internal_value >= energy_floor[index] - tolerance * (1.0 - 1.0e-8)

        return {
            "dt": dt_value,
            "mass": mass,
            "momentum": momentum,
            "energy": energy,
            "angular": angular,
            "geometry": geometry,
            "radius": radius,
            "physical": physical,
            "mass_floor": mass_floor,
            "energy_floor": energy_floor,
            "relative_tolerance": relative_tolerance,
            "valid": valid,
            "cell_valid": cell_valid,
        }

    def _limit_geometry_momentum(self, context, geometric_mom):
        """Limit the spherical pressure-geometry momentum source."""
        momentum = context["momentum"]
        geometry_increment = np.zeros_like(momentum)
        geometry_fraction = np.ones(len(momentum), dtype=float)
        if geometric_mom is None:
            return geometry_increment, geometry_fraction
        geometry_increment = context["dt"] * np.asarray(geometric_mom, dtype=float)
        mass = context["mass"]
        energy = context["energy"]
        angular = context["angular"]
        valid = context["valid"]
        cell_valid = context["cell_valid"]
        affected = (
            context["physical"]
            & valid(mass, momentum, energy, angular)
            & ~valid(
                mass,
                momentum + geometry_increment,
                energy,
                angular,
            )
        )
        for index in np.flatnonzero(affected):
            low, high = 0.0, 1.0
            for _ in range(48):
                middle = 0.5 * (low + high)
                trial = momentum[index] + middle * geometry_increment[index]
                if cell_valid(
                    index,
                    mass[index],
                    trial,
                    energy[index],
                    angular[index] if angular is not None else None,
                ):
                    low = middle
                else:
                    high = middle
                if high - low <= 1.0e-13:
                    break
            geometry_fraction[index] = low
        return geometry_increment, geometry_fraction

    def _recover_invariant_domain_faces(
        self,
        context,
        mass_face,
        delta_mass,
        delta_mom,
        delta_energy,
        delta_angular,
        geometry_increment,
        geometry_fraction,
    ):
        """Recover face fluxes with vectorized invariant-domain bisection."""
        mass = context["mass"]
        momentum = context["momentum"]
        energy = context["energy"]
        angular = context["angular"]
        valid = context["valid"]
        base_mom = momentum + geometry_fraction * geometry_increment
        base_angular = angular.copy() if angular is not None else None
        if not np.all(valid(mass, base_mom, energy, base_angular)):
            raise ValueError(
                "hydro state is outside positivity domain before invariant-domain face recovery",
            )

        def line_state(scale):
            face_mass = delta_mass * scale
            face_mom = delta_mom * scale
            face_energy = delta_energy * scale
            state_mass = mass + face_mass - ru.periodic_roll(face_mass, -1)
            state_mom = base_mom + face_mom - ru.periodic_roll(face_mom, -1)
            state_energy = energy + face_energy - ru.periodic_roll(face_energy, -1)
            state_angular = (
                base_angular + delta_angular * scale - ru.periodic_roll(delta_angular * scale, -1)
                if base_angular is not None
                else None
            )
            return state_mass, state_mom, state_energy, state_angular

        factors = np.ones(len(mass_face), dtype=float)
        for _ in range(64):
            state = line_state(factors)
            invalid = ~valid(*state)
            if not np.any(invalid):
                break
            invalid_faces = invalid | ru.periodic_roll(invalid, 1)
            updated_factors = factors.copy()
            updated_factors[invalid_faces] *= 0.5
            near_zero = invalid_faces & (updated_factors < 1.0e-12)  # noqa: PLR2004
            updated_factors[near_zero] = 0.0
            if np.array_equal(updated_factors, factors):
                raise ValueError(
                    "invariant-domain local face recovery did not converge",
                )
            factors = updated_factors
        else:
            raise ValueError(
                "invariant-domain local face recovery exceeded iteration limit",
            )
        return (*line_state(factors), factors)

    def _repair_wind_reservoir(self, fluid, par, context, mass, momentum, energy):
        """Restore a WindSph floor using the imposed wind state."""
        boundary = getattr(par, "boundary", None)
        if boundary is None or getattr(boundary, "condition", None) != "WindSph":
            return mass, momentum, energy
        mass_floor = context["mass_floor"]
        physical = context["physical"]
        wind_density = float(np.asarray(boundary.rho_outflow_proper))
        wind_velocity = float(np.asarray(boundary.vel_outflow_proper))
        wind_pressure = float(
            np.asarray(
                fluid.eos.pressure(
                    boundary.rho_outflow_proper,
                    boundary.temperature_outflow_proper,
                    boundary.outflow_mu,
                ),
            ),
        )
        wind_internal = (
            wind_pressure / wind_density / (fluid.eos.gamma - 1.0)
            if wind_density > 0.0 and not fluid.eos.is_isothermal
            else 0.0
        )
        wind_specific_energy = 0.5 * wind_velocity**2 + wind_internal
        missing_mass = np.maximum(mass_floor - mass, 0.0)
        reservoir_cells = physical & (missing_mass > 0.0)
        if np.any(reservoir_cells):
            momentum[reservoir_cells] += missing_mass[reservoir_cells] * wind_velocity
            energy[reservoir_cells] += missing_mass[reservoir_cells] * wind_specific_energy
            mass[reservoir_cells] += missing_mass[reservoir_cells]
            self._last_wind_reservoir_mass = float(
                np.sum(missing_mass[reservoir_cells]),
            )
        else:
            self._last_wind_reservoir_mass = 0.0

        kinetic = np.divide(
            0.5 * momentum**2,
            mass,
            out=np.zeros_like(mass),
            where=mass > 0.0,
        )
        floor_edge = physical & (mass <= mass_floor * (1.0 + 1.0e-12))
        energy_deficit = floor_edge & (energy < kinetic + context["energy_floor"])
        for index in np.flatnonzero(energy_deficit):
            base_mass = mass[index]
            base_momentum = momentum[index]
            base_energy = energy[index]
            required_energy = context["energy_floor"][index]

            def reservoir_valid(
                delta_mass,
                base_mass=base_mass,
                base_momentum=base_momentum,
                base_energy=base_energy,
                required_energy=required_energy,
            ):
                trial_mass = base_mass + delta_mass
                trial_momentum = base_momentum + delta_mass * wind_velocity
                trial_energy = base_energy + delta_mass * wind_specific_energy
                return trial_energy - (0.5 * trial_momentum**2 / trial_mass) >= required_energy

            low = 0.0
            high = max(base_mass, mass_floor[index], 1.0) * 1.0e-12
            for _ in range(96):
                if reservoir_valid(high):
                    break
                high *= 2.0
            if not reservoir_valid(high):
                raise ValueError(
                    "wind reservoir could not restore conservative "
                    "energy admissibility at cell %d" % index,
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
        return mass, momentum, energy

    def _analytical_scalar_validators(
        self,
        *,
        factors,
        total_mass,
        total_mom,
        total_energy,
        total_angular,
        delta_mass,
        delta_mom,
        delta_energy,
        delta_angular,
        physical,
        mass_floor,
        energy_floor,
        radius,
        cell_valid,
        count,
    ):
        """Build scalar paired-face admissibility and root validators."""

        def adjacent_valid(face, factor):
            """Check only the two cells changed by one face trial."""
            left = (face - 1) % count
            right = face
            increment = factor - factors[face]
            trial_mass_left = total_mass[left] - increment * delta_mass[face]
            trial_mom_left = total_mom[left] - increment * delta_mom[face]
            trial_energy_left = total_energy[left] - increment * delta_energy[face]
            trial_mass_right = total_mass[right] + increment * delta_mass[face]
            trial_mom_right = total_mom[right] + increment * delta_mom[face]
            trial_energy_right = total_energy[right] + increment * delta_energy[face]
            trial_angular_left = (
                total_angular[left] - increment * delta_angular[face]
                if total_angular is not None
                else None
            )
            trial_angular_right = (
                total_angular[right] + increment * delta_angular[face]
                if total_angular is not None
                else None
            )
            indices = [index for index in (left, right) if physical[index]]
            if not indices:
                return True
            for index in indices:
                trial_mass_value = trial_mass_left if index == left else trial_mass_right
                trial_mom_value = trial_mom_left if index == left else trial_mom_right
                trial_energy_value = trial_energy_left if index == left else trial_energy_right
                trial_angular_value = trial_angular_left if index == left else trial_angular_right
                if not cell_valid(
                    index,
                    trial_mass_value,
                    trial_mom_value,
                    trial_energy_value,
                    trial_angular_value,
                ):
                    return False
            return True

        def analytical_adjacent_factor(face, current):
            """Return a quadratic admissible factor, or ``None``."""
            left = (face - 1) % count
            right = face
            max_increment = 1.0 - current
            if max_increment <= 0.0:
                return 1.0
            largest = _analytical_face_increment_limit(
                (left, right),
                face,
                max_increment,
                total_mass,
                total_mom,
                total_energy,
                delta_mass,
                delta_mom,
                delta_energy,
                physical,
                mass_floor,
                energy_floor,
                total_angular,
                radius,
                delta_angular,
            )
            if largest is None:
                return None
            candidate = current + max(0.0, largest)
            return candidate if adjacent_valid(face, candidate) else None

        return adjacent_valid, analytical_adjacent_factor

    def _analytical_batch_validators(
        self,
        *,
        factors,
        total_mass,
        total_mom,
        total_energy,
        total_angular,
        delta_mass,
        delta_mom,
        delta_energy,
        physical,
        mass_floor,
        energy_floor,
        radius,
        relative_tolerance,
        delta_angular,
        count,
    ):
        """Build vectorized paired-face admissibility and root validators."""

        def batch_cell_valid(
            mass_value,
            momentum_value,
            energy_value,
            indices,
            angular_value=None,
        ):
            """Vectorized admissibility check for disjoint face pairs."""
            mass_limit = mass_floor[indices]
            energy_limit = energy_floor[indices]
            result = (
                np.isfinite(mass_value)
                & np.isfinite(momentum_value)
                & np.isfinite(energy_value)
                & (mass_value >= mass_limit)
            )
            at_floor = result & (mass_value <= mass_limit)
            result[at_floor] = energy_value[at_floor] >= energy_limit[at_floor]
            positive = result & ~at_floor
            if np.any(positive):
                kinetic = np.zeros_like(energy_value)
                kinetic[positive] = 0.5 * momentum_value[positive] ** 2 / mass_value[positive]
                rotational = np.zeros_like(energy_value)
                if angular_value is not None and radius is not None:
                    valid_radius = positive & (radius[indices] > 0.0)
                    rotational[valid_radius] = (
                        0.5
                        * angular_value[valid_radius] ** 2
                        / (mass_value[valid_radius] * radius[indices][valid_radius] ** 2)
                    )
                internal = energy_value - kinetic - rotational
                tolerance = relative_tolerance * np.maximum.reduce(
                    (
                        np.abs(energy_value),
                        kinetic,
                        np.abs(energy_limit),
                        np.full_like(energy_value, _FLOAT_TINY),
                    ),
                )
                tolerance *= 1.0 - 1.0e-8
                result[positive] = (
                    internal[positive] >= energy_limit[positive] - tolerance[positive]
                )
            return result

        def batch_adjacent_valid(face_values, factor_values):
            """Check a disjoint face group without scalar callbacks."""
            left = (face_values - 1) % count
            right = face_values
            increment = factor_values - factors[face_values]
            left_mass = total_mass[left] - increment * delta_mass[face_values]
            left_mom = total_mom[left] - increment * delta_mom[face_values]
            left_energy = total_energy[left] - increment * delta_energy[face_values]
            right_mass = total_mass[right] + increment * delta_mass[face_values]
            right_mom = total_mom[right] + increment * delta_mom[face_values]
            right_energy = total_energy[right] + increment * delta_energy[face_values]
            left_angular = right_angular = None
            if total_angular is not None:
                left_angular = total_angular[left] - increment * delta_angular[face_values]
                right_angular = total_angular[right] + increment * delta_angular[face_values]
            return (
                ~physical[left]
                | batch_cell_valid(
                    left_mass,
                    left_mom,
                    left_energy,
                    left,
                    left_angular,
                )
            ) & (
                ~physical[right]
                | batch_cell_valid(
                    right_mass,
                    right_mom,
                    right_energy,
                    right,
                    right_angular,
                )
            )

        def batch_analytical_factors(face_values):
            """Solve the invariant-domain quadratics for a face group."""
            current = factors[face_values].copy()
            largest = np.ones_like(current) - current
            possible = np.ones(len(face_values), dtype=bool)
            for side, sign in ((face_values - 1, -1.0), (face_values, 1.0)):
                physical_side = physical[side]
                if not np.any(physical_side):
                    continue
                mass0 = total_mass[side]
                mom0 = total_mom[side]
                energy0 = total_energy[side] - energy_floor[side]
                dm = sign * delta_mass[face_values]
                dp = sign * delta_mom[face_values]
                de = sign * delta_energy[face_values]
                finite = (
                    np.isfinite(mass0)
                    & np.isfinite(mom0)
                    & np.isfinite(energy0)
                    & (mass0 >= mass_floor[side])
                )
                possible &= ~physical_side | finite
                mass_bound = np.ones_like(largest)
                decreasing_mass = dm < 0.0
                mass_bound[decreasing_mass] = (
                    mass0[decreasing_mass] - mass_floor[side][decreasing_mass]
                ) / -dm[decreasing_mass]
                largest = np.minimum(largest, mass_bound)
                angular0 = np.zeros_like(mass0)
                angular_delta = np.zeros_like(mass0)
                rotational_factor = np.zeros_like(mass0)
                if total_angular is not None:
                    angular0 = total_angular[side]
                    angular_delta = sign * delta_angular[face_values]
                    radius_side = radius[side]
                    valid_radius = np.isfinite(radius_side) & (radius_side > 0.0)
                    possible &= ~physical_side | valid_radius
                    rotational_factor[valid_radius] = 1.0 / radius_side[valid_radius] ** 2
                a = dm * de - 0.5 * dp**2 - (0.5 * rotational_factor * angular_delta**2)
                b = (
                    mass0 * de
                    + dm * energy0
                    - mom0 * dp
                    - rotational_factor * angular0 * angular_delta
                )
                c = mass0 * energy0 - 0.5 * mom0**2 - 0.5 * rotational_factor * angular0**2
                possible &= ~physical_side | (c >= 0.0)
                bounded = physical_side & (c >= 0.0)
                value_bound = (a * largest + b) * largest + c
                bounded_failure = bounded & (value_bound < 0.0)
                linear = bounded_failure & (np.abs(a) <= _FLOAT_TINY)
                linear_bad = linear & (b >= 0.0)
                possible &= ~linear_bad
                linear_root = np.divide(
                    -c,
                    b,
                    out=np.zeros_like(c),
                    where=linear & (b != 0.0),
                )
                quadratic = bounded_failure & ~linear
                discriminant = b**2 - 4.0 * a * c
                possible &= ~quadratic | (discriminant >= 0.0)
                sqrt_discriminant = np.sqrt(
                    np.maximum(discriminant, 0.0),
                )
                root_a = np.divide(
                    -b - sqrt_discriminant,
                    2.0 * a,
                    out=np.zeros_like(a),
                    where=quadratic & (a != 0.0),
                )
                root_b = np.divide(
                    -b + sqrt_discriminant,
                    2.0 * a,
                    out=np.zeros_like(a),
                    where=quadratic & (a != 0.0),
                )
                root_a = np.where(root_a >= 0.0, root_a, np.inf)
                root_b = np.where(root_b >= 0.0, root_b, np.inf)
                root = np.where(linear, linear_root, np.minimum(root_a, root_b))
                possible &= ~bounded_failure | np.isfinite(root)
                largest = np.minimum(largest, np.where(bounded_failure, root, largest))
            candidates = current + np.maximum(largest, 0.0)
            possible &= batch_adjacent_valid(face_values, candidates)
            return candidates, possible

        return batch_cell_valid, batch_adjacent_valid, batch_analytical_factors

    def _validate_analytical_recovery_state(
        self,
        context,
        geometry_increment,
        geometry_fraction,
    ):
        mass = context["mass"]
        momentum = context["momentum"]
        energy = context["energy"]
        angular = context["angular"]
        radius = context["radius"]
        valid = context["valid"]
        total_mass = mass.copy()
        total_mom = momentum + geometry_fraction * geometry_increment
        total_energy = energy.copy()
        total_angular = angular.copy() if angular is not None else None
        if np.all(valid(total_mass, total_mom, total_energy, total_angular)):
            return total_mass, total_mom, total_energy, total_angular

        invalid = ~valid(total_mass, total_mom, total_energy, total_angular)
        index = int(np.flatnonzero(invalid)[0])
        mass_value = float(total_mass[index])
        momentum_value = float(total_mom[index])
        energy_value = float(total_energy[index])
        kinetic_value = 0.5 * momentum_value**2 / mass_value if mass_value > 0.0 else 0.0
        angular_value = float(total_angular[index]) if total_angular is not None else 0.0
        radius_value = float(radius[index]) if radius is not None else float("nan")
        rotational_value = (
            0.5 * angular_value**2 / (mass_value * radius_value**2)
            if mass_value > 0.0 and radius_value > 0.0
            else 0.0
        )
        internal_value = energy_value - kinetic_value - rotational_value
        specific_value = angular_value / mass_value if mass_value > 0.0 else 0.0
        raise ValueError(
            "hydro state is outside positivity domain before paired "
            f"face construction at cell {index} "
            f"(mass={mass_value} mom={momentum_value} energy={energy_value} "
            f"radius={radius_value} J={angular_value} j={specific_value} "
            f"kinetic={kinetic_value} rotational={rotational_value} "
            f"internal={internal_value})",
        )

    @staticmethod
    def _analytical_face_factor(
        face,
        current,
        adjacent_valid,
        analytical_adjacent_factor,
        factor_method,
        recovery_iterations,
        factor_tolerance,
    ):
        if adjacent_valid(face, 1.0):
            return 1.0
        if factor_method == "analytical":
            accepted = analytical_adjacent_factor(face, current)
            if accepted is not None:
                return accepted
        low, high = current, 1.0
        for _ in range(recovery_iterations):
            middle = 0.5 * (low + high)
            if adjacent_valid(face, middle):
                low = middle
            else:
                high = middle
            if high - low <= factor_tolerance:
                break
        return low

    @staticmethod
    def _apply_analytical_face(
        face,
        accepted,
        factors,
        total_mass,
        total_mom,
        total_energy,
        total_angular,
        delta_mass,
        delta_mom,
        delta_energy,
        delta_angular,
        count,
    ):
        increment = accepted - factors[face]
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
        return increment

    def _recover_analytical_faces(
        self,
        context,
        par,
        mass_face,
        delta_mass,
        delta_mom,
        delta_energy,
        delta_angular,
        geometry_increment,
        geometry_fraction,
    ):
        """Recover paired face factors with sequential analytical limiting."""
        mass = context["mass"]
        physical = context["physical"]
        mass_floor = context["mass_floor"]
        energy_floor = context["energy_floor"]
        radius = context["radius"]
        cell_valid = context["cell_valid"]
        count = len(mass)
        factors = np.zeros(len(mass_face), dtype=float)
        total_mass, total_mom, total_energy, total_angular = (
            self._validate_analytical_recovery_state(
                context,
                geometry_increment,
                geometry_fraction,
            )
        )
        adjacent_valid, analytical_adjacent_factor = self._analytical_scalar_validators(
            factors=factors,
            total_mass=total_mass,
            total_mom=total_mom,
            total_energy=total_energy,
            total_angular=total_angular,
            delta_mass=delta_mass,
            delta_mom=delta_mom,
            delta_energy=delta_energy,
            delta_angular=delta_angular,
            physical=physical,
            mass_floor=mass_floor,
            energy_floor=energy_floor,
            radius=radius,
            cell_valid=cell_valid,
            count=count,
        )
        max_recovery_sweeps = 8
        recovery_iterations = 48
        factor_tolerance = 1.0e-13
        factor_method = str(
            getattr(par, "positivity_factor_method", "invariant_domain"),
        ).lower()
        for sweep in range(max_recovery_sweeps):
            largest_increase = 0.0
            faces = range(len(mass_face)) if sweep % 2 == 0 else range(len(mass_face) - 1, -1, -1)
            for face in faces:
                current = factors[face]
                if current >= 1.0 - factor_tolerance:
                    factors[face] = 1.0
                    continue
                accepted = self._analytical_face_factor(
                    face,
                    current,
                    adjacent_valid,
                    analytical_adjacent_factor,
                    factor_method,
                    recovery_iterations,
                    factor_tolerance,
                )
                largest_increase = max(largest_increase, accepted - current)
                self._apply_analytical_face(
                    face,
                    accepted,
                    factors,
                    total_mass,
                    total_mom,
                    total_energy,
                    total_angular,
                    delta_mass,
                    delta_mom,
                    delta_energy,
                    delta_angular,
                    count,
                )
            if np.all(factors >= 1.0 - factor_tolerance):
                factors[...] = 1.0
                break
            if largest_increase <= factor_tolerance:
                break
        return total_mass, total_mom, total_energy, total_angular, factors

    def _positivity_limited_face_fluxes(
        self,
        fluid,
        dt,
        mesh,
        par,
        mass_face,
        mom_face,
        energy_face,
        geometric_mom=None,
        angular_face=None,
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
        if not getattr(par, "positivity_preserving", True):
            return self._apply_unlimited_face_fluxes(
                fluid,
                dt,
                mesh,
                par,
                mass_face,
                mom_face,
                energy_face,
                geometric_mom,
                angular_face,
            )
        context = self._positivity_context(fluid, mesh, par, dt)
        dt_value = context["dt"]
        mass = context["mass"]
        momentum = context["momentum"]
        energy = context["energy"]
        angular = context["angular"]
        geometry = context["geometry"]
        valid = context["valid"]

        geometry_increment, geometry_fraction = self._limit_geometry_momentum(
            context,
            geometric_mom,
        )

        mass_face = np.asarray(mass_face, dtype=float)
        mom_face = np.asarray(mom_face, dtype=float)
        energy_face = np.asarray(energy_face, dtype=float)
        area_runtime_code = np.asarray(geometry.area_runtime_code, dtype=float)
        delta_mass = dt_value * mass_face * area_runtime_code
        delta_mom = dt_value * mom_face * area_runtime_code
        delta_energy = dt_value * energy_face * area_runtime_code
        delta_angular = (
            dt_value * np.asarray(angular_face, dtype=float) * area_runtime_code
            if angular_face is not None
            else None
        )
        # Accept the unlimited conservative update immediately when possible.
        # This is the overwhelmingly common path and avoids limiter overhead.
        full_mass = mass + delta_mass - ru.periodic_roll(delta_mass, -1)
        full_mom = momentum + geometry_increment + delta_mom - ru.periodic_roll(delta_mom, -1)
        full_energy = energy + delta_energy - ru.periodic_roll(delta_energy, -1)
        full_angular = (
            angular + delta_angular - ru.periodic_roll(delta_angular, -1)
            if angular is not None
            else None
        )
        # Test the complete conservative update, including the spherical
        # pressure source, before limiting either contribution separately.
        # A smooth uniform state can have equal and opposite flux-divergence
        # and geometric terms; limiting those terms independently would
        # manufacture momentum from an otherwise admissible update.
        if np.all(valid(full_mass, full_mom, full_energy, full_angular)):
            fluid.Mass_code[...] = full_mass
            fluid.Mom_code[...] = full_mom
            fluid.Energy_code[...] = full_energy
            if full_angular is not None:
                fluid.AngularMomentum_code[...] = full_angular
            self.last_face_limiter_factors = np.ones_like(
                np.asarray(mass_face, dtype=float),
            )
            return 1.0
        factor_method = str(
            getattr(
                par,
                "positivity_factor_method",
                "invariant_domain",
            ),
        ).lower()
        if factor_method == "invariant_domain":
            mass, momentum, energy, total_angular, factors = self._recover_invariant_domain_faces(
                context,
                mass_face,
                delta_mass,
                delta_mom,
                delta_energy,
                delta_angular,
                geometry_increment,
                geometry_fraction,
            )
        elif np.all(valid(full_mass, full_mom, full_energy, full_angular)):
            factors = np.ones(len(mass_face), dtype=float)
            mass, momentum, energy = full_mass, full_mom, full_energy
            total_angular = full_angular
        else:
            mass, momentum, energy, total_angular, factors = self._recover_analytical_faces(
                context,
                par,
                mass_face,
                delta_mass,
                delta_mom,
                delta_energy,
                delta_angular,
                geometry_increment,
                geometry_fraction,
            )

        mass, momentum, energy = self._repair_wind_reservoir(
            fluid,
            par,
            context,
            mass,
            momentum,
            energy,
        )

        if not np.all(valid(mass, momentum, energy, total_angular)):
            invalid = ~valid(mass, momentum, energy, total_angular)
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                "hydro state is outside positivity domain after face update "
                "at cell %d (mass=%s mom=%s energy=%s)"
                % (index, mass[index], momentum[index], energy[index]),
            )

        fluid.Mass_code[...] = mass
        fluid.Mom_code[...] = momentum
        fluid.Energy_code[...] = energy
        if total_angular is not None:
            fluid.AngularMomentum_code[...] = total_angular
        self.last_face_limiter_factors = factors
        return float(np.min(factors)) if factors.size else 1.0

    def positivity_limited_face_fluxes(self, *args, **kwargs):
        """Apply the positivity-preserving face-flux limiter."""
        return self._positivity_limited_face_fluxes(*args, **kwargs)

    @property
    def last_face_limiter_factors(self):
        """Return the most recent per-face positivity limiter factors."""
        return getattr(self, "_last_face_limiter_factors", None)

    @last_face_limiter_factors.setter
    def last_face_limiter_factors(self, factors):
        self._last_face_limiter_factors = factors

    def _apply_wind_reservoir_flux(self, dt, mesh, fluid, par):
        """Restore rejected WindSph boundary flux as one coupled parcel."""
        if par is None or getattr(getattr(par, "boundary", None), "condition", None) != "WindSph":
            return 0.0
        factors = np.asarray(
            getattr(self, "_last_face_limiter_factors", np.ones(0)),
            dtype=float,
        )
        first = int(par.mesh.ghost_cells)
        geometry = self._geometry_state(mesh, par)
        if first >= len(factors) or first >= len(geometry.area_runtime_code):
            return 0.0
        rejected_fraction = max(0.0, 1.0 - float(factors[first]))
        if rejected_fraction <= 0.0:
            return 0.0

        boundary = par.boundary
        rho_wind = float(np.asarray(boundary.rho_outflow_proper))
        velocity_wind = float(np.asarray(boundary.vel_outflow_proper))
        pressure_wind = float(
            np.asarray(
                fluid.eos.pressure(
                    boundary.rho_outflow_proper,
                    boundary.temperature_outflow_proper,
                    boundary.outflow_mu,
                ),
            ),
        )
        wind_internal = (
            pressure_wind / rho_wind / (fluid.eos.gamma - 1.0)
            if rho_wind > 0.0 and not fluid.eos.is_isothermal
            else 0.0
        )
        wind_specific_energy = 0.5 * velocity_wind**2 + wind_internal
        area_runtime_code = float(np.asarray(geometry.area_runtime_code[first]))
        dt_value = float(np.asarray(dt))
        mass_rate = rho_wind * velocity_wind * area_runtime_code
        momentum_rate = (rho_wind * velocity_wind**2 + pressure_wind) * area_runtime_code
        energy_rate = (
            velocity_wind
            * (
                0.5 * rho_wind * velocity_wind**2
                + fluid.eos.gamma * pressure_wind / (fluid.eos.gamma - 1.0)
            )
            * area_runtime_code
        )
        correction_mass = rejected_fraction * dt_value * mass_rate
        correction_momentum = rejected_fraction * dt_value * momentum_rate
        correction_energy = rejected_fraction * dt_value * energy_rate
        fluid.Mass_code[first] += correction_mass
        fluid.Mom_code[first] += correction_momentum
        fluid.Energy_code[first] += correction_energy
        if hasattr(fluid, "InternalEnergy_code"):
            correction_internal = correction_energy - (
                velocity_wind * correction_momentum - 0.5 * velocity_wind**2 * correction_mass
            )
            fluid.InternalEnergy_code[first] += correction_internal
        mass = np.asarray(fluid.Mass_code, dtype=float)
        momentum = np.asarray(fluid.Mom_code, dtype=float)
        energy = np.asarray(fluid.Energy_code, dtype=float)
        kinetic = 0.5 * momentum[first] ** 2 / mass[first]
        if energy[first] < kinetic:
            # A rejected parcel can still be too fast for the receiving cell
            # when its velocity differs substantially from the wind.  Add
            # only the minimum further parcel of the same wind state needed
            # to make the combined conservative state admissible.
            additional_mass = self._minimum_wind_energy_repair(
                mass[first],
                momentum[first],
                energy[first],
                correction_mass,
                velocity_wind,
                wind_specific_energy,
            )
            mass[first] += additional_mass
            momentum[first] += additional_mass * velocity_wind
            energy[first] += additional_mass * wind_specific_energy
            if hasattr(fluid, "InternalEnergy_code"):
                fluid.InternalEnergy_code[first] += additional_mass * (
                    wind_specific_energy - 0.5 * velocity_wind**2
                )
            self._last_wind_reservoir_mass += additional_mass
        kinetic = 0.5 * momentum[first] ** 2 / mass[first]
        if not (
            np.isfinite(mass[first])
            and np.isfinite(momentum[first])
            and np.isfinite(energy[first])
            and mass[first] > 0.0
            and energy[first] >= kinetic
        ):
            raise ValueError(
                "WindSph reservoir correction produced an inadmissible "
                "conserved state at cell %d" % first,
            )
        self._last_wind_reservoir_mass = (
            getattr(self, "_last_wind_reservoir_mass", 0.0) + correction_mass
        )
        return correction_mass

    @staticmethod
    def _minimum_wind_energy_repair(
        base_mass,
        base_momentum,
        base_energy,
        correction_mass,
        velocity_wind,
        wind_specific_energy,
    ):
        def reservoir_valid(delta_mass):
            trial_mass = base_mass + delta_mass
            trial_momentum = base_momentum + delta_mass * velocity_wind
            trial_energy = base_energy + delta_mass * wind_specific_energy
            return trial_energy >= (0.5 * trial_momentum**2 / trial_mass)

        low = 0.0
        high = max(base_mass, correction_mass, 1.0) * 1.0e-12
        for _ in range(96):
            if reservoir_valid(high):
                break
            high *= 2.0
        if not reservoir_valid(high):
            raise ValueError("WindSph reservoir correction could not restore conservative energy")
        for _ in range(64):
            middle = 0.5 * (low + high)
            if reservoir_valid(middle):
                high = middle
            else:
                low = middle
        return high

    def SetInterFaceFlux(self, mesh, fluid, boundcond, method="Rusanov", verbose=None, order=0):  # noqa: N802
        """Set interface fluxes using GLF, Rusanov, or HLLC fluxes."""
        from .flux_pipeline import set_interface_flux  # noqa: PLC0415

        return set_interface_flux(
            self,
            mesh,
            fluid,
            boundcond,
            method=method,
            verbose=verbose,
            order=order,
        )

    def _prepare_flux_updates(self, mesh, fluid, par):
        # Shift the face fluxes so each cell receives the net in-flow minus
        # out-flow through its two bounding faces.
        geometry = self._geometry_state(mesh, par)
        area_runtime_code = geometry.area_runtime_code
        pressure_runtime_code = None
        velocity_runtime_code = None
        if getattr(mesh, "coordsys", None) == "spherical" or (
            self._dual_energy_enabled(par) and hasattr(fluid, "InternalEnergy_code")
        ):
            _, velocity_runtime_code, pressure_runtime_code, _ = self._active_primitive_arrays(
                fluid,
                par,
            )
        fluid.Mass_code.flux * area_runtime_code - ru.periodic_roll(
            fluid.Mass_code.flux * area_runtime_code,
            -1,
        )
        df_Mom_code = fluid.Mom_code.flux * area_runtime_code - ru.periodic_roll(
            fluid.Mom_code.flux * area_runtime_code,
            -1,
        )
        fluid.Energy_code.flux * area_runtime_code - ru.periodic_roll(
            fluid.Energy_code.flux * area_runtime_code,
            -1,
        )
        df_AngularMomentum = None
        if hasattr(fluid, "AngularMomentum_code"):
            angular_flux_area = fluid.AngularMomentum_code.flux * area_runtime_code
            df_AngularMomentum = angular_flux_area - ru.periodic_roll(angular_flux_area, -1)
        potential_face = self._gravity_potential_faces(mesh, par)
        df_potential = None
        if potential_face is not None:
            potential_flux_area = potential_face * fluid.Mass_code.flux * area_runtime_code
            df_potential = potential_flux_area - ru.periodic_roll(potential_flux_area, -1)
        if getattr(mesh, "coordsys", None) == "spherical":
            # Spherical momentum needs the geometric pressure term from the
            # changing face area, not just the flux divergence.
            area_right = ru.periodic_roll(area_runtime_code, -1)
            df_Mom_code += pressure_runtime_code * (area_right - area_runtime_code)

        dual_energy = (
            self._dual_energy_enabled(par)
            and hasattr(fluid, "InternalEnergy_code")
            and getattr(fluid.eos, "is_polytropic", False)
        )
        df_InternalEnergy = None
        if dual_energy:
            velocity_left = np.asarray(velocity_runtime_code.L, dtype=float)
            velocity_right = np.asarray(velocity_runtime_code.R, dtype=float)
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
            if hasattr(fluid, "rotational_energy_flux"):
                total_energy_flux -= np.asarray(
                    fluid.rotational_energy_flux,
                    dtype=float,
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
            df_InternalEnergy = internal_flux * area_runtime_code - ru.periodic_roll(
                internal_flux * area_runtime_code,
                -1,
            )
            if getattr(mesh, "coordsys", None) == "spherical":
                # Account for spherical pressure work using the same
                # interface pressure implied by the Riemann momentum flux.
                df_InternalEnergy -= (
                    ru.periodic_roll(face_pressure * face_velocity * area_runtime_code, -1)
                    - face_pressure * face_velocity * area_runtime_code
                )

        geometric_mom = None
        if getattr(mesh, "coordsys", None) == "spherical":
            area_right = ru.periodic_roll(area_runtime_code, -1)
            geometric_mom = pressure_runtime_code * (area_right - area_runtime_code)

        return {
            "area_runtime_code": area_runtime_code,
            "pressure_runtime_code": pressure_runtime_code,
            "velocity_runtime_code": velocity_runtime_code,
            "df_angular": df_AngularMomentum,
            "potential_face": potential_face,
            "df_internal": df_InternalEnergy,
            "internal_flux": internal_flux if df_InternalEnergy is not None else None,
            "face_velocity": face_velocity if df_InternalEnergy is not None else None,
            "geometric_mom": geometric_mom,
        }

    def _update_dual_energy_from_flux(
        self,
        dt,
        mesh,
        fluid,
        par,
        old_mass_for_internal,
        area_runtime_code,
        pressure_runtime_code,
        internal_flux,
        face_velocity,
    ):
        # Couple the dual-energy advection to the same face coefficients
        # used by the conservative update.  Applying the minimum face
        # coefficient globally defeats the purpose of the local limiter.
        factors = np.asarray(
            getattr(self, "_last_face_limiter_factors", np.ones(len(fluid.Mass_code.flux))),
            dtype=float,
        )
        limited_internal_flux = np.asarray(internal_flux, dtype=float) * factors
        first = int(par.mesh.ghost_cells)
        count = int(par.mesh.grid_cells)
        physical = np.zeros(len(fluid.InternalEnergy_code), dtype=bool)
        physical[first : first + count] = True
        internal_factors = self._positivity_limited_internal_flux(
            fluid.InternalEnergy_code,
            limited_internal_flux,
            area_runtime_code,
            dt,
            physical,
        )
        limited_internal_flux *= internal_factors
        limited_df_internal = limited_internal_flux * area_runtime_code - ru.periodic_roll(
            limited_internal_flux * area_runtime_code,
            -1,
        )
        if getattr(mesh, "coordsys", None) == "spherical":
            # Retain the established spherical pressure-work
            # discretization.  The positivity limiter acts on the
            # Riemann internal-energy flux above; changing the geometric
            # source and limiting it as a scalar face flux simultaneously
            # can over-limit cold expanding cells.
            limited_df_internal -= pressure_runtime_code * (
                ru.periodic_roll(
                    factors * face_velocity * area_runtime_code,
                    -1,
                )
                - factors * face_velocity * area_runtime_code
            )
        candidate_internal = (
            np.asarray(fluid.InternalEnergy_code, dtype=float) + limited_df_internal * dt
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
            mesh,
            fluid,
            getattr(mesh, "par", getattr(mesh, "_par", None)),
        )
        first = int(par.mesh.ghost_cells)
        count = int(par.mesh.grid_cells)
        physical = np.zeros(len(candidate_internal), dtype=bool)
        physical[first : first + count] = True
        fallback = (
            physical
            & (~np.isfinite(candidate_internal) | (candidate_internal <= 0.0))
            & np.isfinite(conservative_internal)
            & (conservative_internal > 0.0)
        )
        if np.any(fallback):
            candidate_internal[fallback] = conservative_internal[fallback]
            self.dual_energy_pressure_fallback_count += int(
                np.count_nonzero(fallback),
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
                np.count_nonzero(retain_previous),
            )

        # A positivity limiter alone can still leave a tiny positive
        # value after a large cancellation in the spherical pressure-work
        # update.  Treat an abrupt loss below the configured consistency
        # fraction as a failed dual estimate as well.  Prefer E-K when it
        # is admissible; otherwise keep the previous positive dual state.
        consistency_factor = max(
            0.0,
            float(
                np.asarray(getattr(par, "dual_energy_consistency_factor", 1.0e-1), dtype=float),
            ),
        )
        far_below_previous = (
            physical
            & np.isfinite(candidate_internal)
            & np.isfinite(previous_internal)
            & (previous_internal > 0.0)
            & (candidate_internal < consistency_factor * previous_internal)
        )
        conservative_recovery = (
            far_below_previous & np.isfinite(conservative_internal) & (conservative_internal > 0.0)
        )
        if np.any(conservative_recovery):
            candidate_internal[conservative_recovery] = conservative_internal[conservative_recovery]
            self.dual_energy_pressure_fallback_count += int(
                np.count_nonzero(conservative_recovery),
            )
        retain_consistent = far_below_previous & ~conservative_recovery
        if np.any(retain_consistent):
            candidate_internal[retain_consistent] = previous_internal[retain_consistent]
            self.dual_energy_pressure_fallback_count += int(
                np.count_nonzero(retain_consistent),
            )

        # Entropy-stable dual-energy correction for smooth cells.  For an
        # adiabatic ideal gas, the cell entropy proxy is proportional to
        # e/rho**gamma.  The Riemann internal-energy update may lose this
        # quantity through cancellation in the spherical pressure-work
        # term, even while remaining positive.  Preserve the previous
        # entropy only for moderate density changes; strong compression,
        # expansion, and near-vacuum cells are left to the conservative
        # consistency/fallback logic above.
        if getattr(
            par,
            "dual_energy_entropy_limiter",
            False,
        ) and not self._thermochemistry_enabled(fluid, par):
            volume = np.asarray(
                self._geometry_state(mesh, par).volume_runtime_code,
                dtype=float,
            )
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
            moderate_density_change = physical & (density_ratio >= 0.5) & (density_ratio <= 2.0)  # noqa: PLR2004
            isentropic_internal = previous_internal * np.maximum(
                density_ratio,
                0.0,
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
                candidate_internal[entropy_limited] = isentropic_internal[entropy_limited]
                self.dual_energy_entropy_limiter_count += int(
                    np.count_nonzero(entropy_limited),
                )

        # Leave unresolved cells at zero only when the conservative state
        # is also non-positive.  SetPrimitive will then apply the
        # configured positive floor and record that injected energy.
        fluid.InternalEnergy_code = as_named_array(
            np.maximum(candidate_internal, 0.0),
        )

    def AddFluxes(self, dt: float, mesh, fluid, boundcond):  # noqa: N802
        """Apply interface fluxes to conserved quantities and advance time."""
        old_mass_for_internal = np.asarray(fluid.Mass_code, dtype=float).copy()
        par = getattr(mesh, "par", getattr(mesh, "_par", None))
        self._limit_angular_momentum_flux(dt, mesh, fluid, par)
        updates = self._prepare_flux_updates(mesh, fluid, par)
        area_runtime_code = updates["area_runtime_code"]
        self.positivity_limited_face_fluxes(
            fluid,
            dt,
            mesh,
            par,
            fluid.Mass_code.flux,
            fluid.Mom_code.flux,
            fluid.Energy_code.flux,
            geometric_mom=updates["geometric_mom"],
            angular_face=(
                fluid.AngularMomentum_code.flux if updates["df_angular"] is not None else None
            ),
        )
        self._apply_wind_reservoir_flux(dt, mesh, fluid, par)
        if updates["df_internal"] is not None:
            self._update_dual_energy_from_flux(
                dt,
                mesh,
                fluid,
                par,
                old_mass_for_internal,
                area_runtime_code,
                updates["pressure_runtime_code"],
                updates["internal_flux"],
                updates["face_velocity"],
            )
        if updates["potential_face"] is not None:
            factors = np.asarray(
                getattr(self, "_last_face_limiter_factors", np.ones(len(fluid.Mass_code.flux))),
                dtype=float,
            )
            limited_potential_flux_area = (
                updates["potential_face"] * fluid.Mass_code.flux * factors * area_runtime_code
            )
            fluid.GravitationalPotentialEnergy_code += dt * (
                limited_potential_flux_area - ru.periodic_roll(limited_potential_flux_area, -1)
            )
        if getattr(par, "supercomoving_coordinates", False):
            fluid.tau_supercomoving_code += dt
        else:
            fluid.time_proper_code += dt

    def _gravity_model(self, *args, **kwargs):
        from .gravity_sources import _gravity_model  # noqa: PLC0415

        return _gravity_model(self, *args, **kwargs)

    def gravity_model(self, *args, **kwargs):
        """Return the configured gravity model."""
        return self._gravity_model(*args, **kwargs)

    def ApplyGravity(self, *args, **kwargs):  # noqa: N802
        from .gravity_sources import ApplyGravity  # noqa: PLC0415

        return ApplyGravity(self, *args, **kwargs)

    def AdvectIonizationFraction(self, dt, mesh, fluid, par, old_mass, mass_flux):  # noqa: N802
        """Advect the chemistry fraction consistently with the mass flux."""
        return rtc.advect_ionization_fraction(
            dt,
            mesh,
            fluid,
            par,
            old_mass,
            mass_flux,
        )

    def GetSourceTimestepFast(self, mesh, fluid, par, remaining):  # noqa: N802
        """Return a source substep for RT-coupled heating/chemistry."""
        return rtc.get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining)

    def ApplyThermochemistryFast(self, dt, mesh, fluid, par, transport_result=None):  # noqa: N802
        """Fast source update for RT-coupled thermo-chemistry tests."""
        return rtc.apply_thermochemistry_fast(
            dt,
            mesh,
            fluid,
            par,
            transport_result=transport_result,
        )

    def ApplyRadiationPressure(self, dt, mesh, fluid, par, source_result):  # noqa: N802
        """Apply momentum from photons consumed by the thermo-chemistry step."""
        from .radiation import apply_radiation_pressure  # noqa: PLC0415

        return apply_radiation_pressure(
            self,
            dt,
            mesh,
            fluid,
            par,
            source_result,
        )

    def SetBoundary(self, mesh, fluid, par):  # noqa: N802
        """Fill ghost cells according to the selected boundary condition."""
        from .boundaries import set_boundary  # noqa: PLC0415

        return set_boundary(self, mesh, fluid, par)

    def GetTimeStep(self, mesh, fluid, par, CFL=None):  # noqa: N802, N803
        """Return a CFL-limited timestep in the active time coordinate."""
        from .timestep import get_time_step  # noqa: PLC0415

        return get_time_step(self, mesh, fluid, par, CFL=CFL)
