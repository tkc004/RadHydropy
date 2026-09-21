# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Optional one-dimensional long-characteristic radiative transfer."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from radhydropy.constants import DEFAULT_SIGMA_GAMMA_CGS_CM2, PROTON_MASS_CGS, SPEED_OF_LIGHT_CGS
from radhydropy.units import (
    CGS_AREA_UNIT,
    PHOTON_FLUX_UNIT,
    PHOTON_RATE_UNIT,
    _as_cgs_float,
    _code_units,
    code_quantity_to_cgs,
)


@dataclass
class LongCharacteristicResult:
    """Photon field returned by a one-dimensional long-characteristic trace."""

    optical_depth: np.ndarray
    face_photon_flux: np.ndarray
    face_photon_rate: np.ndarray
    cell_photon_flux: np.ndarray
    cell_photon_density: np.ndarray
    absorbed_photon_rate: np.ndarray


@dataclass
class TransportGeometry:
    """Normalized one-dimensional geometry used by radiation transport."""

    boundary_cgs_cm: np.ndarray
    width_cgs_cm: np.ndarray
    volume_cgs_cm3: np.ndarray
    face_area_cgs_cm2: np.ndarray
    coordsys: str


@dataclass
class CausalCellResult:
    """Transport result for one causally ordered cell."""

    outgoing_rate: np.ndarray
    absorbed_rate: np.ndarray
    photon_density: np.ndarray
    attenuation: np.ndarray


def _parameter_value(par, name, default=None):
    """Read a parameter from the flat store or nested parameter group."""
    value = getattr(par, name, None)
    if value is not None:
        return value
    parameter = getattr(par, "_parameter", None)
    return parameter(name, default) if parameter is not None else default


def parameter_value(par, name, default=None):
    """Read a flat or nested runtime parameter for solver orchestration."""
    return _parameter_value(par, name, default)


def _safe_exp_neg(tau):
    tau = np.asarray(tau, dtype=float)
    return np.exp(-np.clip(tau, 0.0, 700.0))


def species_photoionization_rates(ngamma_cgs_cm3, sigma_by_species):
    """Return photoionization and photoheating rates for each absorber."""
    ngamma_cgs_cm3 = np.asarray(ngamma_cgs_cm3, dtype=float)
    rates_cgs_s = {}
    for species, sigma_gamma_cgs_cm2 in sigma_by_species.items():
        sigma_gamma_cgs_cm2 = np.asarray(sigma_gamma_cgs_cm2, dtype=float)
        if ngamma_cgs_cm3.ndim == 2 and sigma_gamma_cgs_cm2.ndim == 1:  # noqa: PLR2004
            sigma_gamma_cgs_cm2 = sigma_gamma_cgs_cm2[:, None]
        rate_cgs_s = SPEED_OF_LIGHT_CGS * sigma_gamma_cgs_cm2 * ngamma_cgs_cm3
        rates_cgs_s[species] = np.sum(rate_cgs_s, axis=0) if rate_cgs_s.ndim > 1 else rate_cgs_s
    return rates_cgs_s


def species_photoionization_heating(ngamma_cgs_cm3, sigma_by_species, epsilon_by_species):
    rates_cgs_erg_cm3_s = {}
    for species, sigma_gamma_cgs_cm2 in sigma_by_species.items():
        epsilon_gamma_cgs_erg = np.asarray(
            epsilon_by_species.get(species, 0.0),
            dtype=float,
        )
        sigma_gamma_cgs_cm2 = np.asarray(sigma_gamma_cgs_cm2, dtype=float)
        if ngamma_cgs_cm3.ndim == 2:  # noqa: PLR2004
            sigma_gamma_cgs_cm2 = (
                sigma_gamma_cgs_cm2[:, None]
                if sigma_gamma_cgs_cm2.ndim == 1
                else sigma_gamma_cgs_cm2
            )
            epsilon_gamma_cgs_erg = (
                epsilon_gamma_cgs_erg[:, None]
                if epsilon_gamma_cgs_erg.ndim == 1
                else epsilon_gamma_cgs_erg
            )
        rate_cgs_erg_cm3_s = (
            SPEED_OF_LIGHT_CGS * sigma_gamma_cgs_cm2 * epsilon_gamma_cgs_erg * ngamma_cgs_cm3
        )
        rates_cgs_erg_cm3_s[species] = (
            np.sum(rate_cgs_erg_cm3_s, axis=0)
            if rate_cgs_erg_cm3_s.ndim > 1
            else rate_cgs_erg_cm3_s
        )
    return rates_cgs_erg_cm3_s


def _attenuation_mean(tau):
    """Return ``(1 - exp(-tau)) / tau`` with the small-tau limit."""
    tau = np.asarray(tau, dtype=float)
    mean = np.ones_like(tau, dtype=float)
    valid = np.absolute(tau) > 1.0e-10  # noqa: PLR2004
    mean[valid] = -np.expm1(-tau[valid]) / tau[valid]
    return mean


def _quantity_or_code_to_cgs(value, code_units, cgs_unit, scale_key):
    if hasattr(value, "to_value"):
        return _as_cgs_float(value, cgs_unit)
    if code_units is not None:
        return code_quantity_to_cgs(value, code_units, scale_key)
    return np.asarray(value, dtype=float)


def _as_cgs_array(value, unit):
    """Convert a physical quantity to cgs, or validate a cgs numeric value."""
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(value, dtype=float)


def _plain_cgs_geometry(name, value):
    """Validate a canonical, unitless cgs geometry array."""
    if value is None or hasattr(value, "units") or hasattr(value, "to_value"):
        raise TypeError(f"{name} must be a plain numeric cgs array")
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional cgs array")
    return result


def _mesh_boundary_cgs_cm(mesh):
    if hasattr(mesh, "boundary_cgs_cm"):
        return _plain_cgs_geometry("boundary_cgs_cm", mesh.boundary_cgs_cm)
    if hasattr(mesh, "boundary"):
        raise ValueError(
            "radiative-transfer geometry must use canonical boundary_cgs_cm; "
            "convert code-unit geometry explicitly before tracing",
        )
    raise AttributeError("radiative-transfer geometry requires boundary_cgs_cm")


def _cell_widths_cm(mesh):
    if hasattr(mesh, "width_cgs_cm"):
        return _plain_cgs_geometry("width_cgs_cm", mesh.width_cgs_cm)
    boundary_cgs_cm = _mesh_boundary_cgs_cm(mesh)
    return np.absolute(boundary_cgs_cm[1:] - boundary_cgs_cm[:-1])


def _cell_volumes_cgs_cm3(mesh, coordsys):
    if hasattr(mesh, "volume_cgs_cm3"):
        return _plain_cgs_geometry("volume_cgs_cm3", mesh.volume_cgs_cm3)
    if hasattr(mesh, "vol"):
        raise ValueError(
            "radiative-transfer geometry must use canonical volume_cgs_cm3",
        )
    boundary = _mesh_boundary_cgs_cm(mesh)
    if coordsys == "spherical":
        return np.absolute(boundary[1:] ** 3 - boundary[:-1] ** 3) * 4.0 * np.pi / 3.0
    return _cell_widths_cm(mesh)


def _face_areas_cgs_cm2(mesh, coordsys):
    boundary = _mesh_boundary_cgs_cm(mesh)
    if coordsys == "spherical":
        return 4.0 * np.pi * boundary**2
    if hasattr(mesh, "face_area_cgs_cm2") and mesh.face_area_cgs_cm2 is not None:
        area = _plain_cgs_geometry("face_area_cgs_cm2", mesh.face_area_cgs_cm2)
        if len(area) == len(boundary):
            return area
        if len(area) == len(boundary) - 1:
            return np.ones(len(boundary)) * area[0]
    elif hasattr(mesh, "area"):
        raise ValueError(
            "radiative-transfer geometry must use canonical face_area_cgs_cm2",
        )
    return np.ones(len(boundary))


def build_transport_geometry(mesh, coordsys=None):
    """Return normalized geometry for one-dimensional radiation transport."""
    coordsys = coordsys or getattr(mesh, "coordsys", "cartesian")
    if coordsys not in ("cartesian", "spherical"):
        raise ValueError(f"coordsys unknown: {coordsys}")
    boundary = _mesh_boundary_cgs_cm(mesh)
    width = (
        _plain_cgs_geometry("width_cgs_cm", mesh.width_cgs_cm)
        if hasattr(mesh, "width_cgs_cm")
        else _cell_widths_cm(mesh)
    )
    volume = _cell_volumes_cgs_cm3(mesh, coordsys)
    if width.shape != volume.shape or width.shape != (boundary.size - 1,):
        raise ValueError("radiative-transfer geometry arrays have inconsistent shapes")
    return TransportGeometry(
        boundary_cgs_cm=boundary,
        width_cgs_cm=width,
        volume_cgs_cm3=volume,
        face_area_cgs_cm2=_face_areas_cgs_cm2(mesh, coordsys),
        coordsys=coordsys,
    )


def propagate_causal_cell(geometry, incoming_rate, optical_depth, cell_index, direction=1):
    """Propagate grouped photon rates through one causal cell.

    ``incoming_rate`` and ``optical_depth`` have one value per group. The
    returned absorption is a photon rate, while ``photon_density`` is the
    cell-averaged number density used by chemistry.
    """
    incoming_rate = np.asarray(incoming_rate, dtype=float)
    optical_depth = np.maximum(np.asarray(optical_depth, dtype=float), 0.0)
    if incoming_rate.size == 1 and optical_depth.size == 1:
        incoming = float(incoming_rate[0])
        tau = float(optical_depth[0])
        attenuation = float(np.exp(-np.clip(tau, 0.0, 700.0)))
        absorbed_rate = incoming * float(-np.expm1(-tau))
        width = geometry.width_cgs_cm[cell_index]
        volume = geometry.volume_cgs_cm3[cell_index]
        attenuation_mean = float(-np.expm1(-tau) / tau) if abs(tau) > 1e-10 else 1.0  # noqa: PLR2004
        if geometry.coordsys == "spherical":
            photon_density = incoming * width * attenuation_mean / volume / SPEED_OF_LIGHT_CGS
        else:
            face_index = cell_index if direction >= 0 else cell_index + 1
            area = geometry.face_area_cgs_cm2[face_index]
            incoming_flux = incoming / area if area > 0.0 else 0.0
            photon_density = incoming_flux * attenuation_mean / SPEED_OF_LIGHT_CGS
        return CausalCellResult(
            outgoing_rate=np.asarray([incoming * attenuation]),
            absorbed_rate=np.asarray([absorbed_rate]),
            photon_density=np.asarray([photon_density]),
            attenuation=np.asarray([attenuation]),
        )
    attenuation = _safe_exp_neg(optical_depth)
    absorbed_rate = incoming_rate * (-np.expm1(-optical_depth))
    face_index = cell_index if direction >= 0 else cell_index + 1
    width = geometry.width_cgs_cm[cell_index]
    volume = geometry.volume_cgs_cm3[cell_index]
    if geometry.coordsys == "spherical":
        photon_density = (
            incoming_rate * width * _attenuation_mean(optical_depth) / volume / SPEED_OF_LIGHT_CGS
        )
    else:
        area = geometry.face_area_cgs_cm2[face_index]
        incoming_flux = np.divide(
            incoming_rate,
            area,
            out=np.zeros_like(incoming_rate),
            where=area > 0.0,
        )
        photon_density = incoming_flux * _attenuation_mean(optical_depth) / SPEED_OF_LIGHT_CGS
    return CausalCellResult(
        outgoing_rate=incoming_rate * attenuation,
        absorbed_rate=absorbed_rate,
        photon_density=photon_density,
        attenuation=attenuation,
    )


def _face_flux_from_rate(face_rate, face_area_cgs_cm2):
    flux = np.zeros(len(face_rate), dtype=float)
    valid = face_area_cgs_cm2 > 0.0
    flux[valid] = face_rate[valid] / face_area_cgs_cm2[valid]
    return flux


def _trace_cartesian(mesh, optical_depth, boundary_flux, direction):
    ncell = len(optical_depth)
    face_area = _face_areas_cgs_cm2(mesh, "cartesian")
    volumes = _cell_volumes_cgs_cm3(mesh, "cartesian")
    speed_of_light = SPEED_OF_LIGHT_CGS

    attenuation = _safe_exp_neg(optical_depth)
    mean_attenuation = _attenuation_mean(optical_depth)
    boundary_flux = _as_cgs_float(boundary_flux, PHOTON_FLUX_UNIT)

    if direction >= 0:
        face_flux = np.empty(ncell + 1, dtype=float)
        face_flux[0] = boundary_flux
        if ncell > 0:
            face_flux[1:] = boundary_flux * np.cumprod(attenuation)
        cell_flux = face_flux[:-1] * mean_attenuation
        absorbed_rate = (face_flux[:-1] * face_area[:-1] - face_flux[1:] * face_area[1:]) / volumes
    else:
        face_flux = np.empty(ncell + 1, dtype=float)
        face_flux[-1] = boundary_flux
        if ncell > 0:
            face_flux[:-1] = boundary_flux * np.cumprod(attenuation[::-1])[::-1]
        cell_flux = face_flux[1:] * mean_attenuation
        absorbed_rate = (face_flux[1:] * face_area[1:] - face_flux[:-1] * face_area[:-1]) / volumes

    face_rate = face_flux * face_area
    cell_density = cell_flux / speed_of_light
    return LongCharacteristicResult(
        optical_depth=optical_depth,
        face_photon_flux=np.asarray(face_flux, dtype=float),
        face_photon_rate=np.asarray(face_rate, dtype=float),
        cell_photon_flux=np.asarray(cell_flux, dtype=float),
        cell_photon_density=np.asarray(cell_density, dtype=float),
        absorbed_photon_rate=np.asarray(absorbed_rate, dtype=float),
    )


def _spherical_boundary_rate(face_area, boundary_flux, source_photon_rate, direction):
    source_rate = _as_cgs_float(source_photon_rate, PHOTON_RATE_UNIT)
    if source_rate != 0.0:
        return source_rate
    boundary_flux = _as_cgs_float(boundary_flux, PHOTON_FLUX_UNIT)
    boundary_area = face_area[0] if direction >= 0 else face_area[-1]
    return boundary_flux * boundary_area


def _trace_spherical(
    mesh,
    optical_depth,
    boundary_flux,
    source_photon_rate,
    direction,
):
    ncell = len(optical_depth)
    face_area = _face_areas_cgs_cm2(mesh, "spherical")
    volumes = _cell_volumes_cgs_cm3(mesh, "spherical")
    widths = _cell_widths_cm(mesh)
    speed_of_light = SPEED_OF_LIGHT_CGS

    attenuation = _safe_exp_neg(optical_depth)
    mean_attenuation = _attenuation_mean(optical_depth)
    incoming_rate = _spherical_boundary_rate(
        face_area,
        boundary_flux,
        source_photon_rate,
        direction,
    )

    if direction >= 0:
        prefix = np.ones(ncell, dtype=float)
        if ncell > 1:
            prefix[1:] = np.cumprod(attenuation[:-1])
        face_rate = np.empty(ncell + 1, dtype=float)
        face_rate[0] = incoming_rate
        face_rate[1:] = incoming_rate * np.cumprod(attenuation)
        cell_density = incoming_rate * prefix * widths * mean_attenuation / volumes / speed_of_light
        absorbed_rate = incoming_rate * prefix * (1.0 - attenuation) / volumes
    else:
        suffix_face = np.ones(ncell, dtype=float)
        suffix_cell = np.ones(ncell, dtype=float)
        if ncell > 1:
            suffix_face[:-1] = np.cumprod(attenuation[::-1])[::-1]
            suffix_cell[:-1] = np.cumprod(attenuation[::-1])[:-1][::-1]
        face_rate = np.empty(ncell + 1, dtype=float)
        face_rate[:-1] = incoming_rate * suffix_face
        face_rate[-1] = incoming_rate
        cell_density = (
            incoming_rate * suffix_cell * widths * mean_attenuation / volumes / speed_of_light
        )
        absorbed_rate = incoming_rate * suffix_cell * (1.0 - attenuation) / volumes

    face_flux = _face_flux_from_rate(face_rate, face_area)
    cell_flux = cell_density * speed_of_light
    return LongCharacteristicResult(
        optical_depth=optical_depth,
        face_photon_flux=np.asarray(face_flux, dtype=float),
        face_photon_rate=np.asarray(face_rate, dtype=float),
        cell_photon_flux=np.asarray(cell_flux, dtype=float),
        cell_photon_density=np.asarray(cell_density, dtype=float),
        absorbed_photon_rate=np.asarray(absorbed_rate, dtype=float),
    )


def _normalize_group_edges(group_edges_eV):  # noqa: N803
    """Validate group edges and return the number of photon groups."""
    if group_edges_eV is None:
        return None
    edges = _as_cgs_array(group_edges_eV, 1.0)
    if edges.ndim != 1 or edges.size < 2:  # noqa: PLR2004
        raise ValueError("radiation_group_edges_eV requires at least two edges")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("radiation_group_edges_eV must be strictly increasing")
    return edges.size - 1


def _normalize_group_values(value, ngroup, name, unit):
    """Return a value as a one-dimensional array with one entry per group."""
    values = _as_cgs_array(value, unit)
    if values.ndim == 0:
        return np.full(ngroup, float(values), dtype=float)
    if values.ndim != 1 or values.size != ngroup:
        raise ValueError(f"{name} must be a scalar or have shape ({ngroup},)")
    return values.astype(float, copy=False)


def _build_group_optical_depth(
    mesh,
    absorber_densities,
    cross_sections_cgs_cm2,
    ngroup,
):
    """Build optical depth per photon group from absorber densities."""
    widths = _cell_widths_cm(mesh)
    optical_depth = np.zeros((ngroup, widths.size), dtype=float)
    for species, density in absorber_densities.items():
        density = np.asarray(density, dtype=float)
        if density.shape != widths.shape:
            raise ValueError(
                f"absorber density for {species!r} must have shape {widths.shape}",
            )
        if species not in cross_sections_cgs_cm2:
            raise ValueError(f"missing cross section for absorber {species!r}")
        sigma = _normalize_group_values(
            cross_sections_cgs_cm2[species],
            ngroup,
            f"cross_sections_cgs_cm2[{species!r}]",
            CGS_AREA_UNIT,
        )
        optical_depth += sigma[:, None] * density[None, :] * widths[None, :]
    return np.maximum(optical_depth, 0.0)


def _stack_group_results(group_results):
    """Stack per-group transport results into the canonical grouped shape."""
    fields = (
        "optical_depth",
        "face_photon_flux",
        "face_photon_rate",
        "cell_photon_flux",
        "cell_photon_density",
        "absorbed_photon_rate",
    )
    values = []
    for field in fields:
        stacked = np.stack([getattr(result, field) for result in group_results])
        values.append(stacked)
    return LongCharacteristicResult(*values)


def trace_long_characteristics(
    mesh,
    *,
    boundary_flux=0.0,
    source_photon_rate=0.0,
    direction=1,
    coordsys=None,
    group_edges_eV=None,  # noqa: N803
    absorber_densities=None,
    cross_sections_cgs_cm2=None,
):
    """Trace grouped photon transport through one-dimensional opacity.

    ``absorber_densities`` maps absorber names to cell-centered number
    densities in ``cm**-3``. ``cross_sections_cgs_cm2`` maps the same names to
    one cross section per photon group in ``cm**2``. All result arrays have
    shape ``(ngroup, ncell)``; a single group is represented as ``ngroup=1``.
    """
    coordsys = coordsys or getattr(mesh, "coordsys", "cartesian")
    if coordsys not in ("cartesian", "spherical"):
        raise ValueError(f"coordsys unknown: {coordsys}")
    geometry = build_transport_geometry(mesh, coordsys)

    edge_ngroup = _normalize_group_edges(group_edges_eV)

    if absorber_densities is None or cross_sections_cgs_cm2 is None:
        raise ValueError(
            "absorber_densities and cross_sections_cgs_cm2 are required",
        )

    if not absorber_densities:
        raise ValueError("at least one absorber density is required")

    inferred_ngroup = _infer_transport_ngroup(
        cross_sections_cgs_cm2,
        boundary_flux,
        source_photon_rate,
    )
    ngroup = edge_ngroup or inferred_ngroup or 1
    if edge_ngroup is not None and inferred_ngroup is not None and edge_ngroup != inferred_ngroup:
        raise ValueError(
            "radiation_group_edges_eV and group rate arrays disagree "
            f"({edge_ngroup} != {inferred_ngroup})",
        )

    boundary_flux = _normalize_group_values(
        boundary_flux,
        ngroup,
        "boundary_flux",
        PHOTON_FLUX_UNIT,
    )
    source_photon_rate = _normalize_group_values(
        source_photon_rate,
        ngroup,
        "source_photon_rate",
        PHOTON_RATE_UNIT,
    )
    optical_depth = _build_group_optical_depth(
        geometry,
        absorber_densities,
        cross_sections_cgs_cm2,
        ngroup,
    )

    return _stack_group_results(
        _trace_transport_groups(
            geometry,
            coordsys,
            optical_depth,
            boundary_flux,
            source_photon_rate,
            direction,
            ngroup,
        ),
    )


def _infer_transport_ngroup(cross_sections, boundary_flux, source_photon_rate):
    for sigma in cross_sections.values():
        sigma_array = _as_cgs_array(sigma, CGS_AREA_UNIT)
        if sigma_array.ndim > 0:
            return sigma_array.size
    for value, unit in (
        (boundary_flux, PHOTON_FLUX_UNIT),
        (source_photon_rate, PHOTON_RATE_UNIT),
    ):
        value_array = _as_cgs_array(value, unit)
        if value_array.ndim > 0:
            return value_array.size
    return None


def _trace_transport_groups(
    geometry,
    coordsys,
    optical_depth,
    boundary_flux,
    source_photon_rate,
    direction,
    ngroup,
):
    group_results = []
    for group in range(ngroup):
        if coordsys == "cartesian":
            result = _trace_cartesian(
                geometry,
                optical_depth[group],
                boundary_flux[group],
                direction,
            )
        else:
            result = _trace_spherical(
                geometry,
                optical_depth[group],
                boundary_flux[group],
                source_photon_rate[group],
                direction,
            )
        group_results.append(result)
    return group_results


def _state_mesh_for_radiative_transfer(state, par):
    """Build a minimal mesh view for the RT helper."""
    boundary = _plain_cgs_geometry("boundary_cgs_cm", state["boundary_cgs_cm"])
    if boundary.size < 2:  # noqa: PLR2004
        raise ValueError("radiative transfer requires at least two cell faces")
    volumes = _plain_cgs_geometry("volume_cgs_cm3", state["volume_cgs_cm3"])
    widths = state.get("width_cgs_cm")
    if widths is None:
        widths = np.absolute(np.diff(boundary))
    areas = state.get("face_area_cgs_cm2")
    return SimpleNamespace(
        coordsys=getattr(par, "coordsys", "spherical"),
        boundary_cgs_cm=boundary,
        width_cgs_cm=_plain_cgs_geometry("width_cgs_cm", widths),
        volume_cgs_cm3=volumes,
        face_area_cgs_cm2=(
            None if areas is None else _plain_cgs_geometry("face_area_cgs_cm2", areas)
        ),
    )


def trace_photon_density(state, par):
    """Trace photons through the selected radiative-transfer implementation."""
    if not getattr(par, "radiative_transfer", False):
        return np.asarray(state.get("ngamma_cgs_cm3", 0.0), dtype=float)
    code = _code_units(par)
    mesh = _state_mesh_for_radiative_transfer(state, par)
    rho_proper_cgs_g_cm3 = np.asarray(state["rho_cgs_g_cm3"], dtype=float)
    xHI_dimensionless = np.asarray(state["xHI"], dtype=float)
    group_edges_eV = getattr(par, "radiation_group_edges_eV", None)
    if group_edges_eV is not None:
        return _trace_grouped_photon_density(
            state,
            par,
            code,
            mesh,
            rho_proper_cgs_g_cm3,
            xHI_dimensionless,
            group_edges_eV,
        )
    sigma_gamma_cgs_cm2 = _quantity_or_code_to_cgs(
        getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA_CGS_CM2),
        code,
        CGS_AREA_UNIT,
        "area_cgs_cm2",
    )
    boundary_flux = _quantity_or_code_to_cgs(
        _parameter_value(par, "radiative_transfer_boundary_flux", 0.0),
        code,
        PHOTON_FLUX_UNIT,
        "photon_flux_per_cgs_cm2_s",
    )
    source_photon_rate = _quantity_or_code_to_cgs(
        _parameter_value(par, "source_photon_rate", 0.0),
        code,
        PHOTON_RATE_UNIT,
        "photon_rate_per_s",
    )
    result = trace_long_characteristics(
        mesh,
        absorber_densities={
            "HI": (
                getattr(par, "hydrogen_mass_fraction", 1.0)
                * rho_proper_cgs_g_cm3
                / PROTON_MASS_CGS
                * np.clip(xHI_dimensionless, 0.0, 1.0)
            ),
        },
        cross_sections_cgs_cm2={"HI": sigma_gamma_cgs_cm2},
        boundary_flux=boundary_flux,
        source_photon_rate=source_photon_rate,
        direction=_parameter_value(par, "radiative_transfer_direction", 1),
        coordsys=getattr(par, "coordsys", "spherical"),
    )
    return np.asarray(result.cell_photon_density, dtype=float)


def _trace_grouped_photon_density(
    state,
    par,
    code,
    mesh,
    rho_proper_cgs_g_cm3,
    xHI_dimensionless,
    group_edges_eV,
):
    sigma_groups = getattr(par, "radiation_group_sigma_gamma", None)
    if sigma_groups is None:
        sigma_groups = getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA_CGS_CM2)
    boundary_groups = getattr(
        par,
        "radiative_transfer_boundary_flux_groups",
        _parameter_value(par, "radiative_transfer_boundary_flux", 0.0),
    )
    source_groups = getattr(
        par,
        "source_photon_rate_groups",
        _parameter_value(par, "source_photon_rate", 0.0),
    )
    sigma_groups = _group_cgs_value(sigma_groups, code, CGS_AREA_UNIT, "area_cgs_cm2")
    boundary_groups = _group_cgs_value(
        boundary_groups,
        code,
        PHOTON_FLUX_UNIT,
        "photon_flux_per_cgs_cm2_s",
    )
    source_groups = _group_cgs_value(
        source_groups,
        code,
        PHOTON_RATE_UNIT,
        "photon_rate_per_s",
    )
    if hasattr(state, "get") and "xHeI" in state:
        nH = getattr(par, "hydrogen_mass_fraction", 0.7) * rho_proper_cgs_g_cm3 / PROTON_MASS_CGS
        nHe = (
            getattr(par, "helium_mass_fraction", 0.28)
            * rho_proper_cgs_g_cm3
            / (4.0 * PROTON_MASS_CGS)
        )
        absorbers = {
            "HI": nH * xHI_dimensionless,
            "HeI": nHe * state["xHeI"],
            "HeII": nHe * state["xHeII"],
        }
        cross_sections = {
            "HI": sigma_groups,
            "HeI": getattr(par, "radiation_group_sigma_gamma_HeI", sigma_groups),
            "HeII": getattr(par, "radiation_group_sigma_gamma_HeII", sigma_groups),
        }
    else:
        absorbers = {
            "HI": getattr(par, "hydrogen_mass_fraction", 1.0)
            * rho_proper_cgs_g_cm3
            / PROTON_MASS_CGS
            * np.clip(xHI_dimensionless, 0.0, 1.0),
        }
        cross_sections = {"HI": sigma_groups}
    result = trace_long_characteristics(
        mesh,
        absorber_densities=absorbers,
        cross_sections_cgs_cm2=cross_sections,
        boundary_flux=boundary_groups,
        source_photon_rate=source_groups,
        direction=_parameter_value(par, "radiative_transfer_direction", 1),
        coordsys=getattr(par, "coordsys", "spherical"),
        group_edges_eV=group_edges_eV,
    )
    return np.asarray(result.cell_photon_density, dtype=float)


def _group_cgs_value(value, code, target_unit, scale_key):
    if hasattr(value, "to_value"):
        return value.to_value(target_unit)
    if code is not None:
        return code_quantity_to_cgs(value, code, scale_key)
    return value
