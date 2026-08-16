"""Optional one-dimensional long-characteristic radiative transfer."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from radhydropy.constants import DEFAULT_SIGMA_GAMMA, PROTON_MASS_CGS, SPEED_OF_LIGHT_CGS
import radhydropy.chemistry_species.hydrogen as rh
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

    boundary_cm: np.ndarray
    width_cm: np.ndarray
    volume_cm3: np.ndarray
    face_area_cm2: np.ndarray
    coordsys: str


@dataclass
class CausalCellResult:
    """Transport result for one causally ordered cell."""

    outgoing_rate: np.ndarray
    absorbed_rate: np.ndarray
    photon_density: np.ndarray
    attenuation: np.ndarray


def _safe_exp_neg(tau):
    tau = np.asarray(tau, dtype=float)
    return np.exp(-np.clip(tau, 0.0, 700.0))


def species_photoionization_rates(ngamma, sigma_by_species):
    """Return photoionization and photoheating rates for each absorber."""
    ngamma = np.asarray(ngamma, dtype=float)
    rates = {}
    for species, sigma in sigma_by_species.items():
        sigma = np.asarray(sigma, dtype=float)
        if ngamma.ndim == 2 and sigma.ndim == 1:
            sigma = sigma[:, None]
        rate = SPEED_OF_LIGHT_CGS * sigma * ngamma
        rates[species] = np.sum(rate, axis=0) if rate.ndim > 1 else rate
    return rates


def species_photoionization_heating(ngamma, sigma_by_species, epsilon_by_species):
    rates = {}
    for species, sigma in sigma_by_species.items():
        epsilon = np.asarray(epsilon_by_species.get(species, 0.0), dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        if ngamma.ndim == 2:
            sigma = sigma[:, None] if sigma.ndim == 1 else sigma
            epsilon = epsilon[:, None] if epsilon.ndim == 1 else epsilon
        value = SPEED_OF_LIGHT_CGS * sigma * epsilon * np.asarray(ngamma, dtype=float)
        rates[species] = np.sum(value, axis=0) if value.ndim > 1 else value
    return rates


def _attenuation_mean(tau):
    """Return ``(1 - exp(-tau)) / tau`` with the small-tau limit."""

    tau = np.asarray(tau, dtype=float)
    mean = np.ones_like(tau, dtype=float)
    valid = np.absolute(tau) > 1.0e-10
    mean[valid] = -np.expm1(-tau[valid]) / tau[valid]
    return mean


def _quantity_or_code_to_cgs(value, code_units, cgs_unit, scale_key):
    if hasattr(value, "to_value"):
        return _as_cgs_float(value, cgs_unit)
    if code_units is not None:
        return code_quantity_to_cgs(value, code_units, scale_key)
    return np.asarray(value, dtype=float)


def _as_cgs_array(value, unit):
    """Return a scalar or array-like value as a plain CGS array."""
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(value, dtype=float)


def _mesh_boundary_cm(mesh):
    return np.asarray(mesh.boundary, dtype=float)


def _cell_widths_cm(mesh):
    boundary_cm = _mesh_boundary_cm(mesh)
    return np.absolute(boundary_cm[1:] - boundary_cm[:-1])


def _cell_volumes_cm3(mesh, coordsys):
    if hasattr(mesh, "vol"):
        return np.asarray(mesh.vol, dtype=float)
    boundary = _mesh_boundary_cm(mesh)
    if coordsys == "spherical":
        return np.absolute(boundary[1:] ** 3 - boundary[:-1] ** 3) * 4.0 * np.pi / 3.0
    return _cell_widths_cm(mesh)


def _face_areas_cm2(mesh, coordsys):
    boundary = _mesh_boundary_cm(mesh)
    if coordsys == "spherical":
        return 4.0 * np.pi * boundary**2
    if hasattr(mesh, "area") and mesh.area is not None:
        area = np.asarray(mesh.area, dtype=float)
        if len(area) == len(boundary):
            return area
        if len(area) == len(boundary) - 1:
            return np.ones(len(boundary)) * area[0]
    return np.ones(len(boundary))


def build_transport_geometry(mesh, coordsys=None):
    """Return normalized geometry for one-dimensional radiation transport."""
    coordsys = coordsys or getattr(mesh, "coordsys", "cartesian")
    if coordsys not in ("cartesian", "spherical"):
        raise ValueError("coordsys unknown: %s" % coordsys)
    boundary = _mesh_boundary_cm(mesh)
    return TransportGeometry(
        boundary_cm=boundary,
        width_cm=_cell_widths_cm(mesh),
        volume_cm3=_cell_volumes_cm3(mesh, coordsys),
        face_area_cm2=_face_areas_cm2(mesh, coordsys),
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
        width = geometry.width_cm[cell_index]
        volume = geometry.volume_cm3[cell_index]
        if abs(tau) > 1.0e-10:
            attenuation_mean = float(-np.expm1(-tau) / tau)
        else:
            attenuation_mean = 1.0
        if geometry.coordsys == "spherical":
            photon_density = (
                incoming
                * width
                * attenuation_mean
                / volume
                / SPEED_OF_LIGHT_CGS
            )
        else:
            face_index = cell_index if direction >= 0 else cell_index + 1
            area = geometry.face_area_cm2[face_index]
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
    width = geometry.width_cm[cell_index]
    volume = geometry.volume_cm3[cell_index]
    if geometry.coordsys == "spherical":
        photon_density = (
            incoming_rate
            * width
            * _attenuation_mean(optical_depth)
            / volume
            / SPEED_OF_LIGHT_CGS
        )
    else:
        area = geometry.face_area_cm2[face_index]
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


def _face_flux_from_rate(face_rate, face_area_cm2):
    flux = np.zeros(len(face_rate), dtype=float)
    valid = face_area_cm2 > 0.0
    flux[valid] = face_rate[valid] / face_area_cm2[valid]
    return flux


def _trace_cartesian(mesh, optical_depth, boundary_flux, direction):
    ncell = len(optical_depth)
    face_area = _face_areas_cm2(mesh, "cartesian")
    volumes = _cell_volumes_cm3(mesh, "cartesian")
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
        absorbed_rate = (
            face_flux[:-1] * face_area[:-1] - face_flux[1:] * face_area[1:]
        ) / volumes
    else:
        face_flux = np.empty(ncell + 1, dtype=float)
        face_flux[-1] = boundary_flux
        if ncell > 0:
            face_flux[:-1] = boundary_flux * np.cumprod(attenuation[::-1])[::-1]
        cell_flux = face_flux[1:] * mean_attenuation
        absorbed_rate = (
            face_flux[1:] * face_area[1:] - face_flux[:-1] * face_area[:-1]
        ) / volumes

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
    face_area = _face_areas_cm2(mesh, "spherical")
    volumes = _cell_volumes_cm3(mesh, "spherical")
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
        cell_density = (
            incoming_rate
            * prefix
            * widths
            * mean_attenuation
            / volumes
            / speed_of_light
        )
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
            incoming_rate
            * suffix_cell
            * widths
            * mean_attenuation
            / volumes
            / speed_of_light
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


def _normalize_group_edges(group_edges_eV):
    """Validate group edges and return the number of photon groups."""
    if group_edges_eV is None:
        return None
    edges = _as_cgs_array(group_edges_eV, 1.0)
    if edges.ndim != 1 or edges.size < 2:
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
    cross_sections_cm2,
    ngroup,
):
    """Build optical depth per photon group from absorber densities."""
    widths = _cell_widths_cm(mesh)
    optical_depth = np.zeros((ngroup, widths.size), dtype=float)
    for species, density in absorber_densities.items():
        density = np.asarray(density, dtype=float)
        if density.shape != widths.shape:
            raise ValueError(
                f"absorber density for {species!r} must have shape {widths.shape}"
            )
        if species not in cross_sections_cm2:
            raise ValueError(f"missing cross section for absorber {species!r}")
        sigma = _normalize_group_values(
            cross_sections_cm2[species],
            ngroup,
            f"cross_sections_cm2[{species!r}]",
            CGS_AREA_UNIT,
        )
        optical_depth += sigma[:, None] * density[None, :] * widths[None, :]
    return np.maximum(optical_depth, 0.0)


def _stack_group_results(group_results, squeeze):
    """Stack single-group results, preserving the legacy one-group shape."""
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
        values.append(stacked[0] if squeeze else stacked)
    return LongCharacteristicResult(*values)


def trace_long_characteristics(
    mesh,
    rho=None,
    xHI=None,
    hydrogen_mass_fraction=1.0,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
    boundary_flux=0.0,
    source_photon_rate=0.0,
    direction=1,
    coordsys=None,
    group_edges_eV=None,
    absorber_densities=None,
    cross_sections_cm2=None,
):
    """Trace one or more photon groups through one-dimensional opacity.

    The legacy ``rho``/``xHI`` arguments describe a hydrogen-only absorber.
    General multi-group transport can instead provide ``absorber_densities``
    and ``cross_sections_cm2``. Densities are in ``cm**-3`` and cross sections
    are in ``cm**2``. Results have shape ``(ngroup, ncell)`` for multiple
    groups, and retain the legacy ``(ncell,)`` shape for one group.
    """

    coordsys = coordsys or getattr(mesh, "coordsys", "cartesian")
    if coordsys not in ("cartesian", "spherical"):
        raise ValueError("coordsys unknown: %s" % coordsys)

    edge_ngroup = _normalize_group_edges(group_edges_eV)

    if absorber_densities is None:
        if rho is None or xHI is None:
            raise ValueError("rho and xHI are required for hydrogen transport")
        rho_g_cm3 = np.asarray(rho, dtype=float)
        xHI = np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)
        absorber_densities = {
            "HI": (
                hydrogen_mass_fraction
                * rho_g_cm3
                / PROTON_MASS_CGS
                * xHI
            )
        }
        if cross_sections_cm2 is None:
            cross_sections_cm2 = {"HI": sigma_gamma}
    elif cross_sections_cm2 is None:
        raise ValueError(
            "cross_sections_cm2 is required with absorber_densities"
        )

    if not absorber_densities:
        raise ValueError("at least one absorber density is required")

    inferred_ngroup = None
    for sigma in cross_sections_cm2.values():
        sigma_array = _as_cgs_array(sigma, CGS_AREA_UNIT)
        if sigma_array.ndim > 0:
            inferred_ngroup = sigma_array.size
            break
    if inferred_ngroup is None:
        for value, unit in (
            (boundary_flux, PHOTON_FLUX_UNIT),
            (source_photon_rate, PHOTON_RATE_UNIT),
        ):
            value_array = _as_cgs_array(value, unit)
            if value_array.ndim > 0:
                inferred_ngroup = value_array.size
                break
    ngroup = edge_ngroup or inferred_ngroup or 1
    if edge_ngroup is not None and inferred_ngroup is not None:
        if edge_ngroup != inferred_ngroup:
            raise ValueError(
                "radiation_group_edges_eV and group rate arrays disagree "
                f"({edge_ngroup} != {inferred_ngroup})"
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
        mesh,
        absorber_densities,
        cross_sections_cm2,
        ngroup,
    )

    group_results = []
    for group in range(ngroup):
        if coordsys == "cartesian":
            result = _trace_cartesian(
                mesh,
                optical_depth[group],
                boundary_flux[group],
                direction,
            )
        else:
            result = _trace_spherical(
                mesh,
                optical_depth[group],
                boundary_flux[group],
                source_photon_rate[group],
                direction,
            )
        group_results.append(result)
    return _stack_group_results(group_results, squeeze=ngroup == 1)


def _state_mesh_for_radiative_transfer(state, par):
    """Build a minimal mesh view for the RT helper."""
    boundary = np.asarray(state["boundary_cm"], dtype=float)
    if boundary.size < 2:
        raise ValueError("radiative transfer requires at least two cell faces")
    volumes = np.asarray(state["volume_cm3"], dtype=float)
    return SimpleNamespace(
        coordsys=getattr(par, "coordsys", "spherical"),
        boundary=boundary,
        vol=volumes,
    )


def _state_fluid_for_radiative_transfer(state, par):
    """Build a minimal fluid view for the RT helper."""
    rho = np.asarray(state["rho_g_cm3"], dtype=float)
    xHI = np.asarray(state["xHI"], dtype=float)
    ngamma = np.asarray(state.get("ngamma_cm3", np.zeros_like(rho)), dtype=float)
    return SimpleNamespace(
        rho=rho,
        xHI=xHI,
        ngamma=ngamma,
    )


def trace_photon_density(state, par):
    """Trace photons through the selected radiative-transfer implementation."""
    if not getattr(par, "radiative_transfer", False):
        return np.asarray(state.get("ngamma_cm3", 0.0), dtype=float)
    code = _code_units(par)
    mesh = _state_mesh_for_radiative_transfer(state, par)
    fluid = _state_fluid_for_radiative_transfer(state, par)
    group_edges_eV = getattr(par, "radiation_group_edges_eV", None)
    if group_edges_eV is not None:
        sigma_groups = getattr(par, "radiation_group_sigma_gamma", None)
        if sigma_groups is None:
            sigma_groups = getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA)
        boundary_groups = getattr(
            par,
            "radiative_transfer_boundary_flux_groups",
            getattr(par, "radiative_transfer_boundary_flux", 0.0),
        )
        source_groups = getattr(
            par,
            "radiative_transfer_source_photon_rate_groups",
            getattr(par, "radiative_transfer_source_photon_rate", 0.0),
        )
        if hasattr(sigma_groups, "to_value"):
            sigma_groups = sigma_groups.to_value(CGS_AREA_UNIT)
        elif code is not None:
            sigma_groups = code_quantity_to_cgs(
                sigma_groups,
                code,
                "area_cm2",
            )
        if hasattr(boundary_groups, "to_value"):
            boundary_groups = boundary_groups.to_value(PHOTON_FLUX_UNIT)
        elif code is not None:
            boundary_groups = code_quantity_to_cgs(
                boundary_groups,
                code,
                "photon_flux_per_cm2_s",
            )
        if hasattr(source_groups, "to_value"):
            source_groups = source_groups.to_value(PHOTON_RATE_UNIT)
        elif code is not None:
            source_groups = code_quantity_to_cgs(
                source_groups,
                code,
                "photon_rate_per_s",
            )
        if hasattr(state, "get") and "xHeI" in state:
            rho_cgs = np.asarray(fluid.rho, dtype=float)
            nH = getattr(par, "hydrogen_mass_fraction", 0.7) * rho_cgs / PROTON_MASS_CGS
            nHe = getattr(par, "helium_mass_fraction", 0.28) * rho_cgs / (4.0 * PROTON_MASS_CGS)
            absorbers = {"HI": nH * fluid.xHI, "HeI": nHe * state["xHeI"], "HeII": nHe * state["xHeII"]}
            cross_sections = {"HI": sigma_groups, "HeI": getattr(par, "radiation_group_sigma_gamma_HeI", sigma_groups), "HeII": getattr(par, "radiation_group_sigma_gamma_HeII", sigma_groups)}
            return np.asarray(trace_long_characteristics(mesh, absorber_densities=absorbers, cross_sections_cm2=cross_sections, boundary_flux=boundary_groups, source_photon_rate=source_groups, direction=getattr(par, "radiative_transfer_direction", 1), coordsys=getattr(par, "coordsys", "spherical"), group_edges_eV=group_edges_eV).cell_photon_density, dtype=float)
        result = trace_long_characteristics(
            mesh,
            fluid.rho,
            fluid.xHI,
            hydrogen_mass_fraction=getattr(par, "hydrogen_mass_fraction", 1.0),
            sigma_gamma=sigma_groups,
            boundary_flux=boundary_groups,
            source_photon_rate=source_groups,
            direction=getattr(par, "radiative_transfer_direction", 1),
            coordsys=getattr(par, "coordsys", "spherical"),
            group_edges_eV=group_edges_eV,
        )
        return np.asarray(result.cell_photon_density, dtype=float)
    sigma_gamma_cm2 = _quantity_or_code_to_cgs(
        getattr(par, "hydrogen_sigma_gamma", DEFAULT_SIGMA_GAMMA),
        code,
        CGS_AREA_UNIT,
        "area_cm2",
    )
    boundary_flux = _quantity_or_code_to_cgs(
        getattr(par, "radiative_transfer_boundary_flux", 0.0),
        code,
        PHOTON_FLUX_UNIT,
        "photon_flux_per_cm2_s",
    )
    source_photon_rate = _quantity_or_code_to_cgs(
        getattr(par, "radiative_transfer_source_photon_rate", 0.0),
        code,
        PHOTON_RATE_UNIT,
        "photon_rate_per_s",
    )
    result = trace_long_characteristics(
        mesh,
        fluid.rho,
        fluid.xHI,
        hydrogen_mass_fraction=getattr(par, "hydrogen_mass_fraction", 1.0),
        sigma_gamma=sigma_gamma_cm2,
        boundary_flux=boundary_flux,
        source_photon_rate=source_photon_rate,
        direction=getattr(par, "radiative_transfer_direction", 1),
        coordsys=getattr(par, "coordsys", "spherical"),
    )
    return np.asarray(result.cell_photon_density, dtype=float)
