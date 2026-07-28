"""Optional one-dimensional long-characteristic radiative transfer."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import unyt

import radhydropy.chemistry_species.hydrogen as rh


PHOTON_FLUX_UNIT = 1.0 / (unyt.cm**2 * unyt.s)
PHOTON_RATE_UNIT = 1.0 / unyt.s
PHOTON_DENSITY_UNIT = 1.0 / unyt.cm**3
PHOTON_ABSORPTION_RATE_UNIT = 1.0 / (unyt.cm**3 * unyt.s)


@dataclass
class LongCharacteristicResult:
    """Photon field returned by a one-dimensional long-characteristic trace."""

    optical_depth: np.ndarray
    face_photon_flux: unyt.unyt_array
    face_photon_rate: unyt.unyt_array
    cell_photon_flux: unyt.unyt_array
    cell_photon_density: unyt.unyt_array
    absorbed_photon_rate: unyt.unyt_array


def _as_photon_flux(value):
    if hasattr(value, "to"):
        return value.to(PHOTON_FLUX_UNIT)
    return np.asarray(value, dtype=float) * PHOTON_FLUX_UNIT


def _as_photon_rate(value):
    if hasattr(value, "to"):
        return value.to(PHOTON_RATE_UNIT)
    return np.asarray(value, dtype=float) * PHOTON_RATE_UNIT


def _safe_exp_neg(tau):
    tau = np.asarray(tau, dtype=float)
    return np.exp(-np.clip(tau, 0.0, 700.0))


def _attenuation_mean(tau):
    """Return ``(1 - exp(-tau)) / tau`` with the small-tau limit."""

    tau = np.asarray(tau, dtype=float)
    mean = np.ones_like(tau, dtype=float)
    valid = np.absolute(tau) > 1.0e-10
    mean[valid] = -np.expm1(-tau[valid]) / tau[valid]
    return mean


def _cell_widths(mesh):
    return np.absolute(mesh.boundary[1:] - mesh.boundary[:-1]).to(unyt.cm)


def _cell_volumes(mesh, coordsys):
    if hasattr(mesh, "vol"):
        return mesh.vol.to(unyt.cm**3)
    boundary = mesh.boundary.to(unyt.cm)
    if coordsys == "spherical":
        return (
            np.absolute(boundary[1:]**3 - boundary[:-1]**3)
            * 4.0
            * np.pi
            / 3.0
        ).to(unyt.cm**3)
    return _cell_widths(mesh) * (1.0 * unyt.cm**2)


def _face_areas(mesh, coordsys):
    boundary = mesh.boundary.to(unyt.cm)
    if coordsys == "spherical":
        return (4.0 * np.pi * boundary**2).to(unyt.cm**2)
    if hasattr(mesh, "area") and mesh.area is not None:
        area = mesh.area.to(unyt.cm**2)
        if len(area) == len(boundary):
            return area
        if len(area) == len(boundary) - 1:
            return np.ones(len(boundary)) * area[0]
    return np.ones(len(boundary)) * unyt.cm**2


def _face_flux_from_rate(face_rate, face_area):
    flux = np.zeros(len(face_rate)) * PHOTON_FLUX_UNIT
    valid = face_area > 0.0 * face_area.units
    flux[valid] = (face_rate[valid] / face_area[valid]).to(PHOTON_FLUX_UNIT)
    return flux


def _optical_depth(mesh, rho, xHI, hydrogen_mass_fraction, sigma_gamma):
    nH = rh.hydrogen_number_density(rho, hydrogen_mass_fraction)
    xHI = rh.clip_neutral_fraction(xHI)
    sigma = rh.photon_cross_section(sigma_gamma)
    tau = (sigma * nH * xHI * _cell_widths(mesh)).to_value(unyt.dimensionless)
    return np.maximum(tau, 0.0)


def _trace_cartesian(mesh, optical_depth, boundary_flux, direction):
    ncell = len(optical_depth)
    face_area = _face_areas(mesh, "cartesian")
    volumes = _cell_volumes(mesh, "cartesian")
    face_flux = np.zeros(ncell + 1) * PHOTON_FLUX_UNIT
    cell_flux = np.zeros(ncell) * PHOTON_FLUX_UNIT
    absorbed_rate = np.zeros(ncell) * PHOTON_ABSORPTION_RATE_UNIT

    attenuation = _safe_exp_neg(optical_depth)
    mean_attenuation = _attenuation_mean(optical_depth)
    boundary_flux = _as_photon_flux(boundary_flux)

    if direction >= 0:
        face_flux[0] = boundary_flux
        for i in range(ncell):
            face_flux[i + 1] = face_flux[i] * attenuation[i]
            cell_flux[i] = face_flux[i] * mean_attenuation[i]
            absorbed = face_flux[i] * face_area[i] - face_flux[i + 1] * face_area[i + 1]
            absorbed_rate[i] = (absorbed / volumes[i]).to(PHOTON_ABSORPTION_RATE_UNIT)
    else:
        face_flux[-1] = boundary_flux
        for i in range(ncell - 1, -1, -1):
            face_flux[i] = face_flux[i + 1] * attenuation[i]
            cell_flux[i] = face_flux[i + 1] * mean_attenuation[i]
            absorbed = face_flux[i + 1] * face_area[i + 1] - face_flux[i] * face_area[i]
            absorbed_rate[i] = (absorbed / volumes[i]).to(PHOTON_ABSORPTION_RATE_UNIT)

    face_rate = (face_flux * face_area).to(PHOTON_RATE_UNIT)
    cell_density = (cell_flux / rh.SPEED_OF_LIGHT).to(PHOTON_DENSITY_UNIT)
    return LongCharacteristicResult(
        optical_depth=optical_depth,
        face_photon_flux=face_flux,
        face_photon_rate=face_rate,
        cell_photon_flux=cell_flux,
        cell_photon_density=cell_density,
        absorbed_photon_rate=absorbed_rate,
    )


def _spherical_boundary_rate(face_area, boundary_flux, source_photon_rate, direction):
    source_rate = _as_photon_rate(source_photon_rate)
    if np.any(np.asarray(source_rate.value) != 0.0):
        return source_rate
    boundary_flux = _as_photon_flux(boundary_flux)
    boundary_area = face_area[0] if direction >= 0 else face_area[-1]
    return (boundary_flux * boundary_area).to(PHOTON_RATE_UNIT)


def _trace_spherical(
    mesh,
    optical_depth,
    boundary_flux,
    source_photon_rate,
    direction,
):
    ncell = len(optical_depth)
    face_area = _face_areas(mesh, "spherical")
    volumes = _cell_volumes(mesh, "spherical")
    widths = _cell_widths(mesh)
    face_rate = np.zeros(ncell + 1) * PHOTON_RATE_UNIT
    cell_density = np.zeros(ncell) * PHOTON_DENSITY_UNIT
    absorbed_rate = np.zeros(ncell) * PHOTON_ABSORPTION_RATE_UNIT

    attenuation = _safe_exp_neg(optical_depth)
    mean_attenuation = _attenuation_mean(optical_depth)
    incoming_rate = _spherical_boundary_rate(
        face_area,
        boundary_flux,
        source_photon_rate,
        direction,
    )

    if direction >= 0:
        face_rate[0] = incoming_rate
        for i in range(ncell):
            face_rate[i + 1] = face_rate[i] * attenuation[i]
            density = face_rate[i] * widths[i] * mean_attenuation[i]
            cell_density[i] = (density / volumes[i] / rh.SPEED_OF_LIGHT).to(
                PHOTON_DENSITY_UNIT
            )
            absorbed_rate[i] = ((face_rate[i] - face_rate[i + 1]) / volumes[i]).to(
                PHOTON_ABSORPTION_RATE_UNIT
            )
    else:
        face_rate[-1] = incoming_rate
        for i in range(ncell - 1, -1, -1):
            face_rate[i] = face_rate[i + 1] * attenuation[i]
            density = face_rate[i + 1] * widths[i] * mean_attenuation[i]
            cell_density[i] = (density / volumes[i] / rh.SPEED_OF_LIGHT).to(
                PHOTON_DENSITY_UNIT
            )
            absorbed_rate[i] = ((face_rate[i + 1] - face_rate[i]) / volumes[i]).to(
                PHOTON_ABSORPTION_RATE_UNIT
            )

    face_flux = _face_flux_from_rate(face_rate, face_area)
    cell_flux = (cell_density * rh.SPEED_OF_LIGHT).to(PHOTON_FLUX_UNIT)
    return LongCharacteristicResult(
        optical_depth=optical_depth,
        face_photon_flux=face_flux,
        face_photon_rate=face_rate,
        cell_photon_flux=cell_flux,
        cell_photon_density=cell_density,
        absorbed_photon_rate=absorbed_rate,
    )


def trace_long_characteristics(
    mesh,
    rho,
    xHI,
    hydrogen_mass_fraction=1.0,
    sigma_gamma=rh.DEFAULT_SIGMA_GAMMA,
    boundary_flux=0.0 * PHOTON_FLUX_UNIT,
    source_photon_rate=0.0 * PHOTON_RATE_UNIT,
    direction=1,
    coordsys=None,
):
    """Trace a 1D photon field through hydrogen opacity.

    The returned cell photon density is the finite-volume, cell-averaged value
    suitable for the hydrogen source terms, ``n_gamma = <F> / c``.
    """

    coordsys = coordsys or getattr(mesh, "coordsys", "cartesian")
    if coordsys not in ("cartesian", "spherical"):
        raise ValueError("coordsys unknown: %s" % coordsys)

    optical_depth = _optical_depth(
        mesh,
        rho,
        xHI,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        sigma_gamma=sigma_gamma,
    )
    if coordsys == "cartesian":
        return _trace_cartesian(mesh, optical_depth, boundary_flux, direction)
    return _trace_spherical(
        mesh,
        optical_depth,
        boundary_flux,
        source_photon_rate,
        direction,
    )


def _interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def _interior_mesh(mesh, interior):
    attrs = dict(
        coordsys=getattr(mesh, "coordsys", "cartesian"),
        boundary=mesh.boundary[interior.start : interior.stop + 1],
        vol=mesh.vol[interior],
    )
    if hasattr(mesh, "area"):
        attrs["area"] = mesh.area[interior]
    return SimpleNamespace(**attrs)


def apply_long_characteristics_to_fluid(mesh, fluid, par):
    """Update ``fluid.ngamma`` with a long-characteristic photon field."""

    if not getattr(par, "radiative_transfer", False):
        return None
    method = getattr(par, "radiative_transfer_method", "long_characteristics")
    if method != "long_characteristics":
        raise ValueError("radiative transfer method unknown: %s" % method)
    if not hasattr(fluid, "xHI"):
        raise AttributeError("xHI is required for long-characteristic opacity")
    if not hasattr(fluid, "ngamma"):
        fluid.ngamma = np.zeros(np.shape(fluid.rho), dtype=float) * PHOTON_DENSITY_UNIT

    interior = _interior_slice(par)
    result = trace_long_characteristics(
        _interior_mesh(mesh, interior),
        fluid.rho[interior],
        fluid.xHI[interior],
        hydrogen_mass_fraction=getattr(par, "hydrogen_mass_fraction", 1.0),
        sigma_gamma=getattr(par, "hydrogen_sigma_gamma", rh.DEFAULT_SIGMA_GAMMA),
        boundary_flux=getattr(
            par,
            "radiative_transfer_boundary_flux",
            0.0 * PHOTON_FLUX_UNIT,
        ),
        source_photon_rate=getattr(
            par,
            "radiative_transfer_source_photon_rate",
            0.0 * PHOTON_RATE_UNIT,
        ),
        direction=getattr(par, "radiative_transfer_direction", 1),
        coordsys=getattr(mesh, "coordsys", "cartesian"),
    )
    fluid.ngamma[interior] = result.cell_photon_density.to(fluid.ngamma.units)
    return result
