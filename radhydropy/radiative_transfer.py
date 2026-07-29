"""Optional one-dimensional long-characteristic radiative transfer."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import unyt

import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.thermo_networks.hydrogen as rth
from radhydropy.units import (
    PHOTON_ABSORPTION_RATE_UNIT,
    PHOTON_DENSITY_UNIT,
    PHOTON_FLUX_UNIT,
    PHOTON_RATE_UNIT,
    _as_photon_flux,
    _as_photon_rate,
    _optional_photon_quantity,
    photon_number_density,
)


@dataclass
class LongCharacteristicResult:
    """Photon field returned by a one-dimensional long-characteristic trace."""

    optical_depth: np.ndarray
    face_photon_flux: unyt.unyt_array
    face_photon_rate: unyt.unyt_array
    cell_photon_flux: unyt.unyt_array
    cell_photon_density: unyt.unyt_array
    absorbed_photon_rate: unyt.unyt_array


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

    optical_depth = rth.trace_spherical_tau(
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


def _state_mesh_for_radiative_transfer(state, par):
    """Build a minimal mesh view for the RT helper."""
    noghost = int(getattr(par, "noghost", 0))
    boundary = np.asarray(state["boundary_cm"], dtype=float)
    if boundary.size < 2:
        raise ValueError("radiative transfer requires at least two cell faces")
    widths = np.asarray(state["width_cm"], dtype=float)
    if noghost > 0:
        left = boundary[0] - np.arange(noghost, 0, -1, dtype=float) * float(widths[0])
        right = boundary[-1] + np.arange(1, noghost + 1, dtype=float) * float(widths[-1])
        boundary = np.concatenate((left, boundary, right))
    full_boundary = boundary * unyt.cm
    volumes = np.asarray(state["volume_cm3"], dtype=float)
    if noghost > 0:
        volumes = np.concatenate(
            (
                np.full(noghost, volumes[0], dtype=float),
                volumes,
                np.full(noghost, volumes[-1], dtype=float),
            )
        )
    return SimpleNamespace(
        coordsys=getattr(par, "coordsys", "spherical"),
        boundary=full_boundary,
        vol=volumes * (unyt.cm**3),
    )


def _state_fluid_for_radiative_transfer(state, par):
    """Build a minimal fluid view for the RT helper."""
    noghost = int(getattr(par, "noghost", 0))
    rho = np.asarray(state["rho_g_cm3"], dtype=float)
    xHI = np.asarray(state["xHI"], dtype=float)
    ngamma = np.asarray(state.get("ngamma_cm3", np.zeros_like(rho)), dtype=float)
    if noghost > 0:
        rho = np.concatenate((np.full(noghost, rho[0]), rho, np.full(noghost, rho[-1])))
        xHI = np.concatenate((np.full(noghost, xHI[0]), xHI, np.full(noghost, xHI[-1])))
        ngamma = np.concatenate(
            (np.zeros(noghost, dtype=float), ngamma, np.zeros(noghost, dtype=float))
        )
    return SimpleNamespace(
        rho=rho * (unyt.g / unyt.cm**3),
        xHI=xHI,
        ngamma=ngamma * (1.0 / unyt.cm**3),
    )


def trace_photon_density(state, par):
    """Trace photons through the selected radiative-transfer implementation."""
    if not getattr(par, "radiative_transfer", False):
        return np.asarray(state.get("ngamma_cm3", 0.0), dtype=float)
    mesh = _state_mesh_for_radiative_transfer(state, par)
    fluid = _state_fluid_for_radiative_transfer(state, par)
    result = trace_long_characteristics(
        mesh,
        fluid.rho,
        fluid.xHI,
        hydrogen_mass_fraction=getattr(par, "hydrogen_mass_fraction", 1.0),
        sigma_gamma=_optional_photon_quantity(
            getattr(par, "hydrogen_sigma_gamma", None),
            rh.DEFAULT_SIGMA_GAMMA,
            unyt.cm**2,
        ),
        boundary_flux=_as_photon_flux(getattr(par, "radiative_transfer_boundary_flux", None)),
        source_photon_rate=_as_photon_rate(
            getattr(par, "radiative_transfer_source_photon_rate", None)
        ),
        direction=getattr(par, "radiative_transfer_direction", 1),
        coordsys=getattr(par, "coordsys", "spherical"),
    )
    interior = slice(
        int(getattr(par, "noghost", 0)),
        int(getattr(par, "noghost", 0)) + int(getattr(par, "nogrid", len(state["xHI"]))),
    )
    return np.asarray(result.cell_photon_density[interior].to_value(1.0 / unyt.cm**3), dtype=float)
