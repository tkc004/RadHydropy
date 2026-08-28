"""Initial conditions and diagnostics for the cosmological virial-shock test."""

from types import SimpleNamespace
from math import erf
from pathlib import Path
import importlib.util

import numpy as np
import unyt

from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import quantity_to_value

_CANONICAL_TOOLS = Path(__file__).resolve().parents[2] / "tools" / "lcdm_correlation.py"
_TOOLS_SPEC = importlib.util.spec_from_file_location(
    "radhydropy_lcdm_correlation", _CANONICAL_TOOLS
)
_LCDM_TOOLS = importlib.util.module_from_spec(_TOOLS_SPEC)
_TOOLS_SPEC.loader.exec_module(_LCDM_TOOLS)

# Keep the example's historical ``tools as et`` interface while maintaining a
# single implementation in the repository-level tools package.
eisenstein_hu_nowiggle_transfer = _LCDM_TOOLS.eisenstein_hu_nowiggle_transfer
linear_matter_power_spectrum = _LCDM_TOOLS.linear_matter_power_spectrum
linear_matter_power_spectrum_shape = _LCDM_TOOLS.linear_matter_power_spectrum_shape
linear_correlation_from_power_spectrum = _LCDM_TOOLS.linear_correlation_from_power_spectrum
generate_lcdm_correlation_table = _LCDM_TOOLS.generate_lcdm_correlation_table
load_lcdm_correlation_table = _LCDM_TOOLS.load_lcdm_correlation_table


def cell_centres(boundary):
    inner, outer = boundary[:-1], boundary[1:]
    return 0.75 * (outer**4 - inner**4) / np.maximum(outer**3 - inner**3, 1.0e-300)


def perturbation_radius(ic, cosmology):
    """Return the comoving top-hat radius for the requested halo mass."""
    if ic.get("target_halo_mass") is None:
        return float(ic["perturbation_radius"])
    t = float(ic["initial_cosmic_time"])
    a = float(cosmology.scale_factor(t))
    rho_comoving = float(cosmology.background_density(t)) * a**3
    overdensity = float(ic["initial_overdensity"])
    return float(
        (
            float(ic["target_halo_mass"])
            / ((4.0 * np.pi / 3.0) * rho_comoving * (1.0 + overdensity))
        )
        ** (1.0 / 3.0)
    )


def _gaussian_correlation_mean(radius, correlation_length):
    """Mean enclosed Gaussian correlation shape, normalized to xi(0)=1."""
    radius = np.asarray(radius, dtype=float)
    x = radius / max(float(correlation_length), 1.0e-30)
    erf_x = np.vectorize(erf, otypes=[float])(x)
    integral = np.sqrt(np.pi) / 4.0 * erf_x - 0.5 * x * np.exp(-x**2)
    result = np.divide(3.0 * integral, np.maximum(x**3, 1.0e-30))
    result = np.asarray(result, dtype=float)
    result[x < 1.0e-4] = 1.0
    return result


def _correlation_profile(radius, table, length_unit_mpc_h):
    """Interpolate xi and its enclosed mean in simulation length units."""
    radius = np.asarray(radius, dtype=float)
    table_radius = np.asarray(table["radius_mpc_h"], dtype=float)
    table_correlation = np.asarray(table["correlation"], dtype=float)
    if table_radius.ndim != 1 or table_correlation.ndim != 1:
        raise ValueError("linear correlation table arrays must be one-dimensional")
    if table_radius.size != table_correlation.size or table_radius.size < 2:
        raise ValueError("linear correlation table arrays have incompatible sizes")
    if np.any(np.diff(table_radius) <= 0.0):
        raise ValueError("linear correlation table radii must be increasing")
    radius_mpc_h = radius * float(length_unit_mpc_h)
    if np.any(radius_mpc_h > table_radius[-1]):
        raise ValueError("initial-condition radius exceeds the correlation table")

    # Include the origin using the first tabulated value, then integrate
    # xi(r) r^2 dr once so arbitrary shell radii get a consistent enclosed
    # mean correlation.
    integration_radius = np.concatenate(([0.0], table_radius))
    integration_xi = np.concatenate(([table_correlation[0]], table_correlation))
    cumulative = np.concatenate((
        [0.0],
        np.cumsum(
            0.5 * (integration_xi[1:] * integration_radius[1:]**2
                   + integration_xi[:-1] * integration_radius[:-1]**2)
            * np.diff(integration_radius)
        ),
    ))
    xi = np.interp(
        radius_mpc_h, table_radius, table_correlation,
        left=table_correlation[0], right=table_correlation[-1],
    )
    enclosed_integral = np.interp(
        radius_mpc_h, integration_radius, cumulative,
        left=0.0, right=cumulative[-1],
    )
    mean_xi = np.where(
        radius_mpc_h <= table_radius[0],
        table_correlation[0],
        3.0 * enclosed_integral / np.maximum(radius_mpc_h**3, 1.0e-300),
    )
    return xi, mean_xi


def density_contrast_profile(
    radius, ic, cosmology, correlation_table=None, length_unit_mpc_h=1.0
):
    """Return ``(delta, mean_delta)`` for the configured growing mode.

    ``linear_correlation`` uses the supplied tabulated linear-theory
    correlation function.  Its amplitude is fixed by the requested mean
    overdensity inside the target Lagrangian radius.
    """
    radius = np.asarray(radius, dtype=float)
    target_radius = perturbation_radius(ic, cosmology)
    overdensity = float(ic["initial_overdensity"])
    profile = str(ic.get("initial_density_profile", "top_hat")).lower()
    if profile == "top_hat":
        inside = radius < target_radius
        delta = overdensity * inside
        mean_delta = overdensity * np.where(
            inside, 1.0, (target_radius / np.maximum(radius, 1.0e-30)) ** 3
        )
        return np.asarray(delta, dtype=float), np.asarray(mean_delta, dtype=float)
    if profile not in ("linear_correlation", "gaussian_correlation"):
        raise ValueError("unknown initial_density_profile %r" % profile)

    if profile == "linear_correlation":
        if correlation_table is None:
            raise ValueError(
                "linear_correlation requires a tabulated correlation table"
            )
        xi, mean_xi = _correlation_profile(
            radius, correlation_table, length_unit_mpc_h
        )
        target_mean_xi = float(
            _correlation_profile(
                np.array([target_radius]), correlation_table,
                length_unit_mpc_h,
            )[1][0]
        )
    else:
        correlation_length = float(
            ic.get("correlation_length", 0.5 * target_radius)
        )
        xi = np.exp(-(radius / max(correlation_length, 1.0e-30)) ** 2)
        mean_xi = _gaussian_correlation_mean(radius, correlation_length)
        target_mean_xi = float(
            _gaussian_correlation_mean(
                np.array([target_radius]), correlation_length
            )[0]
        )
    if target_mean_xi <= 0.0:
        raise ValueError("correlation mean at target radius must be positive")
    amplitude = overdensity / max(target_mean_xi, 1.0e-30)
    return amplitude * xi, amplitude * mean_xi


def pie_temperature(table, hydrogen_density_cm3, redshift, fallback=1.0e4):
    """Return the tabulated UVB PIE temperature (heating=cooling)."""
    logt = np.linspace(table.log_temperature[0], table.log_temperature[-1], 512)
    temperature = 10.0**logt
    heating, cooling = table.rates(
        temperature, hydrogen_density_cm3, metallicity=1.0, redshift=redshift
    )
    net = np.asarray(heating) - np.asarray(cooling)
    crossings = np.flatnonzero(net[:-1] * net[1:] <= 0.0)
    if crossings.size:
        i = crossings[0]
        fraction = abs(net[i]) / max(abs(net[i]) + abs(net[i + 1]), 1.0e-300)
        return float(temperature[i] * (temperature[i + 1] / temperature[i])**fraction)
    return float(np.clip(fallback, temperature[0], temperature[-1]))


def cmb_temperature(redshift, temperature_0=2.7255):
    """Return the CMB blackbody temperature at a given redshift."""
    return float(temperature_0) * (1.0 + float(redshift))


def cmb_equilibrium_electron_fraction(ic):
    """Return the configured residual post-recombination electron fraction.

    Compton scattering equilibrates the gas temperature with the CMB but does
    not ionize hydrogen.  The electron fraction at z=100 is therefore a
    recombination-history input rather than a consequence of the Compton
    source.  The default ``2e-4`` is a standard residual-ionization value and
    can be overridden for convergence studies.
    """
    value = float(ic.get("cmb_residual_electron_fraction", 2.0e-4))
    if not 0.0 <= value <= 1.0:
        raise ValueError("cmb_residual_electron_fraction must lie in [0, 1]")
    return value


class Simwrap:
    """Build a comoving/supercomoving IC accepted by ``writehdf5``."""

    def __init__(self, ic, units, cosmology, pie_table=None, correlation_table=None):
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.par.CodeUnits = units
        self.par.unit_system = units.unit_system
        self.par.nogrid = int(ic["nogrid"])
        self.par.coordsys = "spherical"
        self.par.boxsize = np.array([float(ic["rmax"])])
        cosmic_time = float(ic["initial_cosmic_time"])
        self.par.time = np.array([cosmology.supercomoving_time(cosmic_time)])
        self.par.cosmological_expansion = True
        self.par.supercomoving_coordinates = True
        self.par.cosmological_gravity = True
        self.par.selfgravity = True
        self.par.externalgravity = False
        self.par.cosmology = cosmology
        self.par.cosmology_type = cosmology.type_name
        self.par.cosmology_t_ref = cosmology.t_ref
        self.par.cosmology_a_ref = cosmology.a_ref
        self.par.coordinate_frame = "comoving"
        self.par.time_coordinate = "supercomoving"
        self.par.velocity_representation = "supercomoving_peculiar"
        self.par.density_representation = "comoving"
        self.par.pressure_representation = "supercomoving"
        self.par.temperature_representation = "supercomoving"

        self.mesh.boundary = np.geomspace(float(ic["rmin"]), float(ic["rmax"]), self.par.nogrid + 1)
        # Keep a small, finite comoving inner wall when requested.  Setting
        # this face to zero would turn the test back into the singular
        # spherical-origin problem, whose origin flux is intentionally zero.
        inner_wall = float(ic.get("inner_wall_radius_comoving", ic["rmin"]))
        if inner_wall <= 0.0:
            self.mesh.boundary[0] = 0.0
        else:
            self.mesh.boundary[0] = inner_wall
        self.mesh.coordinate = cell_centres(self.mesh.boundary)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = 4.0 * np.pi / 3.0 * np.diff(self.mesh.boundary**3)

        a = float(cosmology.scale_factor(cosmic_time))
        hubble = float(cosmology.hubble(cosmic_time))
        rho_total = float(cosmology.background_density(cosmic_time))
        rho_comoving = rho_total * a**3
        fb = float(ic["baryon_fraction"])
        delta, mean_delta = density_contrast_profile(
            self.mesh.coordinate,
            ic,
            cosmology,
            correlation_table=correlation_table,
            length_unit_mpc_h=(
                float(units.length_in_cgs)
                / float((1.0 * unyt.Mpc).to_value("cm"))
                * float(ic.get("correlation_h", 0.674))
            ),
        )
        self.fluid.rho = rho_comoving * fb * (1.0 + delta) * np.ones(self.par.nogrid)
        rho_total_cgs = rho_total * units.mass_in_cgs / units.length_in_cgs**3
        rho_g_cgs = rho_total_cgs * fb * (1.0 + delta)
        n_h = (
            rho_g_cgs * float(ic["hydrogen_mass_fraction"])
            / PROTON_MASS_CGS
        )
        redshift = 1.0 / a - 1.0
        if bool(ic.get("cmb_equilibrium_initial", False)):
            temp_phys = np.full(
                self.par.nogrid,
                cmb_temperature(
                    redshift,
                    ic.get("cmb_temperature_0", 2.7255),
                ),
            )
            electron_fraction = np.full(
                self.par.nogrid,
                cmb_equilibrium_electron_fraction(ic),
            )
            self.fluid.xHI = 1.0 - electron_fraction
            self.fluid.mu = 1.0 / (
                float(ic["hydrogen_mass_fraction"])
                * (2.0 - self.fluid.xHI)
            )
        elif redshift > float(ic.get("uv_background_on_redshift", 10.0)):
            # Before the UV background turns on, initialize cold gas; CIE is
            # the active cooling model during this epoch.
            temp_phys = float(ic.get("cie_initial_temperature", 10.0))
        else:
            temp_phys = (
                pie_temperature(pie_table, float(np.median(n_h)), redshift)
                if pie_table else 1.0e4
            )
        self.fluid.temp = temp_phys * a**2 * np.ones(self.par.nogrid)
        if not bool(ic.get("cmb_equilibrium_initial", False)):
            self.fluid.mu = np.full(self.par.nogrid, float(ic["mu"]))
        self.fluid.vel = -a**2 * hubble * mean_delta * self.mesh.coordinate / 3.0


def make_dark_matter(ic, units, cosmology, correlation_table=None):
    count = int(ic["dark_matter_shells"])
    dm_inner = float(ic.get("dm_inner_radius", 1.0e-2))
    central_core_model = bool(ic.get("dm_central_core_model", False))
    central_core_radius = float(
        ic.get("dm_central_core_radius", dm_inner)
    ) if central_core_model else dm_inner
    if central_core_radius < dm_inner:
        raise ValueError("dm_central_core_radius must be >= dm_inner_radius")
    # A fixed unresolved core already represents the excess mass inside its
    # radius.  Do not leave live shells in the same volume and count them a
    # second time when they are later absorbed.
    shell_inner = central_core_radius if central_core_model else dm_inner
    boundaries = np.geomspace(shell_inner, float(ic["rmax"]), count + 1)
    radius = 0.5 * (boundaries[:-1] + boundaries[1:])
    volume = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    t = float(ic["initial_cosmic_time"])
    a = float(cosmology.scale_factor(t))
    hubble = float(cosmology.hubble(t))
    rho = float(cosmology.background_density(t)) * a**3
    dm_fraction = 1.0 - float(ic["baryon_fraction"])
    delta, mean_delta = density_contrast_profile(
        radius,
        ic,
        cosmology,
        correlation_table=correlation_table,
        length_unit_mpc_h=(
            float(units.length_in_cgs)
            / float((1.0 * unyt.Mpc).to_value("cm"))
            * float(ic.get("correlation_h", 0.674))
        ),
    )
    mass = rho * dm_fraction * (1.0 + delta) * volume
    velocity = -a**2 * hubble * mean_delta * radius / 3.0
    central_core_mass = None
    if central_core_model:
        core_radius = central_core_radius
        _, core_mean_delta = density_contrast_profile(
            np.asarray([core_radius]),
            ic,
            cosmology,
            correlation_table=correlation_table,
            length_unit_mpc_h=(
                float(units.length_in_cgs)
                / float((1.0 * unyt.Mpc).to_value("cm"))
                * float(ic.get("correlation_h", 0.674))
            ),
        )
        # Cosmological gravity already subtracts the homogeneous background;
        # only replace the unresolved overdensity excess with a softened
        # central mass.
        central_core_mass = max(
            0.0,
            rho * dm_fraction * float(core_mean_delta[0])
            * 4.0 * np.pi / 3.0 * core_radius**3,
        )
    shells = DarkMatterShells(
        radius=radius, velocity=velocity, mass=mass,
        angular_momentum=np.full(
            count, float(ic.get("dm_specific_angular_momentum", 0.0))
        ),
        softening=float(ic["softening"]), code_units=units,
        fixed_enclosed_mass=central_core_mass,
        central_core_radius=(
            float(ic.get("dm_central_core_radius", dm_inner))
            if central_core_mass is not None else 0.0
        ),
        core_absorption_velocity=float(ic.get("dm_core_absorption_velocity", 0.0)),
        core_absorption_energy=float(ic.get("dm_core_absorption_energy", 0.0)),
    )
    shells.central_core_radius = (
        float(ic.get("dm_central_core_radius", dm_inner))
        if central_core_mass is not None else 0.0
    )
    shells.central_core_mass = (
        float(central_core_mass) if central_core_mass is not None else 0.0
    )
    return shells


def splashback_radius(
    dm_radius_kpc, dm_mass, rvir_kpc=np.nan, bin_count=128,
):
    """Estimate splashback from the steepest outer DM density slope.

    The input shell masses are rebinned exactly, rather than differentiating
    the noisy density assigned to individual infinitesimal shells.  The
    returned radius is in the same proper-kpc basis as ``dm_radius_kpc``.
    """
    radius = np.asarray(dm_radius_kpc, dtype=float)
    mass = np.asarray(dm_mass, dtype=float)
    if not np.isfinite(rvir_kpc) or float(rvir_kpc) <= 0.0:
        return float("nan")
    valid = np.isfinite(radius) & np.isfinite(mass) & (radius > 0.0) & (mass > 0.0)
    radius = radius[valid]
    mass = mass[valid]
    if radius.size < 16:
        return float("nan")
    order = np.argsort(radius)
    radius = radius[order]
    mass = mass[order]
    edges = np.geomspace(
        max(radius[0] * 0.9, 1.0e-12),
        radius[-1] * 1.1,
        int(max(32, bin_count)) + 1,
    )
    shell_mass, _ = np.histogram(radius, bins=edges, weights=mass)
    shell_volume = 4.0 * np.pi / 3.0 * np.diff(edges**3)
    density = shell_mass / np.maximum(shell_volume, 1.0e-300)
    occupied = density > 0.0
    if np.count_nonzero(occupied) < 12:
        return float("nan")
    radii = np.sqrt(edges[:-1] * edges[1:])[occupied]
    density = density[occupied]
    log_radius = np.log(radii)
    log_density = np.log(density)
    # A short boxcar suppresses individual-shell noise while retaining the
    # broad splashback trough.
    window = min(7, log_density.size if log_density.size % 2 else log_density.size - 1)
    if window >= 3:
        padded = np.pad(log_density, (window // 2,), mode="edge")
        log_density = np.convolve(
            padded, np.ones(window) / float(window), mode="valid"
        )
    slope = np.gradient(log_density, log_radius)
    # Splashback is an outer-halo caustic; features inside r200 are inner
    # structure and must not be reported as the splashback radius.
    lower = max(float(rvir_kpc), radii[0])
    upper = min(3.0 * float(rvir_kpc), 0.95 * radii[-1])
    if upper <= lower:
        return float("nan")
    candidates = np.flatnonzero((radii >= lower) & (radii <= upper))
    if candidates.size < 3:
        return float("nan")
    # Avoid reporting a weak numerical edge as splashback.
    local = candidates[np.argmin(slope[candidates])]
    if not np.isfinite(slope[local]) or slope[local] > -1.0:
        return float("nan")
    return float(radii[local])


def profiles(sim, dm, cosmic_time, cosmology, ic):
    """Measure virial, shock, disc radii and enclosed total masses."""
    first = int(sim.par.noghost)
    last = first + int(sim.par.nogrid)
    x = np.asarray(sim.mesh.coordinate[first:last], dtype=float)
    edges = np.asarray(sim.mesh.boundary[first:last + 1], dtype=float)
    rho = np.asarray(sim.fluid.rho[first:last], dtype=float)
    gas_mass = rho * 4.0 * np.pi / 3.0 * np.diff(edges**3)
    gas_cumulative = np.concatenate(([0.0], np.cumsum(gas_mass)))
    dm_order = np.argsort(dm.radius)
    dm_r = dm.radius[dm_order]
    dm_m = dm.mass[dm_order]
    dm_cumulative = np.cumsum(dm_m)
    a = float(cosmology.scale_factor(cosmic_time))
    proper = a * x
    rho_crit = float(cosmology.background_density(cosmic_time))

    def total_mass_at(proper_radius):
        comoving = np.asarray(proper_radius) / a
        cg = np.interp(comoving, edges, gas_cumulative, left=0.0, right=gas_cumulative[-1])
        cd = np.interp(comoving, dm_r, dm_cumulative, left=0.0, right=dm_cumulative[-1])
        return cg + cd

    # Determine r_vir from the live collisionless profile.  The DM shells
    # carry the cleanest Lagrangian mass coordinate; infer the total mass by
    # dividing by the cosmic DM fraction instead of allowing a sparse or
    # shocked gas mesh to set the halo edge.
    fdm = 1.0 - float(ic["baryon_fraction"])
    dm_proper = a * dm_r
    dm_total_mass = dm_cumulative / max(fdm, 1.0e-30)
    dm_mean_density = dm_total_mass / (
        4.0 * np.pi / 3.0 * np.maximum(dm_proper, 1.0e-12) ** 3
    )
    overdensity = dm_mean_density / max(200.0 * rho_crit, 1.0e-30)
    candidates = np.flatnonzero(overdensity >= 1.0)
    target_mass = float(ic.get("target_halo_mass", np.nan))
    target_index = np.searchsorted(dm_total_mass, target_mass)
    rtarget = (
        float(dm_proper[target_index])
        if np.isfinite(target_mass) and target_index < dm_total_mass.size
        else float("nan")
    )
    if candidates.size:
        virial_index = int(candidates[-1])
        rvir = float(dm_proper[virial_index])
        mvir = float(dm_total_mass[virial_index])
    else:
        # No formal r_200 exists if the live DM profile is everywhere below
        # 200 rho_crit.  Keep it NaN rather than relabelling a Lagrangian
        # target-mass radius as a virial radius.
        rvir = float("nan")
        mvir = float("nan")

    if np.isfinite(rvir) and rvir > 0.0 and np.isfinite(mvir):
        temperature_factor = float(
            sim.par.CodeUnits.boltzmann_code
            / sim.par.CodeUnits.proton_mass_code
        )
        tvir = float(
            float(ic.get("mu", 0.59))
            * float(cosmology.gravitational_constant)
            * mvir / (2.0 * rvir * temperature_factor)
        )
    else:
        tvir = float("nan")

    temp_phys = np.asarray(sim.fluid.temp[first:last], dtype=float) / a**2
    velocity_phys = np.asarray(
        cosmology.physical_velocity(
            x,
            np.asarray(sim.fluid.vel[first:last], dtype=float),
            float(sim.fluid.time),
        ),
        dtype=float,
    )
    gamma = float(sim.par.gamma)
    entropy_proxy = temp_phys / np.maximum(rho, 1.0e-300) ** (gamma - 1.0)
    # Temperatures at or below 1 K are numerical-floor/invalid states in this
    # run; allowing them would create enormous artificial entropy jumps.
    finite_entropy = (
        np.isfinite(entropy_proxy) & (entropy_proxy > 0.0)
        & np.isfinite(temp_phys) & (temp_phys > 1.0)
        & np.isfinite(rho) & (rho > 0.0)
    )
    if np.count_nonzero(finite_entropy) >= 7:
        lower_radius = proper[0]
        if np.isfinite(rvir) and rvir > proper[0]:
            # The virial shock is an outer-halo feature.  Exclude inner
            # cooling/centrifugal transitions from the shock diagnostic.
            lower_radius = max(lower_radius, 0.5 * rvir)
        elif np.isfinite(rtarget):
            # The virial shock is an outer-halo feature.  Do not let an
            # unresolved inner cooling/adiabatic feature become r_shock
            # merely because it has a larger cell-to-cell gradient.
            lower_radius = max(lower_radius, 0.3 * rtarget)
        # A percentage-of-radius cut removes too few cells on this logarithmic
        # mesh.  Leave a fixed buffer outside the candidate and its five-cell
        # smoothing stencil so the explicitly reset EdS reservoir cannot be
        # reported as a virial shock.
        outer_buffer_cells = 8
        upper_index = max(3, proper.size - outer_buffer_cells)
        upper_radius = proper[upper_index]
        if np.isfinite(rvir) and rvir > proper[0]:
            upper_radius = min(upper_radius, 3.0 * rvir)
        valid = (
            (proper > lower_radius)
            & (proper < upper_radius)
        )
        candidate = np.flatnonzero(valid)
        if candidate.size:
            # Use the strongest resolved inward entropy increase directly;
            # do not let a floor cell inside the smoothing stencil veto it.
            shock_candidates = candidate
            resolved = []
            resolved_entropy_jumps = []
            for local in shock_candidates:
                inner = max(0, int(local) - 2)
                outer = min(proper.size - 1, int(local) + 2)
                compression = rho[inner] / max(rho[outer], 1.0e-300)
                entropy_jump = entropy_proxy[inner] - entropy_proxy[outer]
                upstream_velocity = velocity_phys[outer]
                downstream_velocity = velocity_phys[inner]
                decelerated = (
                    upstream_velocity < 0.0
                    and downstream_velocity > upstream_velocity
                    and abs(downstream_velocity) < abs(upstream_velocity)
                )
                if (
                    finite_entropy[inner] and finite_entropy[outer]
                    and np.isfinite(entropy_jump)
                    and compression >= 1.2
                    and entropy_jump > 0.0
                    and decelerated
                ):
                    resolved.append(int(local))
                    resolved_entropy_jumps.append(float(entropy_jump))
            if resolved:
                resolved = np.asarray(resolved, dtype=int)
                local = int(resolved[np.argmax(resolved_entropy_jumps)])
                rshock = float(proper[local])
            else:
                rshock = np.nan
        else:
            rshock = np.nan
    else:
        rshock = np.nan

    rsplashback = splashback_radius(
        dm_proper,
        dm_m,
        rvir_kpc=rvir,
        bin_count=int(getattr(sim.par, "dm_density_bins", 128)),
    )

    g_code = float(cosmology.gravitational_constant)
    j = float(ic["specific_angular_momentum"])
    target_mass = float(ic.get("target_halo_mass", np.nan))
    # This is a halo-scale centrifugal-radius diagnostic, not a resolved
    # rotating-disc solution.  Using the local enclosed mass here makes the
    # radius grow artificially when the correlation IC has assembled only a
    # small fraction of the target halo inside the sampled radius.
    disc_mass = target_mass if np.isfinite(target_mass) and target_mass > 0.0 else mvir
    if np.isfinite(disc_mass) and disc_mass > 0.0:
        rdisc = float(j**2 / (g_code * disc_mass))
    else:
        rdisc = float("nan")
    rdisc_max = rtarget if np.isfinite(rtarget) else proper[-1]
    if np.isfinite(rdisc):
        rdisc = float(np.clip(rdisc, proper[0], max(rdisc_max, proper[0])))
    return {
        "time_Gyr": float(cosmic_time * sim.par.CodeUnits.time_unit.to_value("Gyr")),
        "rvir_kpc": rvir,
        "rtarget_kpc": rtarget,
        "rho_crit_code": rho_crit,
        "max_delta200": float(np.nanmax(overdensity)),
        "rshock_kpc": rshock,
        "rsplashback_kpc": rsplashback,
        "rdisc_kpc": rdisc,
        "mvir": mvir,
        "tvir_K": tvir,
        "mshock": float(total_mass_at(rshock)) if np.isfinite(rshock) else np.nan,
        "mdisc": float(total_mass_at(rdisc)),
    }


def density_profiles(sim, dm, cosmic_time, cosmology):
    """Return physical gas and shell-based DM density profiles."""
    first = int(sim.par.noghost)
    last = first + int(sim.par.nogrid)
    a = float(cosmology.scale_factor(cosmic_time))
    gas = gas_density_profile(sim, cosmic_time, cosmology)

    order = np.argsort(dm.radius)
    dm_radius_comoving = np.asarray(dm.radius[order], dtype=float)
    dm_mass = np.asarray(dm.mass[order], dtype=float)
    dm_radius = a * dm_radius_comoving
    if dm_radius.size > 1:
        dm_edges = np.empty(dm_radius.size + 1)
        dm_edges[1:-1] = np.sqrt(dm_radius[:-1] * dm_radius[1:])
        dm_edges[0] = dm_radius[0]**2 / dm_edges[1]
        dm_edges[-1] = dm_radius[-1]**2 / dm_edges[-2]
    else:
        dm_edges = np.array([0.5 * dm_radius[0], 1.5 * dm_radius[0]])
    dm_volume = 4.0 * np.pi / 3.0 * np.diff(dm_edges**3)
    dm_density = dm_mass / np.maximum(dm_volume, 1.0e-30)
    return {
        "time_Gyr": float(cosmic_time * sim.par.CodeUnits.time_unit.to_value("Gyr")),
        "gas_radius_kpc": gas["radius_proper_kpc"],
        "gas_density_code": gas["density_proper_code"],
        "dm_radius_kpc": dm_radius,
        "dm_density_code": dm_density,
        "dm_mass": dm_mass,
        # The softened unresolved core is part of the gravitating DM profile
        # even though it is not represented by a live shell.
        "dm_central_core_mass": float(getattr(dm, "central_core_mass", 0.0)),
        "dm_central_core_radius_kpc": (
            a * float(getattr(dm, "central_core_radius", 0.0))
        ),
    }


def gas_density_profile(sim, cosmic_time, cosmology):
    """Return one snapshot of the physical gas density profile.

    The mesh coordinate is comoving, while ``fluid.rho`` is the
    supercomoving/comoving density used by the solver.  The returned density
    is physical (divide by ``a**3``), and both radius representations are
    stored so an evolution plot can use a fixed comoving x-axis while
    marking the proper virial radius consistently.
    """
    first = int(sim.par.noghost)
    last = first + int(sim.par.nogrid)
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    radius_comoving = np.asarray(
        sim.mesh.coordinate[first:last], dtype=float
    )
    density_comoving = np.asarray(
        sim.fluid.rho[first:last], dtype=float
    )
    return {
        "time_Gyr": float(cosmic_time * sim.par.CodeUnits.time_unit.to_value("Gyr")),
        "scale_factor": scale_factor,
        "radius_comoving_kpc": radius_comoving,
        "radius_proper_kpc": scale_factor * radius_comoving,
        "density_proper_code": density_comoving / scale_factor**3,
    }
class VolumeSmoothedDarkMatter:
    """Use shell mass interpolated linearly in enclosed volume for gas force."""

    def __init__(self, shells):
        self.shells = shells

    def __getattr__(self, name):
        return getattr(self.shells, name)

    def gravitating_enclosed_mass(self, radius=None,
                                  include_shell_mass_with_fixed=False):
        if radius is None:
            return self.shells.gravitating_enclosed_mass(
                radius,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            )
        shell_radius = np.asarray(self.shells.radius, dtype=float)
        shell_enclosed = np.asarray(
            self.shells.gravitating_enclosed_mass(
                shell_radius,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            ),
            dtype=float,
        )
        total = float(np.sum(self.shells.mass))
        if self.shells.fixed_enclosed_mass is not None:
            total += float(self.shells.fixed_enclosed_mass)
        outer_radius = shell_radius[-1] + 0.5 * (
            shell_radius[-1] - shell_radius[-2]
        )
        interpolation_radius = np.concatenate(([0.0], shell_radius, [outer_radius]))
        interpolation_mass = np.concatenate(([0.0], shell_enclosed, [total]))
        requested = np.asarray(radius, dtype=float)
        return np.interp(
            requested**3,
            interpolation_radius**3,
            interpolation_mass,
            left=0.0,
            right=total,
        )
