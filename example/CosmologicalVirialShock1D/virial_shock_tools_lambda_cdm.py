# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Initial conditions and diagnostics for the cosmological virial-shock test."""

from math import erf

import numpy as np
import unyt

from example.CosmologicalVirialShock1D.splashback import splashback_radius
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import quantity_to_value

DEFAULT_CENTRAL_CORE_MODEL = False


def cell_centres(boundary_comoving_code):
    inner, outer = boundary_comoving_code[:-1], boundary_comoving_code[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def radius_perturbation_comoving_code(config):
    """Return the comoving top-hat radius for the requested halo mass."""
    initial_condition = config["initial_condition"]
    code_unit_system = config["_code_unit_system"]
    cosmology = config["_cosmology"]
    if initial_condition.get("target_halo_mass") is None:
        return quantity_to_value(
            initial_condition["radius_perturbation_comoving"],
            code_unit_system.length_unit,
        )
    t = quantity_to_value(initial_condition["time_cosmic"], code_unit_system.time_unit)
    a = float(cosmology.scale_factor(t))
    rho_comoving = float(cosmology.background_density(t)) * a**3
    overdensity = float(initial_condition["initial_overdensity"])
    return float(
        (
            quantity_to_value(initial_condition["target_halo_mass"], code_unit_system.mass_unit)
            / ((4.0 * np.pi / 3.0) * rho_comoving * (1.0 + overdensity))
        )
        ** (1.0 / 3.0),
    )


def _gaussian_correlation_mean(radius_comoving_code, correlation_length):
    """Mean enclosed Gaussian correlation shape, normalized to xi(0)=1."""
    radius_comoving_code = np.asarray(radius_comoving_code, dtype=float)
    x = radius_comoving_code / max(float(correlation_length), 1.0e-30)
    erf_x = np.vectorize(erf, otypes=[float])(x)
    integral = np.sqrt(np.pi) / 4.0 * erf_x - 0.5 * x * np.exp(-(x**2))
    result = np.divide(3.0 * integral, np.maximum(x**3, 1.0e-30))
    result = np.asarray(result, dtype=float)
    result[x < 1.0e-4] = 1.0  # noqa: PLR2004
    return result


def _correlation_profile(radius_comoving_code, table, length_unit_mpc_h):
    """Interpolate xi and its enclosed mean in simulation length units."""
    radius_comoving_code = np.asarray(radius_comoving_code, dtype=float)
    table_radius = np.asarray(table["radius_mpc_h"], dtype=float)
    table_correlation = np.asarray(table["correlation"], dtype=float)
    if table_radius.ndim != 1 or table_correlation.ndim != 1:
        raise ValueError("linear correlation table arrays must be one-dimensional")
    if table_radius.size != table_correlation.size or table_radius.size < 2:  # noqa: PLR2004
        raise ValueError("linear correlation table arrays have incompatible sizes")
    if np.any(np.diff(table_radius) <= 0.0):
        raise ValueError("linear correlation table radii must be increasing")
    radius_mpc_h = radius_comoving_code * float(length_unit_mpc_h)
    if np.any(radius_mpc_h > table_radius[-1]):
        raise ValueError("initial-condition radius exceeds the correlation table")

    # Include the origin using the first tabulated value, then integrate
    # xi(r) r^2 dr once so arbitrary shell radii get a consistent enclosed
    # mean correlation.
    integration_radius = np.concatenate(([0.0], table_radius))
    integration_xi = np.concatenate(([table_correlation[0]], table_correlation))
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5
                * (
                    integration_xi[1:] * integration_radius[1:] ** 2
                    + integration_xi[:-1] * integration_radius[:-1] ** 2
                )
                * np.diff(integration_radius),
            ),
        ),
    )
    xi = np.interp(
        radius_mpc_h,
        table_radius,
        table_correlation,
        left=table_correlation[0],
        right=table_correlation[-1],
    )
    enclosed_integral = np.interp(
        radius_mpc_h,
        integration_radius,
        cumulative,
        left=0.0,
        right=cumulative[-1],
    )
    mean_xi = np.where(
        radius_mpc_h <= table_radius[0],
        table_correlation[0],
        3.0 * enclosed_integral / np.maximum(radius_mpc_h**3, 1.0e-300),
    )
    return xi, mean_xi


def density_contrast_profile(radius_comoving_code, config, length_unit_mpc_h=1.0):
    """Return ``(delta, mean_delta)`` for the configured growing mode.

    ``linear_correlation`` uses the supplied tabulated linear-theory
    correlation function.  Its amplitude is fixed by the requested mean
    overdensity inside the target Lagrangian radius.
    """
    initial_condition = config["initial_condition"]
    config["_cosmology"]
    correlation_table = config.get("_correlation_table")
    radius_comoving_code = np.asarray(radius_comoving_code, dtype=float)
    radius_perturbation_comoving_code_value = radius_perturbation_comoving_code(config)
    overdensity = float(initial_condition["initial_overdensity"])
    profile = str(initial_condition.get("rho_proper_profile", "top_hat")).lower()
    if profile == "top_hat":
        inside = radius_comoving_code < radius_perturbation_comoving_code_value
        delta = overdensity * inside
        mean_delta = overdensity * np.where(
            inside,
            1.0,
            (radius_perturbation_comoving_code_value / np.maximum(radius_comoving_code, 1.0e-30))
            ** 3,
        )
        return np.asarray(delta, dtype=float), np.asarray(mean_delta, dtype=float)
    if profile not in ("linear_correlation", "gaussian_correlation"):
        raise ValueError(f"unknown rho_proper_profile {profile!r}")

    if profile == "linear_correlation":
        if correlation_table is None:
            raise ValueError(
                "linear_correlation requires a tabulated correlation table",
            )
        xi, mean_xi = _correlation_profile(
            radius_comoving_code,
            correlation_table,
            length_unit_mpc_h,
        )
        target_mean_xi = float(
            _correlation_profile(
                np.array([radius_perturbation_comoving_code_value]),
                correlation_table,
                length_unit_mpc_h,
            )[1][0],
        )
    else:
        correlation_length = float(
            initial_condition.get(
                "correlation_length",
                0.5 * radius_perturbation_comoving_code_value,
            ),
        )
        xi = np.exp(-((radius_comoving_code / max(correlation_length, 1.0e-30)) ** 2))
        mean_xi = _gaussian_correlation_mean(radius_comoving_code, correlation_length)
        target_mean_xi = float(
            _gaussian_correlation_mean(
                np.array([radius_perturbation_comoving_code_value]),
                correlation_length,
            )[0],
        )
    if target_mean_xi <= 0.0:
        raise ValueError("correlation mean at target radius must be positive")
    amplitude = overdensity / max(target_mean_xi, 1.0e-30)
    return amplitude * xi, amplitude * mean_xi


def pie_temperature(table, hydrogen_number_density_cgs_cm3, redshift, fallback=1.0e4):
    """Return the tabulated UVB PIE temperature (heating=cooling)."""
    logt = np.linspace(table.log_temperature[0], table.log_temperature[-1], 512)
    temperature_proper_cgs_K = 10.0**logt
    heating, cooling = table.rates(
        temperature_proper_cgs_K,
        hydrogen_number_density_cgs_cm3,
        metallicity=1.0,
        redshift=redshift,
    )
    net = np.asarray(heating) - np.asarray(cooling)
    crossings = np.flatnonzero(net[:-1] * net[1:] <= 0.0)
    if crossings.size:
        i = crossings[0]
        fraction = abs(net[i]) / max(abs(net[i]) + abs(net[i + 1]), 1.0e-300)
        return float(
            temperature_proper_cgs_K[i]
            * (temperature_proper_cgs_K[i + 1] / temperature_proper_cgs_K[i]) ** fraction,
        )
    return float(np.clip(fallback, temperature_proper_cgs_K[0], temperature_proper_cgs_K[-1]))


def cmb_temperature(redshift, temperature_0=2.7255):
    """Return the CMB blackbody temperature at a given redshift."""
    return float(temperature_0) * (1.0 + float(redshift))


def cmb_equilibrium_electron_fraction(initial_condition):
    """Return the configured residual post-recombination electron fraction.

    Compton scattering equilibrates the gas temperature with the CMB but does
    not ionize hydrogen.  The electron fraction at z=100 is therefore a
    recombination-history input rather than a consequence of the Compton
    source.  The default ``2e-4`` is a standard residual-ionization value and
    can be overridden for convergence studies.
    """
    value = float(initial_condition.get("cmb_residual_electron_fraction", 2.0e-4))
    if not 0.0 <= value <= 1.0:
        raise ValueError("cmb_residual_electron_fraction must lie in [0, 1]")
    return value


def make_dark_matter(config):
    initial_condition = config["initial_condition"]
    code_unit_system = config["_code_unit_system"]
    cosmology = config["_cosmology"]
    count = int(initial_condition["dark_matter_shells"])
    dm_inner = quantity_to_value(
        initial_condition.get("radius_inner_dark_matter_comoving", 1.0e-2),
        code_unit_system.length_unit,
    )
    central_core_model = DEFAULT_CENTRAL_CORE_MODEL
    central_core_radius = (
        quantity_to_value(
            initial_condition.get("dm_central_core_radius", dm_inner),
            code_unit_system.length_unit,
        )
        if central_core_model
        else dm_inner
    )
    if central_core_radius < dm_inner:
        raise ValueError("dm_central_core_radius must be >= radius_inner_dark_matter_comoving")
    # A fixed unresolved core already represents the excess mass inside its
    # radius.  Do not leave live shells in the same volume_comoving_code and count them a
    # second time when they are later absorbed.
    shell_inner = central_core_radius if central_core_model else dm_inner
    boundaries = np.geomspace(
        shell_inner,
        quantity_to_value(initial_condition["radius_outer_comoving"], code_unit_system.length_unit),
        count + 1,
    )
    radius_comoving_code = 0.5 * (boundaries[:-1] + boundaries[1:])
    volume_comoving_code = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    t = quantity_to_value(initial_condition["time_cosmic"], code_unit_system.time_unit)
    a = float(cosmology.scale_factor(t))
    hubble = float(cosmology.hubble(t))
    rho_comoving_code = float(cosmology.background_density(t)) * a**3
    dm_fraction = 1.0 - float(initial_condition["baryon_fraction"])
    delta, mean_delta = density_contrast_profile(
        radius_comoving_code,
        config,
        length_unit_mpc_h=(
            float(code_unit_system.length_in_cgs)
            / float((1.0 * unyt.Mpc).to_value("cm"))
            * float(initial_condition.get("correlation_h", 0.674))
        ),
    )
    mass_comoving_code = rho_comoving_code * dm_fraction * (1.0 + delta) * volume_comoving_code
    vel_supercomoving_code = -(a**2) * hubble * mean_delta * radius_comoving_code / 3.0
    central_core_mass = None
    if central_core_model:
        core_radius_comoving_code = central_core_radius
        _, core_mean_delta = density_contrast_profile(
            np.asarray([core_radius_comoving_code]),
            config,
            length_unit_mpc_h=(
                float(code_unit_system.length_in_cgs)
                / float((1.0 * unyt.Mpc).to_value("cm"))
                * float(initial_condition.get("correlation_h", 0.674))
            ),
        )
        # Cosmological gravity already subtracts the homogeneous background;
        # only replace the unresolved overdensity excess with a softened
        # central mass.
        central_core_mass = max(
            0.0,
            rho_comoving_code
            * dm_fraction
            * float(core_mean_delta[0])
            * 4.0
            * np.pi
            / 3.0
            * core_radius_comoving_code**3,
        )
    shells = DarkMatterShells(
        radius_comoving_code,
        vel_supercomoving_code,
        mass_comoving_code,
        angular_momentum=np.full(
            count,
            float(initial_condition.get("dm_specific_angular_momentum", 0.0)),
        ),
        softening=quantity_to_value(initial_condition["softening"], code_unit_system.length_unit),
        code_units=code_unit_system,
        fixed_enclosed_mass=central_core_mass,
        central_core_radius=(
            quantity_to_value(
                initial_condition.get("dm_central_core_radius", dm_inner),
                code_unit_system.length_unit,
            )
            if central_core_mass is not None
            else 0.0
        ),
        core_absorption_velocity=float(initial_condition.get("dm_core_absorption_velocity", 0.0)),
        core_absorption_energy=float(initial_condition.get("dm_core_absorption_energy", 0.0)),
    )
    shells.central_core_radius = (
        quantity_to_value(
            initial_condition.get("dm_central_core_radius", dm_inner),
            code_unit_system.length_unit,
        )
        if central_core_mass is not None
        else 0.0
    )
    shells.central_core_mass = float(central_core_mass) if central_core_mass is not None else 0.0
    return shells


def _find_temperature_shock_radius(
    proper,
    rho_comoving_code,
    temp_phys,
    velocity_phys,
    rvir,
    rtarget,
):
    finite_temperature = np.isfinite(temp_phys) & (temp_phys > 0.0)
    if np.count_nonzero(finite_temperature) < 7:  # noqa: PLR2004
        return np.nan
    log_temperature = np.log10(np.maximum(temp_phys, 1.0e-30))
    smoothed = np.convolve(
        np.pad(log_temperature, (2, 2), mode="edge"),
        np.ones(5) / 5.0,
        mode="valid",
    )
    gradient = np.gradient(smoothed, np.log10(np.maximum(proper, 1.0e-12)))
    lower_radius = proper[0]
    if np.isfinite(rvir) and rvir > proper[0]:
        lower_radius = max(lower_radius, 0.5 * rvir)
    elif np.isfinite(rtarget):
        lower_radius = max(lower_radius, 0.3 * rtarget)
    upper_radius = proper[max(3, proper.size - 8)]
    if np.isfinite(rvir) and rvir > proper[0]:
        upper_radius = min(upper_radius, 3.0 * rvir)
    candidates = np.flatnonzero(
        finite_temperature & (proper > lower_radius) & (proper < upper_radius),
    )
    resolved = []
    for local in candidates[gradient[candidates] < -0.05]:  # noqa: PLR2004
        inner = max(0, int(local) - 2)
        outer = min(proper.size - 1, int(local) + 2)
        compression = rho_comoving_code[inner] / max(rho_comoving_code[outer], 1.0e-300)
        heating = temp_phys[inner] / max(temp_phys[outer], 1.0e-300)
        decelerated = (
            velocity_phys[outer] < 0.0
            and velocity_phys[inner] > velocity_phys[outer]
            and abs(velocity_phys[inner]) < abs(velocity_phys[outer])
        )
        if compression >= 1.2 and heating >= 1.2 and decelerated:  # noqa: PLR2004
            resolved.append(int(local))
    if not resolved:
        return np.nan
    return float(proper[resolved[int(np.argmin(gradient[resolved]))]])


def profiles(sim, dark_matter, time_cosmic_code, config):
    """Measure virial, shock, disc radii and enclosed total masses."""
    initial_condition = config["initial_condition"]
    code_unit_system = config["_code_unit_system"]
    cosmology = config["_cosmology"]
    first = int(sim.par.noghost)
    last = first + int(sim.par.nogrid)
    x = np.asarray(sim.mesh.x_comoving_code[first:last], dtype=float)
    edges = np.asarray(sim.mesh.boundary_comoving_code[first : last + 1], dtype=float)
    rho_comoving_code = np.asarray(sim.fluid.rho_comoving_code[first:last], dtype=float)
    gas_mass_comoving_code = rho_comoving_code * 4.0 * np.pi / 3.0 * np.diff(edges**3)
    gas_cumulative = np.concatenate(([0.0], np.cumsum(gas_mass_comoving_code)))
    dm_order = np.argsort(dark_matter.radius)
    dm_radius_comoving_code = dark_matter.radius[dm_order]
    dm_mass_comoving_code = dark_matter.mass[dm_order]
    dm_cumulative = np.cumsum(dm_mass_comoving_code)
    a = float(cosmology.scale_factor(time_cosmic_code))
    proper = a * x
    rho_crit = float(cosmology.background_density(time_cosmic_code))

    def total_mass_at(proper_radius):
        comoving = np.asarray(proper_radius) / a
        cg = np.interp(comoving, edges, gas_cumulative, left=0.0, right=gas_cumulative[-1])
        cd = np.interp(
            comoving,
            dm_radius_comoving_code,
            dm_cumulative,
            left=0.0,
            right=dm_cumulative[-1],
        )
        return cg + cd

    # Determine r_vir from the live collisionless profile.  The DM shells
    # carry the cleanest Lagrangian mass coordinate; infer the total mass by
    # dividing by the cosmic DM fraction instead of allowing a sparse or
    # shocked gas mesh to set the halo edge.
    fdm = 1.0 - float(initial_condition["baryon_fraction"])
    dm_radius_proper_code = a * dm_radius_comoving_code
    dm_total_mass_comoving_code = dm_cumulative / max(fdm, 1.0e-30)
    dm_mean_density_comoving_code = dm_total_mass_comoving_code / (
        4.0 * np.pi / 3.0 * np.maximum(dm_radius_proper_code, 1.0e-12) ** 3
    )
    overdensity = dm_mean_density_comoving_code / max(200.0 * rho_crit, 1.0e-30)
    candidates = np.flatnonzero(overdensity >= 1.0)
    target_mass = (
        quantity_to_value(
            initial_condition["target_halo_mass"],
            code_unit_system.mass_unit,
        )
        if "target_halo_mass" in initial_condition
        else np.nan
    )
    target_index = np.searchsorted(dm_total_mass_comoving_code, target_mass)
    rtarget = (
        float(dm_radius_proper_code[target_index])
        if np.isfinite(target_mass) and target_index < dm_total_mass_comoving_code.size
        else float("nan")
    )
    if candidates.size:
        virial_index = int(candidates[-1])
        rvir = float(dm_radius_proper_code[virial_index])
        mvir = float(dm_total_mass_comoving_code[virial_index])
    else:
        # No formal r_200 exists if the live DM profile is everywhere below
        # 200 rho_crit.  Keep it NaN rather than relabelling a Lagrangian
        # target-mass radius as a virial radius.
        rvir = float("nan")
        mvir = float("nan")

    if np.isfinite(rvir) and rvir > 0.0 and np.isfinite(mvir):
        temperature_factor = float(
            sim.par.units.CodeUnits.boltzmann_code / sim.par.units.CodeUnits.proton_mass_code,
        )
        tvir = float(
            float(initial_condition.get("mu", 0.59))
            * float(cosmology.gravitational_constant)
            * mvir
            / (2.0 * rvir * temperature_factor),
        )
    else:
        tvir = float("nan")

    temp_phys = np.asarray(sim.fluid.temp_supercomoving_code[first:last], dtype=float) / a**2
    velocity_phys = np.asarray(
        cosmology.physical_velocity(
            x,
            np.asarray(sim.fluid.vel_supercomoving_code[first:last], dtype=float),
            float(sim.fluid.tau_supercomoving_code),
        ),
        dtype=float,
    )
    rshock = _find_temperature_shock_radius(
        proper,
        rho_comoving_code,
        temp_phys,
        velocity_phys,
        rvir,
        rtarget,
    )

    rsplashback = splashback_radius(
        dm_radius_proper_code,
        dm_mass_comoving_code,
        rvir_proper_code=rvir,
        bin_count=int(getattr(sim.par, "dm_density_bins", 128)),
    )

    g_code = float(cosmology.gravitational_constant)
    j = float(initial_condition["specific_angular_momentum"])
    target_mass = (
        quantity_to_value(
            initial_condition["target_halo_mass"],
            code_unit_system.mass_unit,
        )
        if "target_halo_mass" in initial_condition
        else np.nan
    )
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
        "time_cosmic_Gyr": float(
            time_cosmic_code * sim.par.units.CodeUnits.time_unit.to_value("Gyr"),
        ),
        "rvir_kpc": rvir,
        "rtarget_kpc": rtarget,
        "rho_crit_code": rho_crit,
        "max_delta200": float(np.nanmax(overdensity)),
        "rshock_kpc": rshock,
        "rsplashback_kpc": rsplashback,
        "rdisc_kpc": rdisc,
        "mvir": mvir,
        "tvir_cgs_K": tvir,
        "mshock": float(total_mass_at(rshock)) if np.isfinite(rshock) else np.nan,
        "mdisc": float(total_mass_at(rdisc)),
    }


def density_profiles(sim, dark_matter, time_cosmic_code, config):
    """Return physical gas and shell-based DM density profiles."""
    cosmology = config["_cosmology"]
    first = int(sim.par.noghost)
    first + int(sim.par.nogrid)
    a = float(cosmology.scale_factor(time_cosmic_code))
    gas = gas_density_profile(sim, time_cosmic_code, config)

    order = np.argsort(dark_matter.radius)
    dm_radius_comoving_code = np.asarray(dark_matter.radius[order], dtype=float)
    dm_mass_comoving_code = np.asarray(dark_matter.mass[order], dtype=float)
    dm_radius_proper_code = a * dm_radius_comoving_code
    if dm_radius_proper_code.size > 1:
        dm_edges = np.empty(dm_radius_proper_code.size + 1)
        dm_edges[1:-1] = np.sqrt(dm_radius_proper_code[:-1] * dm_radius_proper_code[1:])
        dm_edges[0] = dm_radius_proper_code[0] ** 2 / dm_edges[1]
        dm_edges[-1] = dm_radius_proper_code[-1] ** 2 / dm_edges[-2]
    else:
        dm_edges = np.array([0.5 * dm_radius_proper_code[0], 1.5 * dm_radius_proper_code[0]])
    dm_volume = 4.0 * np.pi / 3.0 * np.diff(dm_edges**3)
    dm_density_proper_code = dm_mass_comoving_code / np.maximum(dm_volume, 1.0e-30)
    return {
        "time_cosmic_Gyr": float(
            time_cosmic_code * sim.par.units.CodeUnits.time_unit.to_value("Gyr"),
        ),
        "gas_radius_kpc": gas["radius_proper_kpc"],
        "gas_rho_proper_code": gas["rho_proper_code"],
        "dm_radius_proper_kpc": dm_radius_proper_code,
        "dm_rho_proper_code": dm_density_proper_code,
        "dm_mass_comoving_code": dm_mass_comoving_code,
        # The softened unresolved core is part of the gravitating DM profile
        # even though it is not represented by a live shell.
        "dm_central_core_mass": float(getattr(dark_matter, "central_core_mass", 0.0)),
        "dm_central_core_radius_kpc": (a * float(getattr(dark_matter, "central_core_radius", 0.0))),
    }


def gas_density_profile(sim, time_cosmic_code, config):
    """Return one snapshot of the physical gas density profile.

    The mesh coordinate is comoving, while ``fluid.rho_comoving_code`` is the
    supercomoving/comoving density used by the solver.  The returned density
    is physical (divide by ``a**3``), and both radius representations are
    stored so an evolution plot can use a fixed comoving x-axis while
    marking the proper virial radius consistently.
    """
    cosmology = config["_cosmology"]
    first = int(sim.par.noghost)
    last = first + int(sim.par.nogrid)
    scale_factor = float(cosmology.scale_factor(time_cosmic_code))
    radius_comoving = np.asarray(
        sim.mesh.x_comoving_code[first:last],
        dtype=float,
    )
    density_comoving = np.asarray(
        sim.fluid.rho_comoving_code[first:last],
        dtype=float,
    )
    return {
        "time_cosmic_Gyr": float(
            time_cosmic_code * sim.par.units.CodeUnits.time_unit.to_value("Gyr"),
        ),
        "scale_factor": scale_factor,
        "radius_comoving_kpc": radius_comoving,
        "radius_proper_kpc": scale_factor * radius_comoving,
        "rho_proper_code": density_comoving / scale_factor**3,
    }


class VolumeSmoothedDarkMatter:
    """Use shell mass interpolated linearly in enclosed volume_comoving_code for gas force."""

    def __init__(self, shells):
        self.shells = shells

    def step(self, *args, **kwargs):
        """Advance the wrapped shells while retaining smoothed force lookup."""
        return self.shells.step(*args, **kwargs)

    def crossing_timestep(self, safety_factor=0.1):
        """Return the wrapped shell crossing limit for hydro timestep control."""
        return self.shells.crossing_timestep(safety_factor=safety_factor)

    @property
    def last_substep_count(self):
        return self.shells.last_substep_count

    @property
    def last_crossing_event_count(self):
        return self.shells.last_crossing_event_count

    @property
    def last_origin_reflection_count(self):
        return self.shells.last_origin_reflection_count

    @property
    def total_crossing_event_count(self):
        return self.shells.total_crossing_event_count

    @property
    def total_origin_reflection_count(self):
        return self.shells.total_origin_reflection_count

    def gravitating_enclosed_mass(
        self,
        radius_comoving_code=None,
        *,
        include_shell_mass_with_fixed=False,
    ):
        if radius_comoving_code is None:
            return self.shells.gravitating_enclosed_mass(
                radius_comoving_code,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            )
        radius_shell_comoving_code = np.asarray(self.shells.radius, dtype=float)
        shell_enclosed = np.asarray(
            self.shells.gravitating_enclosed_mass(
                radius_shell_comoving_code,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            ),
            dtype=float,
        )
        total = float(np.sum(self.shells.mass))
        if self.shells.fixed_enclosed_mass is not None:
            total += float(self.shells.fixed_enclosed_mass)
        outer_radius = radius_shell_comoving_code[-1] + 0.5 * (
            radius_shell_comoving_code[-1] - radius_shell_comoving_code[-2]
        )
        interpolation_radius = np.concatenate(([0.0], radius_shell_comoving_code, [outer_radius]))
        interpolation_mass = np.concatenate(([0.0], shell_enclosed, [total]))
        requested = np.asarray(radius_comoving_code, dtype=float)
        return np.interp(
            requested**3,
            interpolation_radius**3,
            interpolation_mass,
            left=0.0,
            right=total,
        )
