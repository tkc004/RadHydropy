"""Compare cosmological dark-matter top-hat trajectories: EdS versus LCDM."""

from pathlib import Path
import copy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = PROJECT_ROOT / "example" / "CosmologicalVirialShock1D"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(REFERENCE_ROOT))

from radhydropy.cosmology import EinsteinDeSitter as CodeEdS
from radhydropy.cosmology import LambdaCDM as CodeLambdaCDM
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits, quantity_to_value, _gravitational_constant_code
import example_utils as eu
import cosmological_dark_matter_only as reference_example


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
TARGET_MASS = 1000.0  # 10^13 Msun in the reference code units
INITIAL_OVERDENSITY = 0.03
LENGTH_CM = 3.0856775814913673e21
VELOCITY_CM_S = 1.0e5
MASS_G = 1.98847e43


def units():
    return CodeUnits.from_mapping({
        "name": "cosmological_tophat_eds_lcdm_unit_system",
        "InternalUnitSystem": {
        "UnitMass_in_cgs": MASS_G,
            "UnitLength_in_cgs": LENGTH_CM,
            "UnitVelocity_in_cgs": VELOCITY_CM_S,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    })


def reproduce_reference():
    """Run the established EdS calibration using the complete config."""
    config_filename = EXAMPLE_ROOT / "cosmological_dark_matter_correlation_z100.yaml"
    config = eu.load_nested_example_config(config_filename)
    reference_config = copy.deepcopy(config)
    reference_config["par"]["output"]["directory"] = str(OUTPUT_ROOT / "reference")
    reference_units = CodeUnits.from_mapping(
        reference_config["par"]["units"]["CodeUnits"]
    )
    cosmology = CodeEdS.from_code_units(
        reference_units,
        t_ref=quantity_to_value(
            reference_config["par"]["cosmology"]["cosmology_t_ref"],
            reference_units.time_unit,
        ),
        a_ref=float(reference_config["par"]["cosmology"]["cosmology_a_ref"]),
    )
    # These runtime-only objects cannot be represented in YAML. Attach them
    # at this call site while preserving the complete nested config boundary.
    reference_config["_code_unit_system"] = reference_units
    reference_config["_cosmology"] = cosmology
    reference_example.run_lagrangian_top_hat(reference_config)


def reference_step(
    radius_comoving_code,
    vel_supercomoving_code,
    dt_supercomoving_code,
    a_start_dimensionless,
    a_end_dimensionless,
    gravity_constant_code,
    mass_comoving_code,
    rho_comoving_code,
):
    """Match the staggered leapfrog update in CosmologicalVirialShock1D."""
    def acceleration(r, a):
        background_mass_comoving_code = 4.0 * np.pi / 3.0 * rho_comoving_code * r**3
        return -gravity_constant_code * a * (
            mass_comoving_code - background_mass_comoving_code
        ) / max(r**2, 1.0e-30)

    vel_half_supercomoving_code = vel_supercomoving_code + 0.5 * dt_supercomoving_code * acceleration(
        radius_comoving_code, a_start_dimensionless
    )
    radius_new_comoving_code = radius_comoving_code + dt_supercomoving_code * vel_half_supercomoving_code
    return radius_new_comoving_code, vel_half_supercomoving_code + 0.5 * dt_supercomoving_code * acceleration(
        radius_new_comoving_code, a_end_dimensionless
    )


def analytic_turnaround(
    time_cosmic_code,
    radius_comoving_code,
    vel_supercomoving_code,
    cosmology,
    gravity_constant_code,
    mass_comoving_code,
):
    """Locate the proper-radius turnaround from the continuous top-hat ODE."""
    a_initial_dimensionless = float(cosmology.scale_factor(time_cosmic_code))
    hubble_initial_code = float(cosmology.hubble(time_cosmic_code))
    radius_proper_code = a_initial_dimensionless * radius_comoving_code
    vel_proper_code = (
        hubble_initial_code * radius_proper_code
        + vel_supercomoving_code / a_initial_dimensionless
    )
    hubble_ref = float(cosmology.hubble(cosmology.t_ref))
    lambda_acceleration = float(getattr(cosmology, "omega_lambda", 0.0)) * hubble_ref**2

    def rhs(time_cosmic_code, state):
        r, v = state
        return v, -gravity_constant_code * mass_comoving_code / max(r**2, 1.0e-30) + lambda_acceleration * r

    def turnaround_event(time_cosmic_code, state):
        return state[1]

    turnaround_event.direction = -1.0
    turnaround_event.terminal = True
    solution = solve_ivp(
        rhs, (time_cosmic_code, cosmology.t_ref), (radius_proper_code, vel_proper_code),
        rtol=1.0e-11, atol=1.0e-11, events=turnaround_event, max_step=0.01,
    )
    if not solution.t_events[0].size:
        return None
    return float(solution.t_events[0][0]), float(solution.y_events[0][0][0])


def run_case(label, code_class, omega_m, omega_lambda, final_scale_factor, target_mass, initial_overdensity):
    code_units = units()
    # Start at z=100 and continue past z=0 so both turnarounds are visible.
    ai, af = 1.0 / 101.0, final_scale_factor
    t_ref = 14.4
    hubble_ref = 2.0 / (3.0 * t_ref)
    if code_class is CodeEdS:
        cosmology = code_class.from_code_units(code_units, t_ref=t_ref, a_ref=1.0)
    else:
        cosmology = code_class.from_code_units(
            code_units, t_ref=t_ref, a_ref=1.0,
            omega_m=omega_m, omega_lambda=omega_lambda,
            hubble_ref=hubble_ref / np.sqrt(omega_m),
        )
    ti = float(cosmology.cosmic_time_from_scale_factor(ai))
    tf = float(cosmology.cosmic_time_from_scale_factor(af))
    tau_initial_supercomoving_code = float(cosmology.supercomoving_time(ti))
    tau_final_supercomoving_code = float(cosmology.supercomoving_time(tf))
    gravity_constant_code = _gravitational_constant_code(code_units)
    overdensity_dimensionless = initial_overdensity
    _, ai_code, hubble_initial_code = cosmology.background_state_from_supercomoving(
        tau_initial_supercomoving_code
    )
    rho_comoving_code = float(cosmology.background_density(ti)) * ai**3
    mass_comoving_code = target_mass
    radius_comoving_code = (
        mass_comoving_code
        / ((4.0 * np.pi / 3.0) * rho_comoving_code * (1.0 + overdensity_dimensionless))
    ) ** (1.0 / 3.0)
    vel_supercomoving_code = (
        -ai_code**2
        * hubble_initial_code
        * overdensity_dimensionless
        * radius_comoving_code
        / 3.0
    )
    shell = DarkMatterShells(
        [radius_comoving_code], [vel_supercomoving_code], [mass_comoving_code],
        fixed_enclosed_mass=mass_comoving_code, code_units=code_units,
    )
    # The EdS value reproduces the established reference figure.  LCDM uses
    # a larger step because its supercomoving inversion is numerical.
    timestep_supercomoving_code = 0.0005 if code_class is CodeEdS else 0.05
    tau_supercomoving_code = tau_initial_supercomoving_code
    direct_radius_comoving_code = radius_comoving_code
    direct_vel_supercomoving_code = vel_supercomoving_code
    history = [(
        tau_supercomoving_code, ti, ai_code, ai_code * radius_comoving_code,
        ai_code * direct_radius_comoving_code,
    )]
    while tau_supercomoving_code < tau_final_supercomoving_code - 1.0e-14:
        dt_supercomoving_code = min(
            timestep_supercomoving_code,
            tau_final_supercomoving_code - tau_supercomoving_code,
        )
        _, a_start_dimensionless, _ = cosmology.background_state_from_supercomoving(
            tau_supercomoving_code
        )
        cosmic_end, a_end_dimensionless, _ = cosmology.background_state_from_supercomoving(
            tau_supercomoving_code + dt_supercomoving_code
        )
        shell.step(
            dt_supercomoving_code,
            background_enclosed_mass=lambda radius_shell_comoving_code: (
                4.0 * np.pi / 3.0 * rho_comoving_code
                * np.asarray(radius_shell_comoving_code)**3
            ),
            scale_factor=a_start_dimensionless,
            scale_factor_end=a_end_dimensionless,
            cosmological=True,
        )
        direct_radius_comoving_code, direct_vel_supercomoving_code = reference_step(
            direct_radius_comoving_code,
            direct_vel_supercomoving_code,
            dt_supercomoving_code,
            a_start_dimensionless,
            a_end_dimensionless,
            gravity_constant_code,
            mass_comoving_code,
            rho_comoving_code,
        )
        tau_supercomoving_code += dt_supercomoving_code
        history.append((
            tau_supercomoving_code,
            cosmic_end,
            a_end_dimensionless,
            a_end_dimensionless * shell.radius[0],
            a_end_dimensionless * direct_radius_comoving_code,
        ))
    history = np.asarray(history)
    maximum_error = float(np.max(np.abs(history[:, 3] - history[:, 4])))
    analytic = analytic_turnaround(
        ti,
        radius_comoving_code,
        vel_supercomoving_code,
        cosmology,
        gravity_constant_code,
        mass_comoving_code,
    )
    if code_class is CodeEdS:
        collapse_time_cosmic_code = ti * (1.686 / overdensity_dimensionless) ** 1.5
        analytic_time_cosmic_code = 0.5 * collapse_time_cosmic_code
        analytic_density_comoving_code = float(cosmology.background_density(analytic_time_cosmic_code))
        analytic_radius_proper_code = (
            mass_comoving_code
            / ((4.0 * np.pi / 3.0) * (9.0 * np.pi**2 / 16.0) * analytic_density_comoving_code)
        ) ** (1.0 / 3.0)
        print(
            f"{label}: EdS closed-form turnaround t={analytic_time_cosmic_code:.12g}, "
            f"r={analytic_radius_proper_code:.12g}"
        )
    _, final_a, final_h = cosmology.background_state_from_supercomoving(
        tau_supercomoving_code
    )
    print(f"{label}: a_final={final_a:.12g}, H_final={final_h:.12g}, "
          f"final_proper_radius={history[-1][3]:.12g}, reference_radius={history[-1][4]:.12g}, "
          f"max_radius_error={maximum_error:.6e}")
    if maximum_error > 2.0e-5:
        raise RuntimeError(f"{label}: RadHydropy disagrees with reference integration")
    if analytic is None:
        print(f"{label}: no turnaround before a=1")
    else:
        print(f"{label}: analytic turnaround t={analytic[0]:.12g}, r={analytic[1]:.12g}")
        analytic = (analytic[0], analytic[1], float(cosmology.scale_factor(analytic[0])))
    return history, analytic, float(getattr(cosmology, "_big_bang_time", 0.0))


def make_comparison(target_mass, filename, initial_overdensity=INITIAL_OVERDENSITY, final_scale_factor=0.5):
    cases = [
        ("EdS", CodeEdS, 1.0, 0.0, final_scale_factor),
        ("LCDM_0p3_0p7", CodeLambdaCDM, 0.3, 0.7, final_scale_factor),
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure, (axis, error_axis) = plt.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    for label, code_class, omega_m, omega_lambda, final_scale_factor in cases:
        history, analytic, big_bang_time = run_case(
            label, code_class, omega_m, omega_lambda, final_scale_factor, target_mass,
            initial_overdensity,
        )
        time_cosmic_code = history[:, 1] - big_bang_time
        if label.startswith("EdS"):
            line, = axis.plot(
                time_cosmic_code, history[:, 3], linestyle="-", marker="o",
                markerfacecolor="none", markersize=5.0,
                markevery=max(1, len(time_cosmic_code) // 10),
                zorder=3,
                label=f"{label} RadHydropy (circles)",
            )
        else:
            line, = axis.plot(
                time_cosmic_code, history[:, 3], linewidth=2.0,
                label=f"{label} RadHydropy (line)",
            )
        if analytic is not None:
            analytic_time, analytic_radius, analytic_a = analytic
            axis.plot(
                analytic_time - big_bang_time, analytic_radius, "*",
                markersize=11.0, markeredgecolor="black", zorder=4,
                label=f"{label} analytic turnaround",
            )
        error_axis.plot(
            time_cosmic_code, np.maximum(np.abs(history[:, 3] - history[:, 4]), 1.0e-18),
            color=line.get_color(), label=label,
        )
    axis.set_xlabel("cosmic age since Big Bang [code units]")
    axis.set_ylabel("proper top-hat radius [kpc]")
    axis.set_title("Dark-matter top-hat: EdS versus ΛCDM")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    error_axis.set_xlabel("cosmic age since Big Bang [code units]")
    error_axis.set_ylabel("|RadHydropy − reference|")
    error_axis.set_yscale("log")
    error_axis.grid(alpha=0.25, which="both")
    error_axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / filename, dpi=180)
    plt.close(figure)


def run():
    reproduce_reference()
    make_comparison(TARGET_MASS, "CosmologicalTopHatEdSVsLCDM.jpg")
    make_comparison(100.0, "CosmologicalTopHatEdSVsLCDM_1e12Msun.jpg")
    make_comparison(
        100.0, "CosmologicalTopHatEdSVsLCDM_1e12Msun_delta0p02.jpg",
        initial_overdensity=0.02, final_scale_factor=0.8,
    )


if __name__ == "__main__":
    run()
