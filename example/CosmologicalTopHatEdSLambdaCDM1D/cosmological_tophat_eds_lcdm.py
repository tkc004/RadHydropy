"""Compare cosmological dark-matter top-hat trajectories: EdS versus LCDM."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PROJECT_ROOT / "example" / "CosmologicalVirialShock1D"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.cosmology import EinsteinDeSitter as CodeEdS
from radhydropy.cosmology import LambdaCDM as CodeLambdaCDM
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits, _gravitational_constant_code
from radhydropy.example_config import load_example_parameters
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
    """Run the established EdS calibration unchanged before the comparison."""
    config = EXAMPLE_ROOT / "cosmological_dark_matter_correlation_z100.yaml"
    runparams, icparams = load_example_parameters(config)
    runparams = dict(runparams)
    runparams["savedir"] = str(OUTPUT_ROOT / "reference")
    reference_units = CodeUnits.from_mapping(runparams["CodeUnits"])
    cosmology = CodeEdS.from_code_units(
        reference_units,
        t_ref=float(runparams["cosmology_t_ref"]),
        a_ref=float(runparams["cosmology_a_ref"]),
    )
    reference_example.run_lagrangian_top_hat(
        runparams, icparams, reference_units, cosmology,
    )


def reference_step(radius, velocity, dt, a_start, a_end, g_code, mass, rho_comoving):
    """Match the staggered leapfrog update in CosmologicalVirialShock1D."""
    def acceleration(r, a):
        background_mass = 4.0 * np.pi / 3.0 * rho_comoving * r**3
        return -g_code * a * (mass - background_mass) / max(r**2, 1.0e-30)

    velocity_half = velocity + 0.5 * dt * acceleration(radius, a_start)
    radius_new = radius + dt * velocity_half
    return radius_new, velocity_half + 0.5 * dt * acceleration(radius_new, a_end)


def analytic_turnaround(t_initial, radius, velocity, cosmology, g_code, mass):
    """Locate the proper-radius turnaround from the continuous top-hat ODE."""
    a_initial = float(cosmology.scale_factor(t_initial))
    h_initial = float(cosmology.hubble(t_initial))
    proper_radius = a_initial * radius
    proper_velocity = h_initial * proper_radius + velocity / a_initial
    hubble_ref = float(cosmology.hubble(cosmology.t_ref))
    lambda_acceleration = float(getattr(cosmology, "omega_lambda", 0.0)) * hubble_ref**2

    def rhs(time, state):
        r, v = state
        return v, -g_code * mass / max(r**2, 1.0e-30) + lambda_acceleration * r

    def turnaround_event(time, state):
        return state[1]

    turnaround_event.direction = -1.0
    turnaround_event.terminal = True
    solution = solve_ivp(
        rhs, (t_initial, cosmology.t_ref), (proper_radius, proper_velocity),
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
    tau_i = float(cosmology.supercomoving_time(ti))
    tau_f = float(cosmology.supercomoving_time(tf))
    g_code = _gravitational_constant_code(code_units)
    delta = initial_overdensity
    _, ai_code, hi_code = cosmology.background_state_from_supercomoving(tau_i)
    rho_comoving = float(cosmology.background_density(ti)) * ai**3
    mass = target_mass
    radius = (mass / ((4.0 * np.pi / 3.0) * rho_comoving * (1.0 + delta))) ** (1.0 / 3.0)
    velocity = -ai_code**2 * hi_code * delta * radius / 3.0
    shell = DarkMatterShells(
        radius=[radius], velocity=[velocity], mass=[mass],
        fixed_enclosed_mass=mass, code_units=code_units,
    )
    # The EdS value reproduces the established reference figure.  LCDM uses
    # a larger step because its supercomoving inversion is numerical.
    timestep = 0.0005 if code_class is CodeEdS else 0.05
    tau = tau_i
    direct_radius, direct_velocity = radius, velocity
    history = [(tau, ti, ai_code, ai_code * radius, ai_code * direct_radius)]
    while tau < tau_f - 1.0e-14:
        dt = min(timestep, tau_f - tau)
        cosmic_start, a_start, _ = cosmology.background_state_from_supercomoving(tau)
        cosmic_end, a_end, _ = cosmology.background_state_from_supercomoving(tau + dt)
        shell.step(
            dt, background_enclosed_mass=lambda r: 4.0 * np.pi / 3.0 * rho_comoving * np.asarray(r)**3,
            scale_factor=a_start, scale_factor_end=a_end, cosmological=True,
        )
        direct_radius, direct_velocity = reference_step(
            direct_radius, direct_velocity, dt, a_start, a_end,
            g_code, mass, rho_comoving,
        )
        tau += dt
        history.append((tau, cosmic_end, a_end, a_end * shell.radius[0], a_end * direct_radius))
    history = np.asarray(history)
    maximum_error = float(np.max(np.abs(history[:, 3] - history[:, 4])))
    analytic = analytic_turnaround(ti, radius, velocity, cosmology, g_code, mass)
    if code_class is CodeEdS:
        collapse_time = ti * (1.686 / delta) ** 1.5
        analytic_time = 0.5 * collapse_time
        analytic_density = float(cosmology.background_density(analytic_time))
        analytic_radius = (
            mass / ((4.0 * np.pi / 3.0) * (9.0 * np.pi**2 / 16.0) * analytic_density)
        ) ** (1.0 / 3.0)
        print(
            f"{label}: EdS closed-form turnaround t={analytic_time:.12g}, "
            f"r={analytic_radius:.12g}"
        )
    _, final_a, final_h = cosmology.background_state_from_supercomoving(tau)
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
        cosmic_time = history[:, 1] - big_bang_time
        if label.startswith("EdS"):
            line, = axis.plot(
                cosmic_time, history[:, 3], linestyle="-", marker="o",
                markerfacecolor="none", markersize=5.0,
                markevery=max(1, len(cosmic_time) // 10),
                zorder=3,
                label=f"{label} RadHydropy (circles)",
            )
        else:
            line, = axis.plot(
                cosmic_time, history[:, 3], linewidth=2.0,
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
            cosmic_time, np.maximum(np.abs(history[:, 3] - history[:, 4]), 1.0e-18),
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
