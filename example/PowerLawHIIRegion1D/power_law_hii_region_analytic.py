"""Analytic H II-region fronts in a core plus power-law cloud.

This is deliberately independent of :mod:`radhydropy`.  It implements the
thin ionization-front calculation used as the analytic reference in Franco,
Tenorio-Tagle & Bodenheimer (1990), ApJ, 349, 126.  The formation phase is
obtained from photon conservation.  For w <= 3/2 the usual matched D-type
expansion approximation is added, with the characteristic
R proportional to t**(4/(7-2*w)) scaling.

Run from this directory with::

    python power_law_hii_region_analytic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALPHA_B = 2.6e-13  # cm**3 s**-1, the paper's reference value
CI = 1.285e6  # cm s**-1, sound speed for T=1e4 K and mu=0.5
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def density(radius_cgs_cm, nc, rc, w):
    """Return the neutral hydrogen number density in cm**-3."""
    radius_cgs_cm = np.asarray(radius_cgs_cm, dtype=float)
    return nc * np.where(radius_cgs_cm < rc, 1.0, (radius_cgs_cm / rc) ** (-w))


def recombination_integral(radius_cgs_cm, nc, rc, w):
    """Return integral_0^R n_H(r)^2 r^2 dr in cm**-3."""
    radius_cgs_cm = np.asarray(radius_cgs_cm, dtype=float)
    core = rc**3 / 3.0
    if np.isclose(w, 1.5):
        envelope = core + rc**3 * np.log(radius_cgs_cm / rc)
    else:
        envelope = core + rc ** (2.0 * w) * (
            radius_cgs_cm ** (3.0 - 2.0 * w) - rc ** (3.0 - 2.0 * w)
        ) / (3.0 - 2.0 * w)
    result = np.where(radius_cgs_cm <= rc, radius_cgs_cm**3 / 3.0, envelope)
    return nc**2 * result


def front_speed(radius_cgs_cm, q_star, nc, rc, w):
    """Static-cloud R-type front speed from photon conservation."""
    radius_cgs_cm = np.asarray(radius_cgs_cm, dtype=float)
    available = q_star - 4.0 * np.pi * ALPHA_B * recombination_integral(
        radius_cgs_cm, nc, rc, w
    )
    return available / (4.0 * np.pi * radius_cgs_cm**2 * density(radius_cgs_cm, nc, rc, w))


def formation_front(q_star, nc, rc, w, radius_max, samples=20000):
    """Return formation-phase time and radius arrays by quadrature.

    The equation is dR/dt = [Q - recombinations]/[4 pi R^2 n(R)].
    Integrating dt/dR avoids a timestep-dependent front solver and is the
    direct thin-front analytic construction used for the reference plot.
    """
    radius = np.geomspace(max(rc * 1.0e-8, radius_max * 1.0e-10), radius_max, samples)
    # Resolve the finite Strömgren root explicitly.  A logarithmic grid can
    # otherwise jump from a positive front speed to a negative one and miss
    # the 2*c_i formation-to-expansion transition.
    available_at_max = q_star - 4.0 * np.pi * ALPHA_B * recombination_integral(
        radius_max, nc, rc, w
    )
    available_at_min = q_star - 4.0 * np.pi * ALPHA_B * recombination_integral(
        radius[0], nc, rc, w
    )
    if available_at_min > 0.0 and available_at_max < 0.0:
        lo, hi = radius[0], radius_max
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            available = q_star - 4.0 * np.pi * ALPHA_B * recombination_integral(
                mid, nc, rc, w
            )
            if available > 0.0:
                lo = mid
            else:
                hi = mid
        radius = np.geomspace(radius[0], lo * (1.0 - 1.0e-10), samples)
    speed = front_speed(radius, q_star, nc, rc, w)
    positive = speed > 0.0
    if not np.any(positive):
        raise ValueError("the source cannot ionize even the innermost radius")
    last = np.flatnonzero(positive)[-1] + 1
    radius = radius[:last]
    speed = speed[:last]
    dt_dr = 1.0 / speed
    time = np.zeros_like(radius)
    time[1:] = np.cumsum(0.5 * (dt_dr[1:] + dt_dr[:-1]) * np.diff(radius))
    return time, radius, speed


def matched_expansion(time_s, radius_w, w):
    """D-type expansion approximation for w <= 3/2.

    It is normalized to the standard Spitzer solution at w=0 and preserves
    the paper's power-law exponent.  The transition itself is not analytic in
    the paper, so the formation solution is joined continuously at R_w.
    """
    if w > 1.5:
        raise ValueError("the trapped D-type approximation is only for w <= 3/2")
    exponent = 4.0 / (7.0 - 2.0 * w)
    return radius_w * (1.0 + (7.0 - 2.0 * w) * CI * time_s / (4.0 * radius_w)) ** exponent


def champagne_expansion(time_s, radius_start, w, rc, absolute_time_s=None):
    """Return the paper's approximate champagne-phase core radius.

    Franco et al. use r ~= r_c + [1 + sqrt(3/(3-w))] c_i t for
    3/2 < w < 3.  At w=3 their
    fitted integral gives r proportional to t**(1/0.91).  For w > 3 they
    use the strong-shock approximation with delta = 2.8 + 0.55(w-3):

        r = r_c [1 + sqrt(4/(w-3)) * (delta+2-w)/2 * c_i*t/r_c]
            ** (2/(delta+2-w)).
    """
    time_s = np.asarray(time_s, dtype=float)
    if 1.5 < w < 3.0:
        velocity_factor = 1.0 + np.sqrt(3.0 / (3.0 - w))
        return radius_start + velocity_factor * CI * time_s
    if np.isclose(w, 3.0):
        if absolute_time_s is None:
            absolute_time_s = time_s
        # Franco et al. Eq. (25): the fit to the w=3 isothermal champagne
        # expansion for the expanded core.
        return 3.2 * rc * (CI * absolute_time_s / rc) ** 1.1
    delta = 2.8 + 0.55 * (w - 3.0)
    if absolute_time_s is None:
        absolute_time_s = time_s
    exponent_denominator = delta + 2.0 - w
    factor = np.sqrt(4.0 / (w - 3.0)) * exponent_denominator / 2.0
    return rc * (
        1.0 + factor * CI * absolute_time_s / rc
    ) ** (2.0 / exponent_denominator)


def calculate_front(q_star, nc, rc, w, end_time_yr):
    radius_max = 10.0 * 3.085677581e18  # 10 pc, beyond the plotted range
    time, radius, speed = formation_front(q_star, nc, rc, w, radius_max)
    end_time_s = end_time_yr * SECONDS_PER_YEAR

    # The paper defines the end of formation when the R-type front slows to
    # approximately 2 c_i.  If it never does, the cloud is density bounded in
    # the formation phase and the formation curve is returned unchanged.
    if w > 1.5:
        # Equations (24)--(26) use t measured from the start of the analytic
        # expansion. Do not shift time to an arbitrary plotting radius; the
        # y-axis lower limit simply hides the smaller-radius part.
        late_time = np.geomspace(1.0e2 * SECONDS_PER_YEAR, end_time_s, 1200)
        late_radius = champagne_expansion(
            late_time,
            rc,
            w,
            rc,
            absolute_time_s=late_time,
        )
        return late_time, late_radius, False

    if front_speed(radius[-1], q_star, nc, rc, w) > 2.0 * CI:
        formation_mask = time <= end_time_s
        return time[formation_mask], radius[formation_mask], False

    lo, hi = radius[0], radius[-1]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if front_speed(mid, q_star, nc, rc, w) > 2.0 * CI:
            lo = mid
        else:
            hi = mid
    radius_w = 0.5 * (lo + hi)

    # Reintegrate only to R_w.  Integrating all the way to the static
    # Strömgren root would include the logarithmically divergent time there.
    radius = np.geomspace(radius[0], radius_w, 12000)
    speed = front_speed(radius, q_star, nc, rc, w)
    dt_dr = 1.0 / speed
    time = np.zeros_like(radius)
    time[1:] = np.cumsum(0.5 * (dt_dr[1:] + dt_dr[:-1]) * np.diff(radius))
    time_w = time[-1]
    formation_mask = time <= min(time_w, end_time_s)
    time = time[formation_mask]
    radius = radius[formation_mask]
    if time_w > end_time_s:
        return time, radius, False
    late_time = np.linspace(time_w, end_time_s, 1200)
    late_radius = matched_expansion(late_time - time_w, radius_w, w)
    keep = time < time_w
    return np.concatenate((time[keep], late_time)), np.concatenate((radius[keep], late_radius)), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("PowerLawHIIRegion1D.jpg"))
    parser.add_argument("--end-time-myr", type=float, default=0.1)
    parser.add_argument("--exponents", type=float, nargs="+", default=[1.0, 1.5, 3.0, 5.0])
    args = parser.parse_args()

    nc = 1.0e6
    rc = 2.1e16
    q_star = 5.0e49
    end_time_yr = args.end_time_myr * 1.0e6

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for w in args.exponents:
        time, radius, trapped = calculate_front(q_star, nc, rc, w, end_time_yr)
        label = rf"$w={w:g}$"
        if not trapped and w > 1.5:
            label += " (champagne)"
        axis.plot(
            time[1:] / SECONDS_PER_YEAR,
            radius[1:],
            label=label,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("time (yr)")
    axis.set_ylabel("ionization-front radius (cm)")
    axis.set_xlim(1.0e2, 1.0e5)
    axis.set_ylim(1.0e16, 1.0e20)
    axis.set_title(r"Power-law H II regions: $n_c=10^6\,cm^{-3}$, $r_c=2.1\times10^{16}\,cm$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
