"""Standalone Bertschinger (1985) collisional-gas similarity solution.

This module deliberately has no dependency on :mod:`radhydropy`.  It solves
the pressureless exterior from the spherical-collapse parametric solution,
applies the strong accretion-shock jump, and integrates the gas solution
inwards until the regular central boundary condition is met.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ALPHA = 8.0 / 9.0
GAMMA = 5.0 / 3.0
TURNAROUND_MASS = 9.0 * np.pi**2 / 16.0


@dataclass(frozen=True)
class GasSimilaritySolution:
    """Dimensionless exterior, interior, and shock-matching solution."""

    lambda_out: np.ndarray
    density_out: np.ndarray
    velocity_out: np.ndarray
    mass_out: np.ndarray
    lambda_in: np.ndarray
    density_in: np.ndarray
    velocity_in: np.ndarray
    pressure_in: np.ndarray
    mass_in: np.ndarray
    shock_lambda: float


def _collapse_parametric(theta):
    """Return ``tau=t/t_ta`` and its derivative for a cold shell."""
    tau = (theta - np.sin(theta)) / np.pi
    dtau = (1.0 - np.cos(theta)) / np.pi
    return tau, dtau


def exterior_solution(lambda_values, theta_points=20000):
    """Solve the cold exterior using the spherical-collapse parameter ``theta``."""
    theta = np.linspace(1.0e-5, 2.0 * np.pi - 1.0e-6, int(theta_points))
    tau, dtau = _collapse_parametric(theta)
    lam = 0.5 * (1.0 - np.cos(theta)) * tau**(-ALPHA)
    # v/(r_ta/t), obtained by differentiating r=A(1-cos(theta)).
    vel = (np.sin(theta) * tau) / (2.0 * dtau) * tau**(-ALPHA)
    mass = TURNAROUND_MASS * tau**(-2.0 / 3.0)
    order = np.argsort(lam)
    lam, mass, vel = lam[order], mass[order], vel[order]
    density = np.gradient(mass, lam) / (3.0 * lam**2)
    requested = np.asarray(lambda_values, dtype=float)
    if np.any((requested < lam[0]) | (requested > lam[-1])):
        raise ValueError('requested exterior lambda is outside the tabulated range')
    return tuple(np.interp(requested, lam, values) for values in (density, vel, mass))


def _gas_rhs(lam, state, gamma=GAMMA):
    density, velocity, pressure, mass = state
    if density <= 0.0 or pressure <= 0.0 or lam <= 0.0:
        return np.full(4, np.nan)
    q = velocity - ALPHA * lam
    # The unknowns are (d ln D/d lambda, dV/d lambda,
    # d ln P/d lambda).  Keeping logarithmic derivatives here avoids the
    # density/pressure scale factors leaking into the energy equation.
    matrix = np.array([
        [q, 1.0, 0.0],
        [0.0, q, pressure / density],
        [0.0, gamma, q],
    ])
    rhs = np.array([
        2.0 - 2.0 * velocity / lam,
        -(ALPHA - 1.0) * velocity - 2.0 * mass / (9.0 * lam**2),
        4.0 - 2.0 * ALPHA - 2.0 * gamma * velocity / lam,
    ])
    log_density_prime, velocity_prime, log_pressure_prime = np.linalg.solve(matrix, rhs)
    return np.array([
        density * log_density_prime,
        velocity_prime,
        pressure * log_pressure_prime,
        3.0 * lam**2 * density,
    ])


def shock_jump(exterior_state, shock_lambda, gamma=GAMMA):
    """Apply the strong-shock Rankine--Hugoniot conditions."""
    density, velocity, mass = exterior_state
    relative_velocity = velocity - ALPHA * shock_lambda
    density_post = (gamma + 1.0) / (gamma - 1.0) * density
    velocity_post = ALPHA * shock_lambda + (gamma - 1.0) / (gamma + 1.0) * relative_velocity
    pressure_post = 2.0 / (gamma + 1.0) * density * relative_velocity**2
    return np.array([density_post, velocity_post, pressure_post, mass])


def _integrate_inside(shock_lambda, lambda_min, gamma,
                      central_density_limit=1.0e6):
    exterior = exterior_solution(np.array([shock_lambda]))
    postshock = shock_jump((exterior[0][0], exterior[1][0], exterior[2][0]), shock_lambda, gamma)

    def density_limit(lam, state):
        return state[0] - central_density_limit
    density_limit.terminal = True
    density_limit.direction = 1

    solution = solve_ivp(
        lambda lam, state: _gas_rhs(lam, state, gamma),
        (shock_lambda, lambda_min), postshock, rtol=2.0e-9, atol=1.0e-11,
        events=density_limit, max_step=shock_lambda / 400.0, method='RK45',
    )
    if not solution.success and not solution.t_events[0].size:
        raise RuntimeError('post-shock integration failed')
    if np.any(~np.isfinite(solution.y)):
        raise RuntimeError('post-shock integration produced non-finite values')
    return solution


def shoot_shock_lambda(bracket=(0.3388, 0.3391), gamma=GAMMA,
                       central_density_limit=1.0e6):
    """Shoot on ``lambda_s`` for the regular transonic branch."""
    def residual(shock_lambda):
        solution = _integrate_inside(
            shock_lambda, 1.0e-6, gamma, central_density_limit)
        if not solution.t_events[0].size:
            raise RuntimeError('shooting integration did not reach the central asymptote')
        return float(solution.y[1, -1])

    left, right = map(float, bracket)
    return brentq(residual, left, right, xtol=2.0e-10, rtol=2.0e-10)


def solve_bertschinger_gas(shock_lambda=None, points=2048,
                           lambda_min=1.0e-5, gamma=GAMMA):
    """Return the exterior/interior solution matched at a strong shock.

    Bertschinger's ``gamma=5/3`` solution has ``lambda_s ~= 0.339``.  The
    If no shock position is supplied, it is selected by transonic shooting.
    """
    if shock_lambda is None:
        shock_lambda = shoot_shock_lambda(gamma=gamma)
    if not 0.0 < shock_lambda < 1.0 or not 0.0 < lambda_min < shock_lambda:
        raise ValueError('shock_lambda and lambda_min must be positive and ordered')
    # Keep the cold exterior long enough for finite-domain RadHydro runs to
    # compare against the asymptotic Hubble-flow branch without endpoint
    # clamping in interpolation.
    lambda_out = np.geomspace(shock_lambda, 200.0, int(points))[::-1]
    density_out, velocity_out, mass_out = exterior_solution(lambda_out)
    interior = _integrate_inside(shock_lambda, lambda_min, gamma)
    lambda_in = np.geomspace(interior.t[-1], shock_lambda, int(points))
    # solve_ivp stores the integration in decreasing lambda; interpolate in
    # increasing radius for a conventional profile output.
    density_in = np.interp(lambda_in, interior.t[::-1], interior.y[0, ::-1])
    velocity_in = np.interp(lambda_in, interior.t[::-1], interior.y[1, ::-1])
    pressure_in = np.interp(lambda_in, interior.t[::-1], interior.y[2, ::-1])
    mass_in = np.interp(lambda_in, interior.t[::-1], interior.y[3, ::-1])
    return GasSimilaritySolution(
        lambda_out=lambda_out, density_out=density_out,
        velocity_out=velocity_out, mass_out=mass_out,
        lambda_in=lambda_in, density_in=density_in,
        velocity_in=velocity_in, pressure_in=pressure_in,
        mass_in=mass_in, shock_lambda=float(shock_lambda),
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    solution = solve_bertschinger_gas()
    print('shock lambda = %.8f' % solution.shock_lambda)
    print('interior integration reached lambda = %.8e' % solution.lambda_in[0])
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), squeeze=False)
    axes = axes.ravel()
    axes[0].loglog(solution.lambda_out, solution.density_out, label='cold exterior')
    axes[0].loglog(solution.lambda_in, solution.density_in, label='hot interior')
    axes[0].set(xlabel=r'$\lambda$', ylabel=r'$D=\rho/\rho_b$')
    axes[1].semilogx(solution.lambda_out, solution.velocity_out)
    axes[1].semilogx(solution.lambda_in, solution.velocity_in)
    axes[1].set(xlabel=r'$\lambda$', ylabel=r'$V$')
    axes[2].loglog(solution.lambda_in, solution.pressure_in)
    axes[2].set(xlabel=r'$\lambda$', ylabel=r'$P$')
    axes[3].loglog(solution.lambda_out, solution.mass_out, label='cold exterior')
    axes[3].loglog(solution.lambda_in, solution.mass_in, label='hot interior')
    axes[3].set(xlabel=r'$\lambda$', ylabel=r'$M(<\lambda)$')
    for axis in axes:
        axis.axvline(solution.shock_lambda, color='k', linestyle=':', alpha=0.5)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[3].legend()
    figure.tight_layout()
    output = Path(__file__).with_name('BertschingerGasReference.jpg')
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print('figure = %s' % output)
