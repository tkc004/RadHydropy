r"""Numerical reference solver for Bertschinger (1985), equation (4.1)."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


def plot_xi_lambda(solution, filename=None, axis=None, **plot_kwargs):
    """Plot similarity time ``xi`` on x and shell radius ``lambda`` on y."""
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots()
    defaults = {'linewidth': 1.5}
    defaults.update(plot_kwargs)
    axis.plot(solution.xi, solution.lam, **defaults)
    axis.set_xlabel(r'$\xi$')
    axis.set_ylabel(r'$\lambda$')
    axis.grid(alpha=0.25)
    if filename is not None:
        axis.figure.tight_layout()
        axis.figure.savefig(filename, dpi=200)
    return axis


@dataclass(frozen=True)
class BertschingerShellSolution:
    xi: np.ndarray
    lam: np.ndarray
    lam_prime: np.ndarray
    mass: np.ndarray


def solve_eq41_self_similar(xi_end=5.0, points=8192,
                            similarity_exponent=1.0):
    r"""Solve the normalized first-stream Bertschinger trajectory.

    Bertschinger's mass closure is

    ``M(lambda) = sum_i (-1)**(i-1) exp(-(2*s/3)*xi_i)``.

    Before the first centre passage there is only one stream, so the closure
    is exactly ``M(lambda(xi)) = (9*pi**2/16)*exp(-(2*s/3)*xi)``.  The
    trajectory is stopped at its first ``lambda=0`` event; later passages use
    the full alternating crossing sum and are deliberately outside this
    benchmark.
    """
    if xi_end <= 0.0 or points < 2 or similarity_exponent <= 0.0:
        raise ValueError('invalid self-similar ODE parameters')

    # M is normalized by the EdS background mass inside r_ta.  The mass of a
    # radial shell at turnaround is larger than that background mass by this
    # factor; using M(1)=1 with the 2/9 force coefficient mixes conventions.
    turnaround_mass_normalization = 9.0 * np.pi**2 / 16.0
    centre_event_radius = 1.0e-6

    def rhs(time, state):
        radius, radial_velocity = state
        safe_radius = max(radius, 1.0e-8)
        mass = (turnaround_mass_normalization
                * np.exp(-2.0 * similarity_exponent * time / 3.0))
        return [radial_velocity,
                -7.0 / 9.0 * radial_velocity + 8.0 / 81.0 * radius
                - 2.0 / 9.0 * mass / safe_radius**2]

    def centre_event(time, state):
        return state[0] - centre_event_radius
    centre_event.terminal = True
    centre_event.direction = -1
    solution = solve_ivp(
        rhs, (0.0, float(xi_end)), [1.0, -8.0 / 9.0],
        events=centre_event, dense_output=True,
        rtol=2.0e-10, atol=1.0e-12,
        max_step=float(xi_end) / (10 * int(points)), method='RK45')
    if not solution.t_events[0].size:
        raise RuntimeError('Eq. (4.1) did not reach lambda=0')
    zero_time = float(solution.t_events[0][0])
    output_xi = np.linspace(0.0, zero_time, int(points))
    output_state = solution.sol(output_xi)
    output_lam = np.maximum(output_state[0], 0.0)
    output_lam[-1] = 0.0
    output_mass = (turnaround_mass_normalization
                   * np.exp(-2.0 * similarity_exponent * output_xi / 3.0))
    return BertschingerShellSolution(output_xi, output_lam,
                                     output_state[1], output_mass)
