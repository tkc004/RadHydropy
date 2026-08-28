r"""Numerical reference solver for Bertschinger (1985), equation (4.1)."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


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


def first_post_centre_apocentre(solution):
    """Return ``(xi, lambda)`` at the first post-centre apocentre.

    The first negative-to-positive ``lambda_prime`` crossing marks the
    outgoing branch after centre passage.  The following positive-to-negative
    crossing is the first local maximum of ``lambda`` and therefore satisfies
    the splashback criterion.
    """
    xi = np.asarray(solution.xi, dtype=float)
    lam = np.asarray(solution.lam, dtype=float)
    lam_prime = np.asarray(solution.lam_prime, dtype=float)
    if xi.size < 3 or not (xi.size == lam.size == lam_prime.size):
        raise ValueError('invalid Bertschinger shell solution')

    outbound = np.flatnonzero(
        (lam_prime[:-1] <= 0.0) & (lam_prime[1:] >= 0.0))
    if not outbound.size:
        raise RuntimeError('solution contains no post-centre outbound branch')
    start = int(outbound[0] + 1)
    apocentre = np.flatnonzero(
        (lam_prime[start:-1] >= 0.0) & (lam_prime[start + 1:] < 0.0))
    if not apocentre.size:
        raise RuntimeError('solution contains no post-centre apocentre')
    index = int(start + apocentre[0])
    second_derivative = ((lam_prime[index + 1] - lam_prime[index]) /
                         (xi[index + 1] - xi[index]))
    if not second_derivative < 0.0:
        raise RuntimeError('candidate apocentre has non-negative curvature')
    root = brentq(
        lambda value: np.interp(value, xi[index:index + 2],
                                 lam_prime[index:index + 2]),
        float(xi[index]), float(xi[index + 1]),
    )
    return float(root), float(np.interp(root, xi, lam))


def first_outer_caustic(solution, turnaround_exponent=8.0 / 9.0):
    """Return the first fixed-time similarity caustic ``(xi, lambda)``.

    A shell that turned around at ``xi=0`` has proper radius proportional to
    ``exp(-alpha*xi) * lambda(xi)`` when compared at one fixed observation
    time, where ``alpha`` is the turnaround-radius exponent.  The envelope
    therefore satisfies ``lambda_prime = alpha * lambda``.  The first
    positive-to-negative crossing after centre passage is the outer caustic.
    """
    xi = np.asarray(solution.xi, dtype=float)
    lam = np.asarray(solution.lam, dtype=float)
    lam_prime = np.asarray(solution.lam_prime, dtype=float)
    alpha = float(turnaround_exponent)
    if alpha <= 0.0:
        raise ValueError('turnaround exponent must be positive')
    outbound = np.flatnonzero(
        (lam_prime[:-1] <= 0.0) & (lam_prime[1:] >= 0.0))
    if not outbound.size:
        raise RuntimeError('solution contains no post-centre outbound branch')
    start = int(outbound[0] + 1)
    envelope_derivative = lam_prime - alpha * lam
    caustic = np.flatnonzero(
        (envelope_derivative[start:-1] >= 0.0) &
        (envelope_derivative[start + 1:] < 0.0))
    if not caustic.size:
        raise RuntimeError('solution contains no post-centre outer caustic')
    index = int(start + caustic[0])
    root = brentq(
        lambda value: np.interp(value, xi[index:index + 2],
                                 envelope_derivative[index:index + 2]),
        float(xi[index]), float(xi[index + 1]),
    )
    shell_lambda = float(np.interp(root, xi, lam))
    return float(root), float(np.exp(-alpha * root) * shell_lambda)


def solve_eq41_self_similar(xi_end=5.0, points=8192,
                            similarity_exponent=1.0,
                            centre_match_lambda=2.0e-3,
                            centre_matching_velocity=2.30):
    r"""Solve the collisionless Bertschinger trajectory through crossings.

    Bertschinger's mass closure is

    ``M(lambda) = sum_i (-1)**(i-1) exp(-(2*s/3)*xi_i)``.

    The trajectory is integrated branch by branch in phase space.  For every
    radius, all previously completed monotonic branches are interpolated to
    obtain their crossing times and the enclosed mass is evaluated as

    ``M(lambda) = M_ta * sum((-1)**i * exp(-(2*s/3)*xi_i))``.

    The physical centre is singular in the radial equation.  We therefore
    stop at ``centre_match_lambda`` and restart the outgoing branch with the
    finite phase-space matching velocity ``centre_matching_velocity``.  This
    is a controlled lambda-to-zero asymptotic matching condition, rather than
    reflecting the divergent velocity returned by a finite-radius cutoff.
    The matching parameters are recorded by the example driver and should be
    kept fixed when comparing resolutions.
    """
    if (xi_end <= 0.0 or points < 2 or similarity_exponent <= 0.0
            or not 0.0 < centre_match_lambda < 1.0
            or centre_matching_velocity <= 0.0):
        raise ValueError('invalid self-similar ODE parameters')

    # M is normalized by the EdS background mass inside r_ta.  The mass of a
    # radial shell at turnaround is larger than that background mass by this
    # factor; using M(1)=1 with the 2/9 force coefficient mixes conventions.
    turnaround_mass_normalization = 9.0 * np.pi**2 / 16.0
    branches = []
    max_step = float(xi_end) / max(1000, 2 * int(points))

    def branch_roots(radius):
        roots = []
        for branch in branches:
            if branch['minimum'] <= radius <= branch['maximum']:
                roots.append(float(branch['time_of_radius'](radius)))
        return roots

    def rhs(time, state):
        radius, radial_velocity = state
        radius = max(float(radius), centre_match_lambda)
        roots = branch_roots(radius)
        roots.append(float(time))
        roots.sort()
        mass = turnaround_mass_normalization * sum(
            (-1.0) ** index
            * np.exp(-2.0 * similarity_exponent * root / 3.0)
            for index, root in enumerate(roots))
        return [radial_velocity,
                -7.0 / 9.0 * radial_velocity + 8.0 / 81.0 * radius
                - 2.0 / 9.0 * mass / radius**2]

    def centre_event(time, state):
        return state[0] - centre_match_lambda
    centre_event.terminal = True
    centre_event.direction = -1

    def apocentre_event(time, state):
        return state[1]
    apocentre_event.terminal = True
    apocentre_event.direction = -1

    def save_branch(solution):
        # Every branch is monotonic in radius.  PCHIP avoids spurious extrema
        # when a crossing time is requested by a neighbouring branch.
        from scipy.interpolate import PchipInterpolator

        order = np.argsort(solution.y[0])
        radius = solution.y[0][order]
        time = solution.t[order]
        unique = np.concatenate(([True], np.diff(radius) > 1.0e-12))
        if np.count_nonzero(unique) < 2:
            # At the resolution cutoff, a rapidly shrinking late branch can
            # be represented by a single solver point.  It carries no new
            # radial interval and is safely omitted from the closure.
            return
        branch = {
            'minimum': float(radius[unique][0]),
            'maximum': float(radius[unique][-1]),
            'time_of_radius': PchipInterpolator(
                radius[unique], time[unique], extrapolate=False),
        }
        branches.append(branch)

    output_time = []
    output_state = []
    time = 0.0
    state = np.array([1.0, -8.0 / 9.0])
    outbound = False
    while time < xi_end - 1.0e-12:
        event = apocentre_event if outbound else centre_event
        solution = solve_ivp(
            rhs, (time, float(xi_end)), state, events=event,
            rtol=2.0e-10, atol=1.0e-12, max_step=max_step, method='RK45')
        if solution.t.size == 0:
            raise RuntimeError('empty Bertschinger phase-space branch')
        output_time.extend(solution.t.tolist())
        output_state.extend(solution.y.T.tolist())
        save_branch(solution)
        time = float(solution.t[-1])
        state = solution.y[:, -1].copy()
        if time >= xi_end - 1.0e-12:
            break
        if not solution.t_events[0].size:
            raise RuntimeError('Bertschinger branch did not reach its event')
        if outbound:
            # Start the next inward branch just past apocentre.
            state[1] = -1.0e-8
            outbound = False
        else:
            # The divergent finite-cutoff velocity is discarded here.  The
            # matched outgoing branch has a finite phase-space launch speed.
            state[0] = centre_match_lambda
            state[1] = float(centre_matching_velocity)
            outbound = True

    output_time = np.asarray(output_time)
    output_state = np.asarray(output_state)
    unique = np.concatenate(([True], np.diff(output_time) > 1.0e-12))
    output_time = output_time[unique]
    output_state = output_state[unique]
    sample_time = np.linspace(0.0, float(xi_end), int(points))
    output_lam = np.interp(sample_time, output_time, output_state[:, 0])
    output_velocity = np.interp(sample_time, output_time, output_state[:, 1])
    output_mass = np.empty_like(sample_time)
    for index, radius in enumerate(output_lam):
        roots = branch_roots(float(radius))
        roots.sort()
        output_mass[index] = turnaround_mass_normalization * sum(
            (-1.0) ** root_index
            * np.exp(-2.0 * similarity_exponent * root / 3.0)
            for root_index, root in enumerate(roots))
    return BertschingerShellSolution(sample_time, output_lam,
                                     output_velocity, output_mass)
