Self-Gravitating Polytrope Relaxation
=====================================

This non-cosmological example starts from a small radial-velocity perturbation
of the analytic ``n=1`` Lane--Emden sphere:

.. math::

   \rho(r)=\rho_c\frac{\sin(r/a)}{r/a},\qquad R=\pi a,

with ``gamma=2`` and ``P=K rho^2``. Gas self-gravity is enabled and the
simulation evolves with configurable velocity damping toward the hydrostatic reference state. The final output
compares density, velocity, and the normalized residual
``dP/dr + rho*g``. The transient evolution itself has no closed-form solution;
the analytic solution is the equilibrium target.

Run from this directory::

   python self_gravity_polytrope_relaxation1d.py \
       --config self_gravity_polytrope_relaxation1d.yaml
