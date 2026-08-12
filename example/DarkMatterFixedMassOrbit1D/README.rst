Fixed-Mass Dark-Matter Orbit
============================

This benchmark evolves one negligible-mass dark-matter shell in the fixed
spherical mass ``M``. Its radial equation is

.. math::

   \ddot r=-\frac{GM}{(r+a)^2}+\frac{j^2}{r^3}.

The conserved effective-potential energy is

.. math::

   E=\frac{1}{2}\dot r^2-\frac{GM}{r+a}+\frac{j^2}{2r^2}.

The example integrates this same two-dimensional ODE at high accuracy as the
reference trajectory and compares radius and energy against the shell
integrator. The shell class also supports the implicit quadrature
``dt = dr / sqrt(2(E-Phi_eff))``; the ODE reference is used here to handle
turning points robustly.

Run with::

   python dark_matter_fixed_mass_orbit1d.py \
       --config dark_matter_fixed_mass_orbit1d.yaml
