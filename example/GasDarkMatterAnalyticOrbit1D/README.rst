Analytic Gas--Dark-Matter Orbit Benchmark
==========================================

This benchmark freezes a uniform gas background and a central dark-matter
mass, then evolves one negligible-mass dark-matter shell. The enclosed mass is

.. math::

   M(<r)=M_0+\frac{4\pi}{3}\rho_g r^3,

so the shell obeys the analytic ODE

.. math::

   \ddot r=-\frac{G M(<r)}{(r+a)^2}+\frac{j^2}{r^3}.

The shell integrator is compared against a high-accuracy reference solution
of this analytic time-dependent ODE. This is a fixed-background benchmark,
not a mutual gas--dark-matter evolution test.

Run with::

   python gas_dark_matter_analytic_orbit1d.py \
       --config gas_dark_matter_analytic_orbit1d.yaml
