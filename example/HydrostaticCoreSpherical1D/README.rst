HydrostaticCoreSpherical1D
==========================

This example tests the optional pressure-supported unresolved central core.
It evolves an isothermal spherical atmosphere in the analytic gravitational
field of a fixed point mass.  The analytic density profile is

.. math::

   \rho(r) = \rho_0 \exp\left[-\frac{\Phi(r)-\Phi(r_0)}{c_s^2}\right].

The run uses ``gas_core_model: hydrostatic_fixed`` and compares the final
resolved-halo density profile and core/halo interface pressure.  Vary
``gas_core_radius`` to test core-radius convergence.

Run it with::

   cd RadHydropy/example/HydrostaticCoreSpherical1D
   python hydrostatic_core_spherical1d.py
