HydrostaticCoreSpherical1D
==========================

This example tests the optional pressure-supported unresolved central core.
It evolves an isothermal spherical atmosphere in the analytic gravitational
field of a fixed point mass.  The analytic density profile is

.. math::

   \rho(r) = \rho_0 \exp\left[-\frac{\Phi(r)-\Phi(r_0)}{c_s^2}\right].

The run uses ``gas_core_model: hydrostatic_fixed`` and compares the final
resolved-halo density profile and core/halo interface pressure. Vary
``par.gravity.radius_core_proper`` to test core-radius convergence.

Run it with::

   cd RadHydropy/example/HydrostaticCoreSpherical1D
   python hydrostatic_core_spherical1d.py
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/HydrostaticCoreSpherical1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
