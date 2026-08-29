Gas Centrifugal Work Source Benchmark
=====================================

This benchmark isolates the centrifugal source update from hydro advection.
The example writes an HDF5 initial condition, starts ``Rsim`` normally, and
uses an explicit source-step backend that calls the production centrifugal
momentum/energy update.

The gas is a rotating sphere: its tangential kinetic energy produces
centrifugal support, and if that support exceeds the inward force balance it
drives radial expansion.  This source-only benchmark freezes the Eulerian
radius so that the associated radial momentum and work transfer can be
checked without the additional complications of spherical advection.

At fixed radius, with ``a = j**2/r**3``, the exact solution is

.. math::

   P(t) = P_0 + M a t,
   \qquad
   E(t) = E_0 + P_0 a t + \tfrac12 M a^2 t^2.

The run verifies saved momentum, total energy, centrifugal work, and signed
specific angular momentum.  It also verifies that the independently evolved
cold internal energy is unchanged.  The plot shows the numerical and analytic
source histories.

Run with::

   python gas_centrifugal_work_source1d.py
