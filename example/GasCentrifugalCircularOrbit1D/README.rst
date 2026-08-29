Gas Centrifugal Circular-Orbit Benchmark
========================================

This source-only benchmark exercises the optional dynamical rotational
support in a spherical gas state.  A fixed central mass supplies

.. math::

   a_g=-GM/r^2,

while the gas has

.. math::

   j=\sqrt{GMr_0},\qquad a_{\rm cent}=j^2/r_0^3.

The analytic solution is therefore ``r=r0`` and ``v_r=0``.  The benchmark
writes an HDF5 initial condition, initializes ``Rsim`` through the normal
startup flow, attaches a fixed central-gravity model, and runs the spherical
hydro/source simulation.  The saved output is compared with the analytic
solution for radial velocity, momentum, specific angular momentum, mass, and
total energy.  A direct source-level check is retained as an additional
diagnostic while retaining the rotational-energy contribution

.. math::

   e_{\rm rot}=j^2/(2r_0^2).

It also integrates an eccentric source trajectory with
``j = 0.7 sqrt(G M r0)`` using RK4 and compares radius and velocity with a
high-accuracy ``solve_ivp`` reference for

.. math::

   \ddot r=j^2/r^3-GM/r^2.

This checks specific-energy conservation and provides the analytic target for
a future moving-shell/Eulerian coupling test.

The eccentric panel also includes a one-shell moving source simulation driven
by ``Solver.ApplyGravity``.  Its radius is advanced from the updated radial
momentum and is plotted against the analytic trajectory.

Run with::

   python gas_centrifugal_circular_orbit1d.py
