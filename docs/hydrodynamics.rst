Hydrodynamics Solver
====================

RadHydropy advances the fluid with a one-dimensional finite-volume Euler
solver for mass, momentum, and energy. The solver is implemented in
:mod:`radhydropy.solver` and is coordinated by :class:`radhydropy.rsim.Rsim`.

Finite-Volume Update
--------------------

For each hydrodynamic step, RadHydropy:

* applies the configured boundary conditions;
* reconstructs left and right primitive states at cell faces;
* computes numerical fluxes with a GLF/Rusanov update;
* applies the flux divergence to the conserved variables; and
* converts the result back to primitive form.

The standard coupled time step uses the hydrodynamic update together with
source terms by calling ``Step(mode="hydro_sources")`` inside
:meth:`radhydropy.rsim.Rsim.Run` and :meth:`radhydropy.rsim.Rsim.RunAll`.

Reconstruction Order
--------------------

The ``order`` runtime parameter controls how face states are built:

* ``order=0`` uses piecewise-constant states at faces.
* ``order=1`` uses gradient reconstruction with flux limiting.

This lets the same solver support both simple first-order tests and the
reconstructed runs used by the bundled examples.

Boundary Handling
------------------

Boundary conditions are enforced before each hydrodynamic update. For
spherical problems, the solver also applies the origin symmetry corrections
needed to keep the center cell momentum and origin fluxes consistent.

Direct Hydro Steps
------------------

If you want only the hydrodynamics update, call:

.. code-block:: python

   step = sim.Step(mode="hydro")
   print(step["dt"])

For a full coupled update, use ``mode="hydro_sources"`` instead.
