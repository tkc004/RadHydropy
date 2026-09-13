Boundary Conditions
===================

RadHydropy applies boundary conditions by filling ghost cells before each
hydrodynamic update. The active mode is selected with the
``par.boundary.condition`` runtime
parameter and is handled by :meth:`radhydropy.solver.Solver.SetBoundary`.

Supported Modes
---------------

The bundled solver currently supports these boundary-condition names:

* ``Periodic``
* ``Open``
* ``Reflecting``
* ``OpenSph``
* ``InflowSph``
* ``OutflowSph``

Cartesian Boundaries
--------------------

For Cartesian problems:

* ``Periodic`` copies values from the opposite side of the domain.
* ``Open`` copies the nearest interior state into the ghost cells.
* ``Reflecting`` mirrors the scalar fields and flips the velocity sign.

These choices are useful for advection tests, shock tubes, and problems where
the outer state should remain unchanged near the edges of the grid.

Spherical Boundaries
--------------------

For spherical problems:

* ``OpenSph`` applies a symmetric inner boundary and copies the outermost
  interior state into the outer ghost cells.
* ``InflowSph`` uses the symmetric inner boundary and imposes density,
  velocity, and temperature at the outer ghost cells.
* ``OutflowSph`` imposes the primitive state at the inner ghost cells and
  copies the outer interior state outward.
* ``WindSph`` is experimental and is not used by maintained RadHydropy
  examples. Use ``OutflowSph`` for example workflows; the
  ``StellarWindBubble1D`` case uses that established boundary with an explicit
  resolved injection region.

The solver also keeps the origin flux and center momentum consistent with the
spherical geometry so that the cell adjacent to ``r = 0`` remains symmetric.

Boundary-Specific Parameters
----------------------------

The general boundary option is chosen through ``par.boundary.condition``. The spherical
inflow and outflow modes also use these run parameters:

* ``rho_inflow_proper`` / ``rho_outflow_proper``
* ``vel_inflow_proper`` / ``vel_outflow_proper``
* ``temperature_inflow_proper`` / ``temperature_outflow_proper``
* ``mu_inflow`` / ``mu_outflow``

The corresponding pressure is derived from the equation of state.

Pressure-Supported Central Core
-------------------------------

Spherical calculations may optionally replace the unresolved innermost region
with a fixed-mass pressure-supported core::

   gas_core_model: hydrostatic_fixed
   radius_core_proper: {value: 20.0, unit: kpc}

The core retains its primitive state and supplies pressure at the core/halo
interface, while mass and energy flux through that interface are disabled.  It
is not a sink.  Core cells are excluded from the resolved-halo CFL minimum.
This model is disabled by default and should be tested for convergence by
varying ``radius_core_proper``.

Practical Notes
---------------

Boundary updates happen before each flux computation, so the ghost cells are
always refreshed from the latest primitive state. That means a source update
that changes temperature, density, or ionization state will be reflected in the
next hydrodynamic step automatically.
