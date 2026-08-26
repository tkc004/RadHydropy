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

Energy Equations
----------------

The ordinary Euler formulation evolves the conserved state

.. math::

   U = \left(\rho,\; \rho u,\; E\right),

where :math:`\rho` is density, :math:`u` is radial velocity, and

.. math::

   E = \rho e + \frac{1}{2}\rho u^2

is the total energy density.  In Cartesian notation, the equations are

.. math::

   \frac{\partial \rho}{\partial t}
   + \nabla\cdot(\rho u) = 0,

   \frac{\partial (\rho u)}{\partial t}
   + \nabla\cdot(\rho u\otimes u + p I) = \rho g,

   \frac{\partial E}{\partial t}
   + \nabla\cdot\left[(E+p)u\right]
   = \rho u\cdot g + S_E.

Here :math:`e` is the specific internal energy, :math:`p` is pressure,
:math:`g` is acceleration, and :math:`S_E` contains non-hydrodynamic energy
sources such as heating and cooling.  For an ideal gas,

.. math::

   p = (\gamma-1)\rho e.

RadHydropy discretizes these equations in finite-volume form.  In spherical
geometry, each flux is multiplied by the face area and divided by the cell
volume; the radial momentum equation also contains the corresponding
geometric pressure term.  Gravity and thermo-chemistry are applied as source
updates around the conservative hydro flux update.

Dual-Energy Formulation
-----------------------

In a cold, fast flow, recovering thermal energy from the conserved variables
requires subtracting two large quantities:

.. math::

   \rho e = E - \frac{1}{2}\rho u^2.

When kinetic energy dominates, this subtraction can lose precision or produce
an inadmissible negative value even when the physical thermal energy is
positive.  With ``dual_energy: true``, RadHydropy therefore evolves a second
thermal-energy variable, ``InternalEnergy``, in addition to the conserved
``Energy`` field.  Its continuum equation is

.. math::

   \frac{\partial(\rho e)}{\partial t}
   + \nabla\cdot(\rho e u)
   + p\,\nabla\cdot u
   = S_e,

where :math:`S_e` is the thermal part of the source update.  The discrete
internal-energy flux uses the same limited face coefficients and spherical
pressure-work term as the hydro update.

The solver normally computes pressure from total energy.  If
:math:`E - \tfrac12\rho u^2` becomes too small relative to :math:`E`, it uses
``InternalEnergy`` instead, avoiding catastrophic cancellation.  This is a
pressure-reconstruction fallback, not permission to create energy: the total
energy equation remains the conservative equation.

The two representations must remain admissible and mutually consistent:

.. math::

   E \geq \frac{1}{2}\rho u^2 + \rho e_{\min},
   \qquad
   p = (\gamma-1)\rho e.

In particular, using the dual variable to compute pressure while silently
replacing ``Energy`` with ``InternalEnergy + kinetic energy`` can inject a
large amount of energy.  A conservative implementation should instead reject
or limit a hydro update that violates the inequality, retry it with a smaller
timestep or positivity-preserving fluxes, and only synchronize the two energy
representations when the chosen state is admissible.  Energy corrections, if
ever required as a diagnostic fallback, must be explicitly recorded in the
energy audit.

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

The hydro step also supports an optional SSPRK2 time integrator:

.. code-block:: python

   step = sim.Step(mode="hydro", hydro_integrator="ssprk2")

For a full coupled update, use ``mode="hydro_sources"`` instead.
