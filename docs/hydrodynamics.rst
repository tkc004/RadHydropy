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
``InternalEnergy`` instead, avoiding catastrophic cancellation.  Pressure
selection and synchronization use separate thresholds, ``dual_energy_eta1``
and ``dual_energy_eta2``.  If ``InternalEnergy`` is invalid but conservative
``E-K`` is admissible, the solver falls back to ``E-K``.  If both estimates are
invalid, it applies ``dual_energy_pressure_floor`` and records the injected
energy in the dual-energy diagnostics.

For diagnostic comparisons, ``dual_energy_pressure_selection: conservative``
keeps ``InternalEnergy`` evolving but disables the pressure switch and uses
only admissible conservative ``E-K``.  This mode is intentionally fragile in
very cold, dilute gas because cancellation can create an unphysical pressure
or sound speed; the default ``switch`` mode is the production choice.

The two representations must remain admissible and mutually consistent:

.. math::

   E \geq \frac{1}{2}\rho u^2 + \rho e_{\min},
   \qquad
   p = (\gamma-1)\rho e.

In particular, using the dual variable to compute pressure while silently
replacing ``Energy`` with ``InternalEnergy + kinetic energy`` can inject a
large amount of energy.  RadHydropy keeps the conservative ``Energy`` state
authoritative, synchronizes the dual variable only when the conservative
thermal fraction is sufficiently large, and records any pressure-floor
injection separately.  The cumulative event counters are exposed in the
result returned by ``Rsim.Step``.

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
