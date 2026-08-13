Dark Matter and Spherical Shell Gravity
=======================================

RadHydropy includes a pure collisionless dark-matter shell model in
:mod:`radhydropy.dark_matter`. It represents a spherical dark-matter
distribution by infinitesimally thin shells. Each shell has a fixed mass and
specific angular momentum, but its radius and radial velocity evolve in time.

This model is separate from the gas :class:`radhydropy.fluid.Fluid`: dark
matter has no gas pressure, temperature, or Euler fluxes. Its gravity can be
coupled to gas self-gravity through a shared spherical enclosed-mass field.
The pure-shell examples isolate the collisionless dynamics, while
``GasDarkMatterShellCoupling1D`` exercises the mutual gas--dark-matter field.

Shell equation of motion
------------------------

For a shell at radius ``r``,

.. math::

   \ddot r = -\frac{G M(<r)}{(r+a)^2} + \frac{j^2}{r^3},

where ``a`` is a central softening length and ``j`` is the shell's conserved
specific angular momentum. The enclosed shell mass uses the half-shell
convention at an exactly coincident radius:

.. math::

   M(<r_i) = \sum_{r_k<r_i}m_k + \frac{1}{2}m_i.

The angular-momentum term prevents shells with nonzero ``j`` from falling
directly through the centre. A cold shell with ``j=0`` can collapse toward the
softened centre.

Numerical evolution
-------------------

``DarkMatterShells.step`` uses a kick-drift-kick update. After the drift, all
shell arrays are sorted by radius while preserving the association between
radius, velocity, mass, and angular momentum. Shell crossings are therefore
allowed. The coupled gas path uses the same shell update while adding gas
enclosed mass to the shell force.

Before a predicted neighboring-shell crossing, the step is limited using the
linear estimate

.. math::

   \Delta t_{\rm cross} =
   \frac{r_{i+1}-r_i}{v_i-v_{i+1}},

when the inner shell is catching the outer shell. The example advances just
through the event and resorts the shell records.

Cosmological coupling
---------------------

With ``cosmological_gravity`` and supercomoving coordinates enabled, gas and
dark matter use one common excess-mass field. For a shell at comoving radius
``x``,

.. math::

   \frac{d^2x}{d\tau^2} =
   -\frac{G a}{(x+a_\mathrm{soft})^2}
   [M_\mathrm{gas}(<x)+M_\mathrm{DM}(<x)-M_\mathrm{bg}(<x)]
   +\frac{j^2}{x^3},

where ``M_bg`` is the homogeneous comoving background mass. Gas cells use the
same expression, with the live shell mass added to their enclosed mass. The
scale factor and background density are evaluated at the current
supercomoving time, so a homogeneous gas-plus-dark-matter background has zero
peculiar acceleration.

Fixed enclosed-mass analytic benchmark
---------------------------------------

The ``DarkMatterFixedMassOrbit1D`` example evolves a negligible-mass shell in
a prescribed central mass ``M``. Its effective potential is

.. math::

   \Phi_{\rm eff}(r) = -\frac{GM}{r+a} + \frac{j^2}{2r^2},

and the conserved energy is

.. math::

   E = \frac{1}{2}\dot r^2 + \Phi_{\rm eff}(r).

The trajectory can be written as the quadrature

.. math::

   t-t_0 = \int_{r_0}^{r}
   \frac{dr}{\sqrt{2[E-\Phi_{\rm eff}(r)]}}.

The example uses a high-accuracy integration of the equivalent radial ODE as
the reference trajectory because it handles turning points more robustly than
direct quadrature. It compares shell radius and energy drift.

Run the benchmark from its directory:

.. code-block:: bash

   cd example/DarkMatterFixedMassOrbit1D
   python dark_matter_fixed_mass_orbit1d.py

Shell-crossing example
----------------------

``DarkMatterShellCrossing1D`` evolves multiple self-gravitating shells with
fixed masses and angular momenta. It records the sorted shell radii and a
diagnostic energy history:

.. code-block:: bash

   cd example/DarkMatterShellCrossing1D
   python dark_matter_shell_crossing1d.py

Analytic gas--dark-matter orbit benchmark
------------------------------------------

``GasDarkMatterAnalyticOrbit1D`` freezes a uniform gas background and a
central dark-matter mass. A negligible-mass shell then sees

.. math::

   M(<r)=M_0+\frac{4\pi}{3}\rho_g r^3,

which gives a time-dependent analytic radial ODE including angular momentum.
The shell integrator is compared against a high-accuracy reference solution
of that ODE. This benchmark validates the combined enclosed-mass force without
introducing gas back-reaction.

Current scope
-------------

The shell model currently supports:

* spherical self-gravity among dark-matter shells;
* fixed shell masses and specific angular momenta;
* central gravitational softening;
* shell crossing and radius sorting;
* code-unit input through :class:`radhydropy.units.CodeUnits`.
* mutual spherical gas--dark-matter gravity through
  ``GasDarkMatterShellCoupling1D``;
* cosmological supercomoving gas--dark-matter excess-mass coupling;
* gas and dark-matter total-mass diagnostics;
* ``DarkMatter`` snapshot output groups.

It does not yet support:

* live particle deposition onto a gas mesh;
* exact crossing-event energy exchange;
* non-spherical dark-matter dynamics.

These limitations are intentional while the isolated shell integrator is being
validated.
