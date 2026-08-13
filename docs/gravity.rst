Gravity
=======

Cosmological supercomoving gravity
-----------------------------------

For a supercomoving spherical simulation, enable cosmological gravity with
``cosmological_gravity: true`` and ``supercomoving_coordinates: true``. The
gravity model then subtracts the homogeneous comoving background density and
uses the enclosed excess mass:

.. math::

   \Delta M(<x)=4\pi\int_0^x
   [\varrho(x')-\bar{\varrho}]x'^2\,dx',
   \qquad
   g_{\rm sc}(x)=-\frac{G a\,\Delta M(<x)}{x^2}.

Here ``x`` is the comoving radius, ``varrho`` is the comoving gas density, and
``a`` is obtained from ``par.cosmology`` at the current supercomoving time.
The homogeneous background therefore produces zero peculiar acceleration;
only density contrast collapses or expands relative to the cosmological
background.

The public interface is::

   gravity = Gravity(
       selfgravity=True,
       cosmological=True,
       cosmology=par.cosmology,
       code_units=par.CodeUnits,
   )

The solver also constructs this mode automatically when
``par.cosmological_gravity`` is enabled. Existing physical-coordinate
``selfgravity`` and external-gravity calculations are unchanged.

RadHydropy can use an optional external gravitational field through
:class:`radhydropy.gravity.Gravity`. The field can be supplied either as a
tabulated potential, a callable, or a direct acceleration profile.

Like the rest of the runtime, gravity inputs are normalized through the shared
internal code-unit helpers in ``radhydropy.units``. That keeps tabulated
profiles, callables, and solver source terms on the same internal unit system
after startup conversion.

If no ``CodeUnits`` block is provided, the default internal unit system is
cgs.

For example scripts, the recommended pattern is to keep the public helper
interface ``unyt``-friendly while converting to code units or floats inside the
helper itself. That lets the example remain readable at the YAML boundary
without paying repeated unit-conversion overhead in the solver loop.

Point-Mass Potential Example
----------------------------

A common setup is a point mass with potential

.. math::

   \Phi(r) = -\frac{G M}{r}.

You can define it with ``unyt`` quantities like this:

.. code-block:: python

   import numpy as np
   import unyt
   from radhydropy.gravity import Gravity

   G = unyt.physical_constants.gravitational_constant
   M = 1.0 * unyt.Msun
   eps = 1.0e-3 * unyt.pc

   def point_mass_potential(r):
       return -(G * M / np.maximum(r, eps)).to(unyt.cm**2 / unyt.s**2)

   gravity = Gravity(
       externalgravity=True,
       potential=point_mass_potential,
   )

   # Later, once the mesh exists:
   # acceleration = gravity.acceleration_on_mesh(mesh)

When the solver runs with ``par.gravity = gravity`` or with
``externalgravity=True`` plus ``gravity_potential`` configured, it uses the
potential to compute the source term for momentum and energy. The same
pattern applies to the bundled point-mass and hydrostatic examples: they pass
``CodeUnits`` into the helper at setup time, then evaluate the gravity source
in code units internally.

NFW Potential Example
---------------------

The built-in :func:`radhydropy.gravity.nfw_potential` helper makes it easy to
set up a Navarro-Frenk-White halo:

.. code-block:: python

   import unyt
   from radhydropy.gravity import Gravity, nfw_potential

   rho_s = 1.0e-24 * unyt.g / unyt.cm**3
   r_s = 10.0 * unyt.kpc

   def halo_potential(r):
       return nfw_potential(r, rho_s=rho_s, r_s=r_s)

   gravity = Gravity(
       externalgravity=True,
       potential=halo_potential,
   )

   # Later, once the mesh exists:
   # acceleration = gravity.acceleration_on_mesh(mesh)

This keeps the halo definition in one place while letting the solver evaluate
the potential on whatever mesh is being used.

Notes
-----

* If you provide a tabulated potential array, also provide the matching
  coordinate array.
* If you already know the acceleration profile, you can pass it directly with
  ``acceleration=...`` instead of a potential.
* Self-gravity and external gravity can be enabled together. In spherical
  geometry, self-gravity is computed from the enclosed gas mass and added to
  the external field before the momentum and energy source update.
* The external field remains fixed while the self-gravity field is recomputed
  from the current density at every hydrodynamic stage.
* Cartesian self-gravity uses the plane-parallel Poisson equation and requires
  an explicit ``selfgravity_boundary_acceleration`` value.
