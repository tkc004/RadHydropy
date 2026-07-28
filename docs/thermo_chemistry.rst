Thermo-Chemistry Solver
=======================

RadHydropy includes an optional thermo-chemistry subsystem for neutral-fraction
and thermal source terms. The species-level microphysics lives under
``radhydropy.chemistry_species`` and the composition presets are selected
through :mod:`radhydropy.chemistry`. The active source-term network is
controlled by the run parameters and is usually coupled to the hydrodynamics
and radiative-transfer updates through :class:`radhydropy.rsim.Rsim`.

Activation and Coupling
-----------------------

The thermo-chemistry network is enabled with ``hydrogen_chemistry=True`` and
the active source-term network is currently ``hydrogen``. When enabled, the
runner can evolve the neutral hydrogen fraction ``xHI = nHI / nH`` together
with heating and cooling source terms.

In the standard coupled update, RadHydropy:

* advances hydrodynamics with the finite-volume solver;
* updates the radiation field when long-characteristic transport is enabled;
* applies the hydrogen source terms; and
* subcycles the source update using the thermo-chemistry timestep controls.

The thermal equation is advanced first when enabled, then the updated
temperature is used for an implicit neutral-fraction solve.

Useful Runtime Parameters
-------------------------

The full parameter table lives in :doc:`parameters`. The thermo-chemistry
controls most commonly used by the bundled examples are:

* ``thermochemistry_network``: selects the source-term network, currently
  ``hydrogen``.
* ``chemistry_key``: selects the composition preset, such as ``H`` or
  ``HHe``.
* ``hydrogen_chemistry``: enables hydrogen thermal and neutral-fraction
  updates.
* ``hydrogen_mass_fraction``: hydrogen mass fraction used to compute ``nH``.
* ``hydrogen_update_mu``: updates the mean molecular weight from ``xHI``.
* ``hydrogen_thermal_coupling``: applies heating and cooling to the gas energy.
* ``hydrogen_collisional_ionization``: includes collisional ionization.
* ``hydrogen_source_CFL`` and ``hydrogen_source_dtmin``: control source
  subcycling.
* ``hydrogen_alpha_B`` and ``hydrogen_beta``: optional fixed rate coefficients.
* ``hydrogen_radiation_field`` and ``hydrogen_radiation_evolution``: control
  the local photon field update when ray tracing is not active.
* ``hydrogen_ngamma_initial`` and ``hydrogen_sigma_gamma``: initial photon
  density and photo-ionization opacity.

When the radiative-transfer module is enabled, ``fluid.ngamma`` is supplied by
the ray tracer before the hydrogen source terms are applied.

Static Thermochemistry
----------------------

For fixed-density tests, use
:meth:`radhydropy.rsim.Rsim.EvolveStaticThermochemistry` to evolve the
thermo-chemistry state without a hydrodynamic flux update. This is the path
used by the static Stromgren sphere examples.

Example Workflows
-----------------

The bundled thermo-chemistry examples include:

* early and late H II region expansion;
* hydrogen photoheating and recombination;
* dynamic Stromgren sphere photoheating; and
* static Stromgren sphere benchmarks.

These examples demonstrate both coupled hydrodynamic runs and static source
evolution with a fixed density field.

Composition presets such as ``H``, ``HHe``, ``HHeM``, ``HHeMol``, and
``HHeMMol`` are exposed through :mod:`radhydropy.chemistry` for future multi-
species extensions.
