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

The source-term network is selected with ``thermochemistry_network``. The
available networks are ``hydrogen`` and ``cie_cooling``. The hydrogen network
can evolve the neutral hydrogen fraction ``xHI = nHI / nH`` together with
hydrogen heating and cooling source terms. The CIE network uses tabulated
collisional-ionization-equilibrium ion fractions and radiative cooling rates.

In the standard coupled update, RadHydropy:

* advances hydrodynamics with the finite-volume solver;
* updates the radiation field when long-characteristic transport is enabled;
* applies the selected thermo-chemistry source terms; and
* subcycles the source update using the thermo-chemistry timestep controls.

Hydrogen thermal energy is advanced explicitly, followed by an implicit
neutral-fraction solve. CIE cooling is also applied explicitly, but each
hydrodynamic step is adaptively subcycled so that every cooling substep is at
most ``cooling_safety_factor`` times the local cooling time. The CIE cooling
rate and temperature are recomputed after every substep.

Useful Runtime Parameters
-------------------------

The full parameter table lives in :doc:`parameters`. The thermo-chemistry
controls most commonly used by the bundled examples are:

* ``thermochemistry_network``: selects ``hydrogen`` or ``cie_cooling``.
* ``chemistry_key``: selects the composition preset, such as ``H`` or
  ``HHe``.
* ``hydrogen_chemistry``: enables hydrogen thermal and neutral-fraction
  updates.
* ``hydrogen_mass_fraction``: hydrogen mass fraction used to compute ``nH``.
* ``hydrogen_update_mu``: updates the mean molecular weight from ``xHI``.
* ``hydrogen_thermal_coupling``: applies heating and cooling to the gas energy.
* ``compton_cmb_enabled``: adds optional Compton heating/cooling from the CMB.
* ``compton_cmb_redshift``: redshift used to calculate the CMB temperature.
* ``cmb_temperature_0``: present-day CMB temperature, default ``2.7255 K``.
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

CMB Compton Heating
-------------------

Compton coupling is disabled by default and can be enabled independently of
the radiative-transfer photon groups::

   compton_cmb_enabled: true
   compton_cmb_redshift: 10.0

The source uses ``T_CMB = cmb_temperature_0 * (1 + compton_cmb_redshift)`` and
adds the following volumetric rate to the selected thermo-chemistry network:

.. math::

   \dot{e}_{\rm C} =
   \frac{4\sigma_{\rm T} c a_{\rm r} k_{\rm B}}{m_e c^2}
   n_e T_{\rm CMB}^4 (T_{\rm CMB} - T).

A positive rate heats the gas and a negative rate cools it. Electron density is
computed from the current ionization state for the hydrogen and H/He networks;
the CIE network obtains it from its ion-fraction table. The term is included in
both source-rate and source-timestep calculations.

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

CIE Cooling
-----------

The ``cie_cooling`` network assumes collisional ionization equilibrium. It
does not evolve ion fractions as additional fluid variables; instead, it
interpolates equilibrium ion fractions and CHIANTI radiative cooling rates at
the current temperature and electron density. The volumetric cooling rate is

.. math::

   \dot{e}_{\rm cool} = -n_e n_H \Lambda(T, n_e, Z).

The default tables are searched for in the sibling
``CHIANTI_11.0.2_database`` directory. Their locations can be overridden with
``cie_ion_fraction_table``, ``cie_cooling_table``, and
``cie_abundance_file``. Important runtime controls are:

* ``cie_cooling``: enables the CIE cooling source;
* ``metallicity``: metallicity in solar units;
* ``hydrogen_mass_fraction``: used to calculate ``n_H`` from mass density;
* ``cooling_safety_factor``: explicit cooling subcycle fraction, default
  ``0.1``; and
* ``cooling_temperature_floor``: minimum temperature maintained by the source
  update.

The CIE cooling table includes electron-density dependence. Consequently,
electron density can affect both the tabulated cooling coefficient and the
overall ``n_e n_H`` normalization.

Composition presets such as ``H``, ``HHe``, ``HHeM``, ``HHeMol``, and
``HHeMMol`` are exposed through :mod:`radhydropy.chemistry` for future multi-
species extensions.
