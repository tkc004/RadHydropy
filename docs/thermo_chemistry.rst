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
available networks are ``hydrogen``, ``hydrogen_helium``, and ``cie_cooling``.
The hydrogen network
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

* ``thermochemistry_network``: selects ``hydrogen``, ``hydrogen_helium``, or
  ``cie_cooling``.
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

For ``radiative_transfer_temporal_scheme: c2ray``, the
``hydrogen_helium`` network uses causal source-to-cell ordering. Each cell
receives the spectrum transmitted by the preceding cell, then a local coupled
backward-Euler solve advances H I, He I, He II, He III, and thermal energy.
Opacity is iterated from the time-averaged species fractions before the
outgoing photon rate is passed onward. The local solve uses damped Newton
iterations and substeps a stiff cell only when needed.

The local multigroup field is represented as ``(group, cell)`` before rates
are evaluated, so all configured groups are summed for each species. In
particular, He II photoionization uses the groups above its 54.4 eV threshold;
the high-energy groups must not be discarded when a cell is solved locally.
The resulting photon density is synchronized back to ``fluid.ngamma`` after
the source step.

Hydrogen--Helium Microphysics
-----------------------------

The ``hydrogen_helium`` network evolves H I, He I, He II, and He III with the
multigroup photoionization rates from the configured spectrum. The atomic fits
used by the network are:

* Verner et al. (1996) group-averaged photoionization cross-sections;
* Theuns et al. (1998) collisional ionization and collisional cooling rates;
* Hui & Gnedin (1997) H II and He III recombination rates and H/He
  recombination cooling fits;
* Hummer & Storey (1998) He II radiative recombination;
* Aldrovandi & Pequignot (1973) He II dielectronic recombination; and
* Black (1981) He II dielectronic-recombination cooling.

The H/He implementation uses case-B recombination cooling by default, matching
the on-the-spot treatment used by the static multifrequency examples. The
individual case-A and case-B hydrogen/helium cooling fits are available in the
rate modules for future configuration of the escape/recombination treatment.

Optional Metal PIE Cooling
--------------------------

The H/He network can optionally add metal photoionization-equilibrium (PIE)
rates from an HDF5 table without evolving a metal network. Enable it with:

.. code-block:: yaml

   metal_pie_enabled: true
   metal_pie_table_filename: path/to/metal_pie_table_Z1_metals.h5
   metallicity: 1.0

For each local source update, the multigroup H/He radiation field is traced,
then the ionization parameter is estimated as
``U = sum(n_gamma,g) / nH``. The table is interpolated in ``log10(T)``,
``log10(nH)``, and ``log10(U)`` during the coupled implicit H/He energy and
ion-fraction solve. The updated H/He state is used for one subsequent
multigroup retrace. The metal rates are added as
``metal_photoheating - metal_cooling`` in volumetric cgs units.
For HM12 UV-background PIE tables, metal PIE photoheating is disabled by
default for ``nH > 50 cm^-3`` to represent self-shielded gas, while metal PIE
cooling remains active. This cutoff is not applied to other PIE tables.
Configure it with ``metal_pie_photoheating_max_density_cm3`` or set it to
``null`` to disable the HM12 cutoff.

The current table loader supports a singleton metallicity plane. The supplied
table has ``log10(U)`` bounds of ``[-7, 0]``; lookup values outside any table
axis are clipped to the nearest boundary. This makes extrapolation explicit
and stable, but a table spanning the simulation's expected ``T``, ``nH``, and
``U`` range is recommended for physical accuracy. Metal nuclei are not added
to the particle count, so the mean molecular mass and H/He opacity remain
determined only by the H/He abundances.

PIE Validation Tests
~~~~~~~~~~~~~~~~~~~~

The PIE implementation is covered by ``tests/test_metal_pie.py``. Run these
tests with:

.. code-block:: bash

   pytest -q tests/test_metal_pie.py

The test suite verifies that:

* values at HDF5 table nodes are reproduced exactly;
* log-space interpolation reproduces known power-law tables;
* vectorized and cell-by-cell lookups agree;
* values outside the table domain are clipped to stable boundary values;
* ``U`` uses the sum of all multigroup photon densities;
* metal heating increases, and metal cooling decreases, the thermal source
  rate; and
* the supplied production table returns finite, non-negative rates.

These are local source-term tests. Full radiation/thermo-chemistry behavior
can additionally be checked with the multifrequency H/He examples documented
in :doc:`multifrequency_radiative_transfer_sph1d`.

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
