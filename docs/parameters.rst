Run Parameters
==============

Runtime parameters are passed to :class:`radhydropy.params.Par` as a dictionary.
Missing keys are filled from :data:`radhydropy.params.refparams`.

Unit System
-----------

Every run must define a mandatory ``CodeUnits`` block in ``runparams``.

Cosmological initial conditions and outputs carry their cosmology and variable
representation in the HDF5 ``Header``. When
``supercomoving_coordinates: true`` is selected, the file stores comoving
radius, supercomoving time, comoving density, supercomoving velocity, and
supercomoving thermodynamic variables. The loader restores this mode from the
header, including the Einstein--de Sitter reference time and scale factor.
RadHydropy uses that block to build a shared internal code-unit system and
converts runtime state into those code units during initialization. After that
startup conversion, the solver, geometry, gravity, and source-term updates
operate in the internal unit system instead of repeatedly converting units on
the hot paths.

Snapshot and initial-condition files still carry units in HDF5, and
``radhydropy.io.readhdf5`` now requires ``Header.attrs["CodeUnits"]`` to be
present so it can convert the stored quantities back into code-unit numeric
arrays when the run starts. There is no cgs fallback in the current workflow.

Example helpers can still accept ``unyt`` quantities at the script boundary,
but they should convert to code units or plain floats internally before they
enter any repeated solver loop or gravity calculation.

The YAML form used by the examples is:

.. code-block:: yaml

   CodeUnits:
     name: galactic_unit_system
     InternalUnitSystem:
       UnitMass_in_cgs:     4.92e31
       UnitLength_in_cgs:   3.08567758e21
       UnitVelocity_in_cgs: 1.0e5
       UnitCurrent_in_cgs:  1.0
       UnitTemp_in_cgs:     1.0

If you already have a :class:`unyt.unit_systems.UnitSystem`, it can also be
passed as ``CodeUnits``. The loader converts that object into the same
``CodeUnits`` dataclass used by the runtime.

Common Runtime Keys
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 54 22

   * - Key
     - Meaning
     - Typical unit
   * - ``simname``
     - Simulation name used by scripts and logs.
     - dimensionless
   * - ``ICfilename``
     - HDF5 initial-condition file path.
     - path string
   * - ``outdir``
     - Directory for output files.
     - path string
   * - ``outfileprefix``
     - Prefix for HDF5 outputs written by :meth:`radhydropy.rsim.Rsim.Run`.
     - string
   * - ``coordsys``
     - Coordinate system. Supported values are ``cartesian`` and ``spherical``.
     - string
   * - ``EOStype``
     - Equation-of-state type. Supported values are ``polytropic`` and
       ``isothermal``.
     - string
   * - ``gamma``
     - Adiabatic index for polytropic gas.
     - dimensionless
   * - ``temperature``
     - Default gas/background temperature used for scalar temperature
       parameters. The default is ``2.7 K``.
     - temperature
   * - ``timesim``
     - Final simulation time.
     - time
   * - ``outdeltatime``
     - Output cadence.
     - time
   * - ``outputtimefilename``
     - Optional txt file containing explicit output times. The first non-empty
       row gives the time unit and the remaining rows list the output times.
     - path string
   * - ``CFL``
     - Courant factor used by :meth:`radhydropy.solver.Solver.GetTimeStep`.
     - dimensionless
   * - ``boundcond``
     - Boundary condition, such as ``Periodic``, ``Open``, ``Reflecting``,
       ``OpenSph``, ``InflowSph``, or ``OutflowSph``.
     - string
   * - ``order``
     - Reconstruction order. ``0`` uses piecewise constant fluxes; ``1`` uses
       reconstructed states with flux limiting.
     - dimensionless
   * - ``noghost``
     - Number of ghost cells on each side of the domain.
     - cells
   * - ``dtmin`` / ``dtmax``
     - Minimum and maximum allowed timesteps.
     - time
   * - ``area``
     - Cartesian cross-sectional area used to calculate volumes.
     - area

If ``outputtimefilename`` is provided, RadHydropy ignores ``outdeltatime`` and
writes outputs at the explicit times listed in the txt file. The file format is
one time unit on the first non-empty line, followed by one output time per
line. Include ``timesim`` in the list if you want the final state written as an
output file. For example:

.. code-block:: text

   yr
   0.0
   1.0e4
   2.0e4


Boundary-Specific Keys
----------------------

The spherical inflow and outflow boundary conditions use additional primitive
state parameters:

.. list-table::
   :header-rows: 1
   :widths: 28 50 22

   * - Key
     - Meaning
     - Typical unit
   * - ``rho_inflow`` / ``rho_outflow``
     - Density imposed at the inflow or outflow ghost cells.
     - mass density
   * - ``vel_inflow`` / ``vel_outflow``
     - Velocity imposed at the inflow or outflow ghost cells.
     - length / time
   * - ``temp_inflow`` / ``temp_outflow``
     - Temperature used to derive boundary pressure.
     - temperature
   * - ``mu_inflow`` / ``mu_outflow``
     - Mean molecular weight used to derive boundary pressure.
     - dimensionless

Thermo-Chemistry Keys
---------------------

Thermo-chemistry is disabled by default. The active network is selected by
``thermochemistry_network``; available networks are ``hydrogen``,
``hydrogen_helium``, ``cie_cooling``, and ``pie_uvbg_cooling``.
The species composition preset is selected separately with ``chemistry_key``
and currently supports values such as ``H`` and ``HHe``. Set
``hydrogen_chemistry=True`` to evolve the neutral hydrogen fraction
``xHI = nHI / nH`` and apply the associated line, ionization, bremsstrahlung,
and case-B recombination cooling source terms. Source terms are subcycled
inside each hydrodynamic step: the thermal equation is advanced explicitly
when enabled, then the updated temperature is used for a backward-Euler
neutral-fraction solve.

The ``cie_cooling`` network uses CHIANTI collisional-ionization-equilibrium
tables. It applies the tabulated radiative cooling rate explicitly and
adaptively subcycles each hydrodynamic step using
``cooling_safety_factor``. CIE ion fractions are looked up from the table and
are not advected as independent fluid fields.

.. list-table::
   :header-rows: 1
   :widths: 28 50 22

   * - Key
     - Meaning
     - Typical unit
   * - ``thermochemistry_network``
     - Thermo-chemistry network name: ``hydrogen``, ``hydrogen_helium``,
       ``cie_cooling``, or ``pie_uvbg_cooling``.
     - string
   * - ``cie_cooling``
     - Enable the CIE radiative cooling source when using the ``cie_cooling``
       network.
     - boolean
   * - ``metallicity``
     - Metallicity in solar units used by the CIE cooling and electron-fraction
       tables.
     - ``Z/Zsun``
   * - ``cie_ion_fraction_table`` / ``cie_cooling_table``
     - Optional paths to the CIE ion-fraction and CHIANTI cooling HDF5 tables.
     - path
   * - ``cie_abundance_file``
     - Optional path to the CHIANTI abundance file used for electron fractions.
     - path
   * - ``cooling_safety_factor``
     - Fraction of the local cooling time allowed for each explicit CIE
       substep. The default is ``0.1``.
     - dimensionless
   * - ``cooling_temperature_floor``
     - Minimum temperature enforced after a CIE cooling update.
     - temperature
   * - ``chemistry_key``
     - Composition preset name used by :mod:`radhydropy.chemistry`.
     - string
   * - ``hydrogen_chemistry``
     - Enable hydrogen thermal and neutral-fraction source terms.
     - boolean
   * - ``hydrogen_mass_fraction``
     - Hydrogen mass fraction used to compute ``nH`` from density.
     - dimensionless
   * - ``hydrogen_xHI_initial``
     - Initial neutral fraction used when the HDF5 file has no
       ``NeutralFraction`` dataset.
     - dimensionless
   * - ``hydrogen_xHI_inflow`` / ``hydrogen_xHI_outflow``
     - Neutral fraction imposed by spherical inflow/outflow ghost cells.
     - dimensionless
   * - ``hydrogen_source_CFL``
     - Fractional subcycle limiter for ``u / |du/dt|`` and
       ``xHI / |dxHI/dt|``.
     - dimensionless
   * - ``hydrogen_source_solver``
     - Source integrator; defaults to ``coupled_implicit`` for a backward-Euler
       solve of internal energy and ``xHI`` together. ``explicit`` selects the
       standard subcycled update.
     - string
   * - ``hydrogen_implicit_tolerance`` / ``hydrogen_implicit_max_iterations``
     - Convergence tolerance and iteration limit for the coupled implicit
       source solve.
     - dimensionless / integer
   * - ``hydrogen_implicit_convergence_tolerance``
     - Relative difference tolerated between one implicit step and two
       half-sized implicit steps.
     - dimensionless
   * - ``hydrogen_implicit_max_refinements``
     - Maximum factor-of-two timestep refinements used to compare one
       implicit step with two half-sized steps.
     - integer
   * - ``hydrogen_implicit_fallback``
     - Action after a failed coupled solve: ``explicit`` or ``error``.
     - string
   * - ``hydrogen_update_mu``
     - Update mean molecular weight from ``xHI`` for pure-hydrogen runs.
     - boolean
   * - ``hydrogen_thermal_coupling``
     - Apply hydrogen heating/cooling to the gas energy. Disable this for
       fixed-temperature chemistry tests.
     - boolean
   * - ``hydrogen_collisional_ionization``
     - Include collisional ionization in the neutral-fraction equation.
     - boolean
   * - ``hydrogen_alpha_B`` / ``hydrogen_beta``
     - Optional fixed recombination or collisional ionization coefficients.
       Leave as ``None`` to use the temperature-dependent fits.
     - ``cm^3 s^-1``
   * - ``hydrogen_radiation_field``
     - Enable photo-ionization and photo-heating from ``fluid.ngamma``.
     - boolean
   * - ``hydrogen_radiation_evolution``
     - Apply the local analytic hydrogen photon absorption update when no
       ray-tracing transport is active.
     - boolean
   * - ``hydrogen_ngamma_initial``
     - Initial photon number density used when the HDF5 file has no
       ``PhotonNumberDensity`` dataset.
     - ``cm^-3``
   * - ``hydrogen_sigma_gamma``
     - Hydrogen photo-ionization cross-section.
     - area
   * - ``metal_pie_enabled``
     - Add optional metal PIE heating and cooling to the coupled H/He source
       update. Metals do not change the mean molecular mass or H/He opacity.
     - boolean
   * - ``metal_pie_table_filename``
     - HDF5 metal PIE table containing volumetric photoheating and cooling
       rates on ``(T, nH, U)`` axes. Loaded once during startup.
     - path
   * - ``metal_pie_photoheating_max_density_cm3``
     - For HM12 UV-background PIE tables, disable metal photoheating above
       this hydrogen density while retaining metal PIE cooling. Other PIE
       tables are unaffected. Set to ``null`` to disable this cutoff.
     - ``cm^-3``
   * - ``metal_pie_redshift``
     - HM12 UV-background redshift used by the non-RT ``pie_uvbg_cooling``
       network.
     - dimensionless
   * - ``pie_uvbg_implicit_tolerance``
     - Relative energy-difference tolerance between a full implicit step and
       two implicit half-steps. The default is ``1e-3``.
     - dimensionless
   * - ``pie_uvbg_implicit_max_retries`` / ``pie_uvbg_implicit_max_iterations``
     - Maximum timestep halvings / bisection iterations for the implicit PIE
       update. Defaults are ``8`` and ``64``.
     - integer
   * - ``hydrogen_epsilon_gamma``
     - Excess photo-ionization energy per absorbed photon.
     - energy

Radiative Transfer Keys
-----------------------

Set ``radiative_transfer=True`` to compute ``fluid.ngamma`` from the optional
one-dimensional long-characteristic ray tracer before the hydrogen source terms
are applied. See :doc:`radiative_transfer` for the implementation details.

.. list-table::
   :header-rows: 1
   :widths: 32 48 20

   * - Key
     - Meaning
     - Typical unit
   * - ``radiative_transfer``
     - Enable optional long-characteristic radiative transfer.
     - boolean
   * - ``radiative_transfer_method``
     - Transport method. Currently ``long_characteristics``.
     - string
   * - ``radiative_transfer_temporal_scheme``
     - ``instantaneous`` for the existing update or ``c2ray`` for causal,
       time-averaged C²-Ray source integration. With
       ``thermochemistry_network: hydrogen_helium``, it also enables the
       coupled H/He C²-Ray update.
     - string
   * - ``radiative_transfer_c2ray_max_iterations``
     - Maximum opacity iterations for each source cell in C²-Ray mode.
     - dimensionless
   * - ``radiative_transfer_c2ray_tolerance``
     - Absolute neutral-fraction convergence tolerance for each source cell.
     - dimensionless
   * - ``radiative_transfer_c2ray_relaxation``
     - Under-relaxation factor for the C²-Ray mean neutral fraction.
     - dimensionless
   * - ``radiative_transfer_c2ray_nonconvergence``
     - ``warn`` (default), ``raise``, or silent handling after the iteration
       limit is reached.
     - string
   * - ``radiative_transfer_c2ray_ode_max_iterations``
     - Maximum damped-Newton iterations for each local coupled H/He solve.
     - dimensionless
   * - ``radiative_transfer_c2ray_ode_tolerance``
     - Scaled residual tolerance for the local coupled H/He solve.
     - dimensionless
   * - ``radiative_transfer_boundary_flux``
     - Incident photon number flux for Cartesian rays or spherical boundary
       illumination.
     - ``cm^-2 s^-1``
   * - ``radiative_transfer_source_photon_rate``
     - Spherical source photon rate. Prefer this for radial traces starting at
       ``r = 0``.
     - ``s^-1``
   * - ``radiative_transfer_direction``
     - ``+1`` for left-to-right or inner-to-outer tracing; ``-1`` for the
       opposite direction.
     - dimensionless
   * - ``radiation_group_edges_eV``
     - Increasing photon-energy edges. The number of radiation groups is one
       less than the number of edges. Omit this for legacy single-group mode.
     - eV
   * - ``radiation_group_sigma_gamma``
     - H I photo-ionization cross-section for each radiation group.
     - ``cm^2`` per group
   * - ``radiation_group_epsilon_gamma``
     - Excess photoheating energy for each radiation group.
     - erg per group
   * - ``radiative_transfer_source_photon_rate_groups``
     - Spherical source photon rate for each radiation group.
     - ``s^-1`` per group
   * - ``radiative_transfer_boundary_flux_groups``
     - Boundary photon flux for each radiation group.
     - ``cm^-2 s^-1`` per group
   * - ``radiation_spectrum_filename``
     - HDF5 file containing the ``RadiationSpectrum`` group and spectrum
       datasets. It is loaded during startup and after IC header restoration.
     - path
   * - ``radiation_spectrum_total_photon_rate``
     - Optional total ionizing photon rate. It rescales all ionizing HDF5
       groups by one common factor while preserving the spectrum.
     - ``s^-1``

Direct Radiation Pressure Keys
-------------------------------

Direct radiation pressure uses the absorbed photon rate returned by the
thermo-chemistry source update. It is applied afterward as a momentum source;
thermo-chemistry itself only updates the absorbed-photon bookkeeping and the
thermal/chemical state. See :doc:`radiation_pressure` for the equations and
the dedicated dynamic example.

.. list-table::
   :header-rows: 1
   :widths: 32 48 20

   * - Key
     - Meaning
     - Typical unit
   * - ``radiation_pressure``
     - Enable momentum deposition from absorbed photons. The default is
       ``false``.
     - boolean
   * - ``radiation_pressure_efficiency``
     - Dimensionless coupling efficiency multiplying the absorbed photon
       momentum. ``1.0`` transfers all absorbed photon momentum to the gas.
     - dimensionless

For photon group ``g``, the momentum-rate density is proportional to
``absorbed_photon_rate[g] * photon_energy_erg[g] / c``. The transport result
must provide physical-cell-only absorbed rates and photon energies. The ray
direction is taken from ``radiative_transfer_direction``. Cells with zero
density are skipped safely.

The standard 20 pc radiation-pressure example uses:

.. code-block:: yaml

   radiative_transfer: true
   radiation_pressure: true
   radiation_pressure_efficiency: 1.0

The example-specific ``radiation_pressure_source_luminosity`` key used by the
isolated thin-shell benchmark is not a core solver parameter; it supplies the
synthetic source luminosity for that example's source-only step backend.




Gravity Keys
------------

Gravity source terms are disabled by default. They can combine external
gravity, gas self-gravity, cosmological gravity, and live dark-matter shells.
See :doc:`gravity` and :doc:`dark_matter` for the source models.

.. list-table::
   :header-rows: 1
   :widths: 32 48 20

   * - Key
     - Meaning
     - Typical unit
   * - ``selfgravity``
     - Enable gas self-gravity computed from the enclosed gas mass in spherical
       geometry or the plane-parallel Poisson field in Cartesian geometry.
     - boolean
   * - ``externalgravity``
     - Enable an externally supplied potential or acceleration profile.
     - boolean
   * - ``gravity``
     - Optional preconstructed :class:`radhydropy.gravity.Gravity` object. This
       is normally supplied by an example script rather than YAML.
     - object
   * - ``gravity_potential``
     - External potential profile, callable, or tabulated potential used when
       ``externalgravity`` is enabled.
     - potential
   * - ``gravity_coordinate``
     - Coordinates corresponding to a tabulated external potential or field.
     - length
   * - ``gravity_acceleration``
     - Direct external acceleration profile, callable, or tabulated field.
     - acceleration
   * - ``selfgravity_softening``
     - Softening length used by gas self-gravity.
     - length
   * - ``selfgravity_boundary_acceleration``
     - Boundary acceleration used by Cartesian self-gravity.
     - acceleration
   * - ``dark_matter_crossing_safety_factor``
     - Safety factor used to limit timesteps when live dark-matter shells are
       predicted to cross.
     - dimensionless
   * - ``dark_matter``
     - Runtime :class:`radhydropy.dark_matter.DarkMatterShells` object. It is
       generally constructed by an example IC helper or restored from an HDF5
       snapshot.
     - object

``externalgravity`` and ``selfgravity`` may be enabled together. The solver
adds their accelerations before updating gas momentum and energy. A live
``dark_matter`` object is coupled through enclosed gas and dark-matter masses.

Cosmology Keys
--------------

Cosmological expansion uses an Einstein--de Sitter background and can be
combined with supercomoving coordinates. The cosmology object is constructed
automatically by :class:`radhydropy.params.Par` when
``cosmological_expansion`` is enabled.

.. list-table::
   :header-rows: 1
   :widths: 32 48 20

   * - Key
     - Meaning
     - Typical unit
   * - ``cosmological_expansion``
     - Enable cosmological expansion and construct the configured background
       cosmology.
     - boolean
   * - ``cosmological_gravity``
     - Enable density-contrast cosmological gravity. The homogeneous background
       is subtracted from the enclosed mass.
     - boolean
   * - ``supercomoving_coordinates``
     - Store and evolve comoving coordinates, supercomoving time, comoving
       density, peculiar velocity, and supercomoving thermodynamic variables.
     - boolean
   * - ``cosmology_type``
     - Background model. The current supported value is
       ``einstein_de_sitter`` (also accepted as ``EinsteinDeSitter``).
     - string
   * - ``cosmology_t_ref``
     - Reference cosmic time used to normalize the Einstein--de Sitter scale
       factor.
     - code time
   * - ``cosmology_a_ref``
     - Reference scale factor at ``cosmology_t_ref``.
     - dimensionless
   * - ``coordinate_frame``
     - Coordinate representation, normally ``physical`` or automatically set
       to ``comoving`` for supercomoving runs.
     - string
   * - ``time_coordinate``
     - Time representation, normally ``cosmic`` or automatically set to
       ``supercomoving``.
     - string
   * - ``velocity_representation``
     - Velocity representation, such as ``physical`` or
       ``supercomoving_peculiar``.
     - string
   * - ``density_representation``
     - Density representation, ``physical`` or ``comoving``.
     - string
   * - ``pressure_representation``
     - Pressure representation, ``physical`` or ``supercomoving``.
     - string
   * - ``temperature_representation``
     - Temperature representation, ``physical`` or ``supercomoving``.
     - string

For a supercomoving run, set at minimum:

.. code-block:: yaml

   cosmological_expansion: true
   cosmological_gravity: true
   supercomoving_coordinates: true
   cosmology_type: einstein_de_sitter
   cosmology_t_ref: 1.0
   cosmology_a_ref: 1.0
