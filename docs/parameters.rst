Parameters
==========

Runtime parameters are passed to :class:`radhydropy.params.Par` as a dictionary.
Missing keys are filled from :data:`radhydropy.params.refparams`.

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
   * - ``timesim``
     - Final simulation time.
     - time
   * - ``outdeltatime``
     - Output cadence.
     - time
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

Thermo-Chemistry Keys
---------------------

Thermo-chemistry is disabled by default. The active network is selected by
``thermochemistry_network``; currently the available network is ``hydrogen``.
Set ``hydrogen_chemistry=True`` to evolve the neutral hydrogen fraction
``xHI = nHI / nH`` and apply the associated line, ionization, bremsstrahlung,
and case-B recombination cooling source terms. Source terms are subcycled
inside each hydrodynamic step: the thermal equation is advanced explicitly
when enabled, then the updated temperature is used for a backward-Euler
neutral-fraction solve.

.. list-table::
   :header-rows: 1
   :widths: 28 50 22

   * - Key
     - Meaning
     - Typical unit
   * - ``thermochemistry_network``
     - Thermo-chemistry network name. Currently ``hydrogen``.
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
