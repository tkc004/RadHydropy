Examples
========

Example scripts live under ``example/``. They construct an initial-condition
file, run :class:`radhydropy.rsim.Rsim`, and often plot the output.

Available Examples
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Directory
     - Scenario
   * - ``example/Advection1D``
     - Cartesian one-dimensional advection.
   * - ``example/AdvectionSph1D``
     - Spherical one-dimensional advection.
   * - ``example/HIIRegionExpansion1D``
     - Spherical H II region expansion with hydrodynamics and a simplified
       piecewise-isothermal neutral/ionized equation of state.
   * - ``example/HydrostaticEquilibrium1D``
     - Cartesian isothermal atmosphere in a constant gravitational field used as
       a hydrostatic-equilibrium check.
   * - ``example/HydrostaticEquilibriumSphericalPointMass1D``
     - Spherical isothermal hydrostatic equilibrium in a point-mass potential
       without including the origin.
   * - ``example/BallisticInfallSphericalPointMass1D``
     - Spherical ballistic infall in a point-mass potential without including the
       origin.
   * - ``example/Inflow1D``
     - Cartesian inflow setup.
   * - ``example/InflowSph1D``
     - Spherical inflow setup.
   * - ``example/HydrogenCooling1D``
     - Uniform ionized hydrogen box with cooling and chemistry enabled.
   * - ``example/HydrogenRecombination1D``
     - Fixed-temperature case-B hydrogen recombination box.
   * - ``example/Outflow1d``
     - Cartesian outflow setup.
   * - ``example/OutflowSph1d``
     - Spherical outflow setup.
   * - ``example/StellarWindBubble1D``
     - Spherical stellar-wind bubble setup with a Weaver et al. 1977
       energy-driven shell reference.
   * - ``example/RadiativeTransferSph1D``
     - Spherical central-source long-characteristic radiative transfer without
       hydrodynamic or thermo-chemical evolution.
   * - ``example/StaticStromgrenSphere1D``
     - Static spherical Stromgren benchmark with constant density, temperature,
       radiative transfer, and implicit hydrogen chemistry.
   * - ``example/SedovTaylor1D``
     - Cartesian Sedov-Taylor blast-wave setup with analytic helper.
   * - ``example/SedovTaylorSph1d``
     - Spherical Sedov-Taylor blast-wave setup with analytic helper.
   * - ``example/SodShock1D``
     - Sod shock tube setup with analytic helper.


Running An Example
------------------

Run examples from their own directory so relative output paths and analytic
helper imports resolve correctly:

.. code-block:: bash

   cd example/SodShock1D
   python sodshock1d.py

Most scripts write ``InitialCondition.hdf5`` and one or more ``Output_*.hdf5``
files in the example directory.



Detailed Example Pages
----------------------

.. toctree::
   :maxdepth: 1

   stellar_wind_bubble1d
   static_stromgren_sphere1d
   static_stromgren_sphere_photoheating1d
   hii_region_expansion1d
   dynamic_stromgren_sphere_photoheating1d

