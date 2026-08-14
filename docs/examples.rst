Examples
========

Example scripts live under ``example/``. They construct an initial-condition
file, run :class:`radhydropy.rsim.Rsim`, and often plot the output. Every
example uses a mandatory ``CodeUnits`` block in ``runparams`` and writes that
unit system into the HDF5 initial-condition header before the run starts.

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
     - Cartesian isothermal atmosphere in a constant gravitational field used
       as a hydrostatic-equilibrium check. The helper stays ``unyt``-friendly
       at the boundary but evaluates gravity internally in code units.
   * - ``example/HydrostaticEquilibriumSphericalPointMass1D``
     - Spherical isothermal hydrostatic equilibrium in a point-mass potential
       without including the origin. The example writes a real evolved output
       snapshot and compares it against the analytic profile.
   * - ``example/NFWHydrostaticEquilibrium1D``
     - Isothermal gas in hydrostatic equilibrium inside a ``1e8 Msun`` NFW
       dark-matter halo at its virial temperature.
   * - ``example/NFWVirialShock1D``
     - Adiabatic evolution of cosmic-mean gas in Hubble flow through the same
       ``1e8 Msun`` NFW halo, with a Rankine--Hugoniot shock diagnostic.
   * - ``example/DarkMatterFixedMassOrbit1D``
     - Fixed-enclosed-mass dark-matter shell with angular momentum, compared
       against a conserved-energy reference orbit.
   * - ``example/DarkMatterShellCrossing1D``
     - Pure self-gravitating spherical dark-matter shells with angular
       momentum, shell crossings, and radius sorting.
   * - ``example/GasDarkMatterShellCoupling1D``
     - Spherical gas coupled to live dark-matter shells through mutual
       enclosed-mass gravity.
   * - ``example/GasDarkMatterAnalyticOrbit1D``
     - Fixed uniform-gas and central-dark-matter background with an analytic
       shell time-evolution reference.
   * - ``example/EinsteinDeSitterHomogeneous1D``
     - Phase 1 homogeneous Einstein--de Sitter supercomoving expansion diagnostic.
   * - ``example/EinsteinDeSitterTopHatGravity1D``
     - Spherical supercomoving top-hat overdensity with an analytic cosmological
       self-gravity field.
   * - ``example/EinsteinDeSitterTopHatGrowth1D``
     - Cold spherical top-hat evolved as an Einstein--de Sitter linear-growth
       benchmark.
   * - ``example/EinsteinDeSitterDarkMatterShellGrowth1D``
     - Collisionless dark-matter shell background and top-hat linear-growth
       benchmark in an Einstein--de Sitter cosmology.
   * - ``example/BertschingerReference``
     - Collisionless radial Bertschinger ``epsilon=1`` similarity reference
   * - ``example/BertschingerGasReference``
     - Standalone Bertschinger ``epsilon=1``, ``gamma=5/3`` collisional-gas
       similarity equations and accretion-shock jump reference.
       profiles in an Einstein--de Sitter universe.
   * - ``example/BallisticInfallSphericalPointMass1D``
     - Spherical ballistic infall in a point-mass potential without including
       the origin. The point-mass helper now converts to code units internally
       and the example compares against the analytic free-fall profile.
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
   * - ``example/MultiFrequencyRadiativeTransferSph1D``
     - Static pure-hydrogen multifrequency radiative transfer with photoheating
       and temperature-dependent thermo-chemistry.
   * - ``example/MultiFrequencyRadiativeTransferSph1D_HHe_100Myr``
     - Five-group H/He multifrequency radiative transfer with ``X=0.75``,
       ``Y=0.25``, and a 100 Myr static thermo-chemistry evolution.
   * - ``example/MultiFrequencyRadiativeTransferSph1D_HHe_30000K_100Myr``
     - The same H/He multifrequency setup using a 30,000 K blackbody spectrum.
   * - ``example/StaticStromgrenSphere1D``
     - Static spherical Stromgren benchmark with constant density, temperature,
       radiative transfer, and implicit hydrogen chemistry.
   * - ``example/DynamicStromgrenSpherePhotoheating20pc1D``
     - Dense 20 pc dynamic photoheated Stromgren sphere with ``n_H = 100``
       cm^-3, source rate ``10^49`` s^-1, and a 1 Myr runtime.
   * - ``example/DynamicStromgrenSpherePhotoheating20pcStellarWind1D``
     - The same dense photoheated Stromgren sphere with a 10^-6 M☉ yr^-1,
       1000 km s^-1 central stellar wind.
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
files in the example directory. The typical flow is:

1. load ``runparams`` and ``ICparams`` from the example YAML file;
2. build a ``CodeUnits`` object from ``runparams["CodeUnits"]``;
3. write ``InitialCondition.hdf5`` with that ``CodeUnits`` attached; and
4. launch ``Rsim`` with the same run parameters.

If an example reloads snapshots for plotting, it should use the file header
``CodeUnits`` rather than re-parsing the YAML.



Detailed Example Pages
----------------------

.. toctree::
   :maxdepth: 1

   stellar_wind_bubble1d
   static_stromgren_sphere1d
   static_stromgren_sphere_photoheating1d
   hii_region_expansion1d
   nfw_hydrostatic_equilibrium1d
   nfw_virial_shock1d
   multifrequency_radiative_transfer_sph1d
   multifrequency_radiative_transfer_sph1d_hhe_30000k_100myr
   dynamic_stromgren_sphere_photoheating1d
   dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d
   bertschinger_reference
   bertschinger_gas_reference
