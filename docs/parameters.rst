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
