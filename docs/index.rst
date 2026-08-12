RadHydropy Documentation
========================

RadHydropy is a Python package for one-dimensional hydrodynamics experiments.
It focuses on idealized Cartesian and spherical problems with explicit units
through ``unyt`` and HDF5-based initial conditions and outputs.

The package is organized around a small simulation workflow:

* define run parameters with :class:`radhydropy.params.Par`;
* read or generate an HDF5 initial condition with :mod:`radhydropy.io`;
* build mesh geometry with :class:`radhydropy.mesh.Mesh`;
* initialize primitive and conserved fluid variables with
  :class:`radhydropy.fluid.Fluid`;
* advance the solution with :class:`radhydropy.solver.Solver`; and
* coordinate the full run with :class:`radhydropy.rsim.Rsim`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   parameters
   icparams
   hydrodynamics
   thermo_chemistry
   boundary_conditions
   gravity
   dark_matter
   cosmology
   initial_conditions
   snapshots
   radiative_transfer
   radiation_spectrum_generator
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
