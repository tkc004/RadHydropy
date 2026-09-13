Run Parameters
==============

RadHydropy runtime parameters are read from the nested ``par`` mapping. The
complete annotated template, including defaults and available fields, is
maintained in :doc:`all_parameters_default` and
``example/all_parameters_default.yaml``.

Use this page to understand configuration ownership and the most common
settings. Detailed behavior belongs to the subsystem pages linked below.

Configuration model
-------------------

An example configuration has three top-level sections:

``par``
   Runtime, mesh, solver, output, and code-unit parameters.
``initial_condition``
   Physical inputs used by the initial-condition builder.
``example``
   Plotting, comparison, and workflow-specific settings.

The runtime parameter path always includes its group. For example, the CFL
number is ``par.hydrodynamics.CFL`` and the radiative-transfer switch is
``par.radiation.radiative_transfer``.

Required code units
-------------------

Every run defines ``par.units.CodeUnits``. The unit system is written into the
HDF5 initial-condition header and is used to restore typed runtime fields when
an IC or snapshot is loaded.

Physical YAML values use ``{value, unit}`` mappings:

.. code-block:: yaml

   par:
     simulation:
       coordinate_system: cartesian
       final_time: {value: 2.0, unit: s}
     mesh:
       grid_cells: 100
       ghost_cells: 2
     hydrodynamics:
       eos_type: polytropic
       gamma: 1.4
       CFL: 0.1
     units:
       CodeUnits:
         name: cgs_unit_system
         InternalUnitSystem:
           UnitMassInCGS: 1.0
           UnitLengthInCGS: 1.0
           UnitVelocityInCGS: 1.0
           UnitCurrentInCGS: 1.0
           UnitTempInCGS: 1.0

   initial_condition:
     grid_cells: 100
     coordinate_system: cartesian
     box_size_proper: {value: 1.0, unit: cm}
     time_proper: {value: 0.0, unit: s}
     rho_proper: {value: 1.0, unit: g/cm**3}
     vel_proper: {value: 0.0, unit: cm/s}
     temperature_proper: {value: 1.0, unit: K}
     mean_molecular_weight: 1.0

Use semantic physical keys such as ``rho_proper``, ``vel_proper``,
``temperature_proper``, ``time_cosmic``, and ``radius_outer_comoving``.
Convert unit-bearing Python values with ``quantity_to_value`` or
``.to_value(...)`` before passing them to numerical routines.

Runtime groups
--------------

The following table lists the canonical owners for the main runtime settings.
The full field-level reference remains in
``example/all_parameters_default.yaml``.

.. list-table:: Canonical nested runtime groups
   :header-rows: 1
   :widths: 25 35 40

   * - Group
     - Common fields
     - Purpose
   * - ``par.simulation``
     - ``name``, ``initial_condition_filename``, ``coordinate_system``, ``final_time``
     - Run identity, geometry, IC path, and stopping time.
   * - ``par.mesh``
     - ``grid_cells``, ``ghost_cells``, ``area_proper``
     - Active resolution, ghost zones, and Cartesian cell area.
   * - ``par.hydrodynamics``
     - ``eos_type``, ``gamma``, ``CFL``, ``order``, ``riemann_solver``
     - Equation of state, reconstruction, fluxes, and hydro integration.
   * - ``par.boundary``
     - ``condition``, inflow/outflow primitive fields
     - Ghost-cell boundary treatment and reservoir states.
   * - ``par.timestep``
     - ``dtmin``, ``dtmax``, source and cooling controls
     - Hydro, source, chemistry, and cosmological timestep limits.
   * - ``par.output``
     - ``directory``, ``filename_prefix``, ``cadence``, ``time_list_filename``
     - Snapshot destination and output scheduling.
   * - ``par.units``
     - ``CodeUnits`` and ``InternalUnitSystem``
     - Required internal unit scales.
   * - ``par.diagnostics``
     - ``verbose``, plot limits, energy diagnostics
     - Runtime logging and optional diagnostic histories.
   * - ``par.thermochemistry``
     - ``network``, cooling, chemistry, Compton, and source-solver controls
     - Thermal and chemical source networks. See :doc:`thermo_chemistry`.
   * - ``par.chemistry``
     - composition fractions and species-state controls
     - Initial and boundary composition for thermo-chemistry.
   * - ``par.gravity``
     - self-gravity, external gravity, and cosmology controls
     - Gas gravity and cosmological background. See :doc:`gravity` and
       :doc:`cosmology`.
   * - ``par.dark_matter``
     - shell crossing, softening, and timestep controls
     - Live dark-matter shell evolution. See :doc:`dark_matter`.
   * - ``par.radiation``
     - transport, source, spectrum, multigroup, and pressure controls
     - Radiative transfer and radiation pressure. See
       :doc:`radiative_transfer` and :doc:`radiation_pressure`.

Boundary parameters
-------------------

Spherical inflow and outflow conditions use these fields under
``par.boundary``:

.. list-table:: Spherical boundary fields
   :header-rows: 1
   :widths: 34 46 20

   * - Field
     - Meaning
     - Unit
   * - ``rho_inflow_proper`` / ``rho_outflow_proper``
     - Density imposed in the corresponding ghost cells.
     - mass density
   * - ``vel_inflow_proper`` / ``vel_outflow_proper``
     - Velocity imposed in the corresponding ghost cells.
     - velocity
   * - ``temperature_inflow_proper`` / ``temperature_outflow_proper``
     - Temperature used to derive boundary pressure.
     - temperature
   * - ``inflow_mu`` / ``outflow_mu``
     - Mean molecular weight used to derive boundary pressure.
     - dimensionless

Maintained spherical examples use ``OutflowSph``. The
``StellarWindBubble1D`` no-metal configuration uses hydrodynamics ``order: 0``.
See :doc:`boundary_conditions` for the complete boundary behavior.

Common configuration recipes
-----------------------------

Enable hydrogen thermo-chemistry:

.. code-block:: yaml

   par:
     thermochemistry:
       network: hydrogen
       hydrogen_chemistry: true
       hydrogen_thermal_coupling: true
     chemistry:
       key: H

Use CIE cooling:

.. code-block:: yaml

   par:
     thermochemistry:
       network: cie_cooling
       cie_cooling: true
       metallicity: 1.0
       cie_ion_fraction_table: path/to/ion_fractions.h5
       cie_cooling_table: path/to/cooling_table.h5

Enable long-characteristic radiative transfer:

.. code-block:: yaml

   par:
     radiation:
       radiative_transfer: true
       radiative_transfer_method: long_characteristics
       radiative_transfer_temporal_scheme: c2ray
       source_photon_rate: {value: 5.0e48, unit: 1/s}

Schedule explicit output times:

.. code-block:: yaml

   par:
     output:
       time_list_filename: output_times.txt

The first non-empty line of the time-list file is its unit, followed by one
time per line:

.. code-block:: text

   yr
   0.0
   1.0e4
   2.0e4

Include the configured final time when the final state should be written.

Cosmological parameters
-----------------------

Cosmological runs keep their coordinate and variable representations explicit
in ``par.simulation`` and ``par.gravity``. A minimal Einstein--de Sitter setup
is:

.. code-block:: yaml

   par:
     simulation:
       coordinate_frame: comoving
       time_coordinate: supercomoving
       velocity_representation: supercomoving_peculiar
       density_representation: comoving
       pressure_representation: supercomoving
       temperature_representation: supercomoving
     gravity:
       cosmological_expansion: true
       cosmological_gravity: true
       supercomoving_coordinates: true
       cosmology_type: einstein_de_sitter
       cosmology_t_ref: {value: 1.0, unit: Myr}
       cosmology_a_ref: 1.0

See :doc:`cosmology` for scale-factor conventions and cosmological initial
conditions. Gas angular momentum and dark-matter shell parameters are covered
in :doc:`dark_matter` and the relevant example pages.

Initial conditions and snapshots
--------------------------------

Initial-condition construction is documented in :doc:`initial_conditions`.
The current example boundary is ``InitialConditionWriter``; use
``writer.write(filename, validate=True)`` before starting ``Rsim``.

For post-processing, load an IC or snapshot with the complete nested
configuration and use typed ``*_radarray`` views:

.. code-block:: python

   import radhydropy.io as rio

   snapshot = rio.loadhdf5(config, "Output_001.hdf5")
   radius_proper_unyt = snapshot.mesh.boundary_radarray
   density_proper_unyt = snapshot.fluid.rho_radarray

See :doc:`snapshots` for the HDF5 layout, field metadata, and provenance.

Source and diagnostic references
--------------------------------

* :doc:`thermo_chemistry` — hydrogen, H/He, CIE, PIE, and Compton sources.
* :doc:`radiative_transfer` — long-characteristic and C²-Ray transport.
* :doc:`radiation_pressure` — absorbed-photon momentum sources.
* :doc:`gravity` — self-gravity and external gravitational fields.
* :doc:`dark_matter` — live dark-matter shell parameters.
* :doc:`hydrodynamics` — reconstruction, fluxes, and timestep behavior.
* :doc:`boundary_conditions` — Cartesian and spherical ghost-cell modes.
* :doc:`examples` — runnable configurations and detailed example pages.
