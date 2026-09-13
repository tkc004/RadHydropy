Snapshot Files
==============

RadHydropy writes simulation output snapshots as HDF5 files with the same core
layout as the initial-condition file. The filenames usually follow the pattern
``Output_*.hdf5``.

File Layout
-----------

Snapshot files contain two top-level groups:

* ``Header``
* ``Data``

The ``Header`` group stores:

* the coordinate-system, grid, time, and box-size metadata;
* ``CodeUnits`` describing the complete base code-unit system;
* the active runtime configuration as header attributes; and
* optionally, a ``Provenance`` group containing the YAML configuration used
  to create the snapshot.

The ``Data`` group stores the evolved fluid fields:

* representation-qualified mesh and fluid fields such as
  ``boundary_proper_code``, ``rho_proper_code``, ``vel_proper_code``, and
  ``temp_proper_code``;
* ``Energy_code`` and ``Mass_code`` when those fields are present;
* ``mu`` and ``xHI`` for the corresponding chemistry state; and
* ``ngamma_code`` when radiative transfer is active.

Dimensional datasets declare their storage convention in the
``storage_unit`` attribute. Canonical hydrodynamic IC/snapshot fields use
``storage_unit = code`` and store numerical code values directly. Chemistry
and thermochemistry fields with a documented cgs contract may use
``storage_unit = cgs`` instead. Representation metadata is stored in
attributes such as ``quantity``, ``dimensions``, ``code_unit_cgs``,
``representation``, ``coordinate_frame``, ``physical_relation``, and
``scale_factor_power`` where applicable. ``Header/CodeUnits`` supplies the
base code-unit system for fields stored in code units. The ``_code`` suffix
identifies the runtime field and normally agrees with code-unit storage, but
the authoritative storage convention is the dataset metadata.

When a snapshot is reloaded, :func:`radhydropy.io.loadhdf5` uses the required
``Header/CodeUnits`` block and each field's ``storage_unit`` plus
representation metadata to restore typed runtime fields such as
``fluid.rho_proper_code`` or ``fluid.rho_comoving_code``.

The loader returns an ``Rsim`` object. Its main analysis-facing components are:

``snapshot.par``
   Restored runtime parameters, code units, representation metadata, and
   optional ``field_metadata`` and provenance.
``snapshot.mesh``
   Typed mesh geometry, including ``boundary_radarray`` and cell-center
   geometry derived from it.
``snapshot.fluid``
   Typed fluid fields, including ``rho_radarray``, ``vel_radarray``, and
   ``temp_radarray``. Chemistry adds ``mu`` and ``xHI``; radiative transfer
   adds photon-number fields when present.

Field formats
-------------

The canonical field names depend on the coordinate representation:

.. list-table:: Canonical restored runtime fields
   :header-rows: 1
   :widths: 30 34 36

   * - Quantity
     - Proper-coordinate file
     - Cosmological file
   * - Mesh boundary
     - ``boundary_proper_code``
     - ``boundary_comoving_code``
   * - Density
     - ``rho_proper_code``
     - ``rho_comoving_code``
   * - Velocity
     - ``vel_proper_code``
     - ``vel_supercomoving_code``
   * - Temperature
     - ``temp_proper_code``
     - ``temp_supercomoving_code``
   * - Runtime time
     - ``time_proper_code``
     - ``tau_supercomoving_code``
   * - Runtime box size
     - ``box_size_proper_code``
     - ``box_size_comoving_code``

The ``*_code`` fields are numerical solver fields. Their interpretation comes
from the field metadata and ``Header/CodeUnits``; do not infer a physical unit
from the suffix alone. For analysis, use the corresponding typed RadArray
views, which preserve units, representation, coordinate frame, and cosmology
metadata.

Optional fields include ``mu`` for mean molecular weight, ``xHI`` for neutral
hydrogen fraction, ``ngamma_code`` for photon number density, and dark-matter
shell fields when the corresponding physics is enabled. The exact datasets
present can be inspected through ``snapshot.par.field_metadata``.

Snapshot Provenance
-------------------

Snapshot provenance is optional. When supplied to
:func:`radhydropy.io.writehdf5`, it creates ``Header/Provenance`` with:

* ``source_config_yaml``: the original YAML text;
* ``effective_config_yaml``: the complete configuration after case and output
  overrides; and
* ``source_config_sha256`` and ``effective_config_sha256``: hashes of those
  two YAML documents.

The group may also contain ``source_config_filename``, ``schema_version``,
``git_commit``, ``git_dirty``, and ``initial_condition_sha256`` attributes.
For a running simulation, attach the provenance mapping to
``par.provenance`` so numbered snapshot output carries the same reproducibility
metadata. On readback, the mapping is restored as ``par.provenance``.

Reading Snapshot Files
----------------------

Use :func:`radhydropy.io.loadhdf5` with the complete nested ``config`` to
reload a snapshot into a parameter, mesh, and fluid object. This is the same
validated loader used for initial-condition files.

Analyzing a snapshot
--------------------

Use the RadArray views for physical analysis. Convert explicitly at the point
where a plotting or numerical library needs plain values:

.. code-block:: python

   import numpy as np
   import radhydropy.io as rio

   snapshot = rio.loadhdf5(config, "Output_001.hdf5")
   ghost_cells = int(snapshot.par.mesh.ghost_cells)
   active = slice(
       ghost_cells,
       ghost_cells + int(snapshot.par.mesh.grid_cells),
   )

   radius_proper_radarray = snapshot.mesh.boundary_radarray[:-1][active]
   density_proper_radarray = snapshot.fluid.rho_radarray[active]
   temperature_proper_radarray = snapshot.fluid.temp_radarray[active]

   radius_kpc = radius_proper_radarray.to_value("kpc")
   density_g_cm3 = density_proper_radarray.to_value("g/cm**3")
   temperature_K = temperature_proper_radarray.to_value("K")
   finite = (
       np.isfinite(radius_kpc)
       & np.isfinite(density_g_cm3)
       & np.isfinite(temperature_K)
   )

For a spherical snapshot, ``boundary_radarray`` has one more entry than the
cell-centered fluid fields. Use cell centers for profiles:

.. code-block:: python

   boundary_proper_radarray = snapshot.mesh.boundary_radarray
   radius_center_proper_radarray = 0.5 * (
       boundary_proper_radarray[:-1] + boundary_proper_radarray[1:]
   )
   radius_center_kpc = radius_center_proper_radarray[active].to_value("kpc")

For cosmological snapshots, the same access pattern preserves the file's
comoving or supercomoving representation. Convert to a physical quantity only
after deciding whether the diagnostic should use the stored coordinate or a
scale-factor transformation. See :doc:`cosmology` for those transformations.

To inspect available fields and their storage metadata:

.. code-block:: python

   for name, metadata in snapshot.par.field_metadata.items():
       print(name, metadata.get("storage_unit"), metadata.get("representation"))

The metadata is the authoritative description of a dataset's storage
convention. This is safer than assuming that every HDF5 array is in cgs or
that every field with a ``_code`` suffix has the same representation.

Practical Notes
---------------

The bundled examples usually write an initial snapshot at index ``000`` and
then continue with numbered outputs as the run advances. The same HDF5 layout
lets you post-process a snapshot with the plotting helpers or restart a run
from a saved state.
