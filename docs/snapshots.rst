Snapshot Files
==============

RadHydropy writes simulation output snapshots as HDF5 files with the same core
layout as the initial-condition file. The filenames usually follow the pattern
``Output_*.hdf5``.

File Layout
-----------

Snapshot files contain the following top-level groups:

* ``Header``
* ``Data``
* ``DarkMatter`` when live shell state is present.

``Header`` group
~~~~~~~~~~~~~~~~

``Header`` contains the information needed to interpret and restart the
datasets in ``Data``.  Its contents are divided between datasets and HDF5
attributes:

Header datasets:

* ``time_proper_code`` and ``box_size_proper_code`` for a proper-coordinate
  snapshot;
* ``tau_supercomoving_code`` and ``box_size_comoving_code`` for a
  cosmological snapshot.

The ``CodeUnits`` attribute stores the complete base code-unit mapping.  It
is required by the current loader for fields stored in code units.

Header attributes commonly describe:

* geometry and mesh: ``CoordinateSystem``, ``GridCells``, and ``GhostCells``;
* the runtime state: ``ICfilename``, ``ScaleFactor``, and the saved time or
  box-size conventions;
* field interpretation: ``CoordinateFrame``, ``TimeCoordinate``,
  ``VelocityRepresentation``, ``DensityRepresentation``,
  ``PressureRepresentation``, and ``TemperatureRepresentation``;
* cosmology, when enabled: ``CosmologyType``, the reference scale and time,
  and the serialized cosmology context;
* solver and physics configuration, including the EOS, gravity,
  thermochemistry, radiation, and diagnostic settings; and
* cumulative diagnostics such as gravity work and hydro-boundary energy.

Not every attribute is present in every file.  Values may be strings,
scalars, arrays, or serialized mappings, so use ``loadhdf5`` to interpret
them as RadHydropy parameters rather than parsing attributes by hand.

``Data`` group
~~~~~~~~~~~~~~

``Data`` contains one HDF5 dataset for each saved mesh, fluid, chemistry, or
radiation field.  The dataset name is the on-disk runtime name, for example:

* geometry and hydro state: ``boundary_proper_code``, ``rho_proper_code``,
  ``vel_proper_code``, and ``temp_proper_code`` (or the corresponding
  comoving/supercomoving names);
* conserved or diagnostic fields: ``Mass_code``, ``Energy_code``, and
  ``InternalEnergy_code`` when present; and
* optional physics fields: ``mu``, ``xHI``, and ``ngamma_code`` when their
  physics modules are enabled.

Cell-centered fields have one value per active/ghost cell in the saved
layout.  A one-dimensional boundary field has one more value than the
corresponding cell-centered field because it describes cell interfaces.

Each dimensional dataset can carry attributes such as ``units``,
``storage_unit``, ``quantity``, ``dimensions``, ``code_unit_cgs``,
``representation``, ``coordinate_frame``, ``physical_relation``,
``scale_factor``, and ``scale_factor_power``.  These attributes describe how
the stored numbers are to be interpreted.  In particular, ``storage_unit``
may be ``code`` or ``cgs``; the ``_code`` suffix alone is not a sufficient
unit declaration.  ``Header.attrs["CodeUnits"]`` supplies the base scales for fields
stored in code units.

Canonical hydrodynamic IC/snapshot fields normally use
``storage_unit = code`` and store numerical code values directly. Chemistry
and thermochemistry fields with a documented cgs contract may use
``storage_unit = cgs`` instead. The ``_code`` suffix normally agrees with
code-unit storage, but the dataset metadata is authoritative.

``DarkMatter`` group
~~~~~~~~~~~~~~~~~~~~

When live spherical dark-matter shells are enabled, the snapshot contains a
top-level ``DarkMatter`` group alongside ``Header`` and ``Data``. It stores
one dataset per shell property:

.. list-table:: Dark-matter shell datasets
   :header-rows: 1
   :widths: 38 42 20

   * - Dataset
     - Meaning
     - Stored quantity
   * - ``Radius``
     - Current shell radius, sorted in shell order.
     - code length
   * - ``RadialVelocity``
     - Current radial shell velocity.
     - code velocity
   * - ``Mass``
     - Mass carried by each shell.
     - code mass
   * - ``SpecificAngularMomentum``
     - Specific angular momentum carried by each shell.
     - code length-squared/time

The group may also have a ``Softening`` attribute. The shell datasets are
Lagrangian state arrays rather than mesh fields, so they are not exposed as
``*_radarray`` mesh accessors. Their storage units are interpreted using the
snapshot's code-unit system. In a cosmological run, the shell radius and
velocity follow the run's comoving/supercomoving runtime representation; do
not interpret them as proper physical values just from the dataset names.

When a snapshot is reloaded, :func:`radhydropy.io.loadhdf5` uses
``Header.attrs["CodeUnits"]`` and each field's ``storage_unit`` plus representation
metadata to restore typed runtime fields such as ``fluid.rho_proper_code`` or
``fluid.rho_comoving_code``.

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
from the field metadata and ``Header.attrs["CodeUnits"]``; do not infer a physical unit
from the suffix alone. For analysis, use the corresponding typed RadArray
views, which preserve units, representation, coordinate frame, and cosmology
metadata.

Optional fields include ``mu`` for mean molecular weight, ``xHI`` for neutral
hydrogen fraction, and ``ngamma_code`` for photon number density. The exact
datasets present in ``Data`` can be inspected through
``snapshot.par.field_metadata``; dark-matter shell datasets are listed in the
separate ``DarkMatter`` group.

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
reload a snapshot into a parameter, mesh, and fluid object. This is the
preferred path for analysis and restart because it validates the header,
restores code units, and reconstructs the field metadata and cosmology
context.

.. code-block:: python

   import radhydropy.io as rio

   snapshot = rio.loadhdf5(config, "Output_001.hdf5")

   # Header-derived runtime information.
   print(snapshot.par.simulation.coordinate_system)
   print(snapshot.par.mesh.grid_cells)
   print(snapshot.par.mesh.ghost_cells)
   print(snapshot.par.CodeUnits)

   # Data fields as metadata-carrying RadArray values.
   boundary_radarray = snapshot.mesh.boundary_radarray
   density_radarray = snapshot.fluid.rho_radarray
   velocity_radarray = snapshot.fluid.vel_radarray
   temperature_radarray = snapshot.fluid.temp_radarray

   # Dataset metadata restored by the loader.  The exact on-disk name depends
   # on whether this is a proper or cosmological snapshot.
   for field_name, metadata in snapshot.par.field_metadata.items():
       if field_name.startswith("rho_"):
           print(field_name, metadata)

The neutral ``*_radarray`` accessors select the representation used by the
file.  For example, a cosmological file returns its comoving density and
supercomoving velocity through the same ``rho_radarray`` and ``vel_radarray``
properties.  These are ``RadArray`` objects, not plain ``unyt_array``
objects: they preserve units, field specification, coordinate frame, and
cosmology context.  Use ``to_proper()`` or ``to_comoving()`` when a
representation conversion is needed, and use ``to_value()`` or ``to_cgs()``
only at the boundary to plotting or numerical code that expects ordinary
values.

For file-layout debugging only, the raw HDF5 groups can be inspected with
``h5py``:

.. code-block:: python

   import h5py

   with h5py.File("Output_001.hdf5", "r") as handle:
       print(sorted(handle["Header"].keys()))
       print(sorted(handle["Data"].keys()))
       if "DarkMatter" in handle:
           print(sorted(handle["DarkMatter"].keys()))
       print(handle["Header"].attrs["CoordinateSystem"])
       first_data_name = next(iter(handle["Data"]))
       print(first_data_name, handle["Data"][first_data_name].attrs)

Raw ``h5py`` access returns stored numbers and HDF5 metadata; it does not
construct RadHydropy's units or cosmology-aware arrays.  Use it to inspect a
file's physical layout, then use ``loadhdf5`` for interpretation and analysis.

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

Reading dark-matter shells
--------------------------

``loadhdf5`` restores a live shell group to ``snapshot.par.dark_matter`` as a
``DarkMatterShells`` object. Its shell arrays are numerical code-unit arrays,
not ``RadArray`` objects, because they are particle-like Lagrangian state
rather than cell-centered mesh fields. Convert them explicitly with the
restored code-unit system before analysis:

.. code-block:: python

   import unyt
   import radhydropy.io as rio

   snapshot = rio.loadhdf5(config, "Output_001.hdf5")
   shells = getattr(snapshot.par, "dark_matter", None)

   if shells is not None:
       radius = unyt.unyt_array(
           shells.radius, snapshot.par.CodeUnits.length_unit
       )
       radial_velocity = unyt.unyt_array(
           shells.velocity, snapshot.par.CodeUnits.velocity_unit
       )
       mass = unyt.unyt_array(shells.mass, snapshot.par.CodeUnits.mass_unit)
       specific_angular_momentum = unyt.unyt_array(
           shells.angular_momentum,
           snapshot.par.CodeUnits.length_unit
           * snapshot.par.CodeUnits.velocity_unit,
       )

       radius_kpc = radius.to_value("kpc")
       velocity_km_s = radial_velocity.to_value("km/s")
       total_mass = mass.sum()

The raw restored mapping is also available as
``snapshot.par.dark_matter_snapshot`` with keys ``radius``, ``velocity``,
``mass``, ``angular_momentum``, and ``softening``. Use the ``DarkMatterShells``
object for shell ordering and runtime quantities; use the mapping when a
simple serialization-compatible view is sufficient. If the file has no
``DarkMatter`` group, neither live shell state nor this mapping is created.

Practical Notes
---------------

The bundled examples usually write an initial snapshot at index ``000`` and
then continue with numbered outputs as the run advances. The same HDF5 layout
lets you post-process a snapshot with the plotting helpers or restart a run
from a saved state.
