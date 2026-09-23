Initial-condition files
=======================

RadHydropy uses the same representation-aware HDF5 schema for initial
conditions and runtime snapshots. The bundled example scripts generate
``InitialCondition.hdf5`` from the nested ``initial_condition`` section before
launching a run. The initial-condition file is therefore a valid input to the
same loader and field-restoration path used for snapshots.

File layout
-----------

Canonical schema
~~~~~~~~~~~~~~~~

The following is the authoritative current schema. Names in the ``proper``
and ``cosmological`` columns are alternatives: a file must use one complete
representation, not a mixture of both.

.. list-table:: Canonical RadHydropy HDF5 schema
   :header-rows: 1
   :widths: 24 38 38

   * - Location
     - Proper-coordinate file
     - Cosmological file
   * - Top-level groups
     - ``Header``, ``Data``; optional ``DarkMatter``
     - ``Header``, ``Data``; optional ``DarkMatter``
   * - Header datasets
     - ``time_proper_code``, ``box_size_proper_code``
     - ``tau_supercomoving_code``, ``box_size_comoving_code``
   * - Required header attribute
     - ``CodeUnits``
     - ``CodeUnits``
   * - Header representation attributes
     - ``CoordinateSystem``, ``GridCells``, ``GhostCells``,
       ``CoordinateFrame``, ``TimeCoordinate``, and field representation
       attributes
     - The same attributes, with cosmology metadata such as
       ``CosmologyType``, ``ScaleFactor``, and ``CosmicTime``
   * - Primary ``Data`` fields
     - ``boundary_proper_code``, ``rho_proper_code``, ``vel_proper_code``,
       ``temp_proper_code``
     - ``boundary_comoving_code``, ``rho_comoving_code``,
       ``vel_supercomoving_code``, ``temp_supercomoving_code``
   * - Optional ``Data`` fields
     - Conserved, chemistry, radiation, and diagnostic fields such as
       ``Mass_code``, ``Energy_code``, ``mu``, ``xHI``, and ``ngamma_code``
     - The same optional fields, with the applicable representation metadata
   * - Dataset metadata
     - ``storage_unit`` and field metadata such as ``quantity``,
       ``representation``, and ``coordinate_frame``
     - The same metadata, including cosmology conversion information
   * - Optional provenance
     - ``Header/Provenance``
     - ``Header/Provenance``

``CodeUnits`` is stored as a ``Header`` attribute, not as a dataset. The
loader requires it when reading code-unit fields. Dataset metadata is
authoritative for interpreting stored values; the ``_code`` suffix alone does
not define a unit.

``Header`` stores scalar runtime and cosmology metadata as attributes. The
time and box-size datasets identify the representation, while the field names
in ``Data`` identify the representation of mesh and fluid arrays. Optional
``DarkMatter`` contains live-shell state when that physics module is enabled;
its datasets and typed analysis views are documented in :doc:`snapshots`.

Initial-condition files contain the common top-level groups:

* ``Header``
* ``Data``

Each dataset carries a ``storage_unit`` attribute. Canonical hydrodynamic
fields normally use ``storage_unit = code``; fields with a documented physical
storage contract may use ``storage_unit = cgs``. On load, canonical fields are
restored as explicitly named runtime arrays such as
``fluid.rho_proper_code`` or ``fluid.rho_comoving_code``.

Reading and Writing
-------------------

Use :class:`radhydropy.initial_condition_writer.InitialConditionWriter` for
the preferred writer-backed path. Write the returned writer with
``writer.write(filename, validate=True)`` and load the initial condition or a
snapshot through :func:`radhydropy.io.loadhdf5` with the complete nested
configuration. The loader uses the header ``CodeUnits`` block to recover the
runtime unit system and returns typed mesh/fluid fields.

An already assembled typed ``Rsim`` state may still be serialized as an
initial-condition file with :func:`radhydropy.io.writehdf5`; this is a
separate state-serialization path, not the preferred example initial-condition
boundary. For a runtime snapshot, use
:func:`radhydropy.io.write_snapshot_hdf5` instead.

Initial-condition builder contract
----------------------------------

The example-side ``build_initial_condition(config)`` function is the boundary
between a nested YAML configuration and a typed runtime state. It receives the
complete configuration mapping, including ``par``, ``initial_condition``, and
``example`` and selects the appropriate sections internally. A typical runner
follows this pattern:

.. code-block:: python

   config = example_utils.load_nested_example_config(config_filename)
   writer = build_initial_condition(config)
   writer.write(
       config["par"]["simulation"]["initial_condition_filename"],
   )

Builders that return an already assembled ``Rsim`` state instead use
``rio.writehdf5(state, filename)`` for the initial-condition file. In both
cases the builder receives the complete nested configuration; do not project
``config["par"]`` into a flat
initial-condition mapping.

The optional ``provenance`` mapping writes a ``Header/Provenance`` group to
both initial-condition files and snapshots. It stores the original YAML as
``source_config_yaml`` and the configuration actually used after case/output
overrides as ``effective_config_yaml``. The group records separate
``source_config_sha256`` and ``effective_config_sha256`` attributes, so the
source file and effective run configuration can be verified independently.
Optional ``git_commit``, ``git_dirty``, and
``initial_condition_sha256`` metadata may also be supplied. Private runtime
objects and keys beginning with ``_`` are excluded from serialized effective
configuration YAML.

``build_initial_condition`` owns the example-specific work: it reads physical
inputs from ``config["initial_condition"]``, converts them explicitly to the
configured code-unit scale, constructs analytic profiles or source fields,
and returns either an ``InitialConditionWriter`` or an already assembled typed
``Rsim`` state. Runtime-only objects that cannot be written in YAML may be
attached to the complete configuration under a descriptive private key at the
call site.

Physical YAML quantities must be converted with ``quantity_to_value`` or
``.to_value`` before becoming NumPy arrays—``float(quantity)`` is not a unit
conversion. Cosmological examples retain their explicit comoving or
supercomoving representation through initial-condition construction and serialization.

After construction, write a returned writer with ``writer.write(filename)`` or
an assembled state with ``radhydropy.io.writehdf5``. For readback, call
``radhydropy.io.loadhdf5(config, filename)`` and use its ``*_radarray`` views
for dimensional mesh and fluid data.
When a provenance group is present, ``radhydropy.io.loadhdf5()`` restores it as
``par.provenance``; a subsequent snapshot write can reuse that metadata.
Avoid ad-hoc ``SimpleNamespace``/dynamic containers and direct snapshot
``h5py`` reads in active example workflows.

Practical Notes
---------------

The example YAML files typically point
``par.simulation.initial_condition_filename`` at a file named
``InitialCondition.hdf5`` inside the example directory. The same file layout is
used by the output snapshot reader, so an output file can be reloaded with the
same HDF5 structure. For the full ``Header``, ``Data``, and ``DarkMatter``
field details, see :doc:`snapshots`; this page defines which parts apply to
initial conditions as well as snapshots.
