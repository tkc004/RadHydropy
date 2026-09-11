Initial-Condition Files
=======================

RadHydropy uses a compact HDF5 layout for initial-condition files. The
bundled example scripts generate ``InitialCondition.hdf5`` from the nested
``initial_condition`` section before launching a run.

File Layout
-----------

Initial-condition files contain two top-level groups:

* ``Header``
* ``Data``

The ``Header`` group stores:

* ``Coordinate_System``
* ``Number_Grids``
* ``Time``
* ``BoxSize``
* ``CodeUnits``

The ``Data`` group stores:

* ``Boundary``
* ``Density``
* ``Velocity``
* ``Temperature``
* ``Mol_weight``
* ``NeutralFraction`` when hydrogen thermo-chemistry is enabled
* ``PhotonNumberDensity`` when radiative transfer is enabled

Unit-bearing datasets store the unit string in a ``units`` attribute.

When ``CodeUnits`` is enabled, RadHydropy writes fields such as ``Density`` in
their stored physical units and converts them back into code-unit numeric
arrays when the file is loaded. In practice this means fields are read back as
explicitly named arrays such as ``fluid.rho_proper_code`` or
``fluid.rho_comoving_code`` in the runtime code-unit system. ``readhdf5`` now
requires ``Header.attrs["CodeUnits"]`` to be present and raises an error if it
is missing.

Reading and Writing
-------------------

Use :func:`radhydropy.io.writehdf5` to write an initial-condition file and
:func:`radhydropy.io.readhdf5` to load it into a simulation. The helper
functions preserve the units attached to the stored quantities, and the reader
uses the header ``CodeUnits`` block to recover the runtime unit system.

IC builder contract
-------------------

The example-side ``build_initial_condition(config)`` function is the boundary
between a nested YAML configuration and a typed runtime state. It receives the
complete configuration mapping, including ``par``, ``initial_condition``, and
``example``; it must not receive a projected ``par`` mapping or legacy
``icparams``/``runparams`` arguments. A typical runner follows this pattern:

.. code-block:: python

   config = example_utils.load_nested_example_config(config_filename)
   config["_code_units"] = CodeUnits.from_mapping(
       config["par"]["units"]["CodeUnits"]
   )
   writer = build_initial_condition(config)
   writer.write(
       config["par"]["simulation"]["initial_condition_filename"],
   )

Builders that return an already assembled ``Rsim`` state instead use
``rio.writehdf5(state, filename)``. In both cases the builder receives the
complete nested configuration; do not project ``config["par"]`` into a flat
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

For the basic proper-coordinate hydro examples, the shared
``example/basic_hydro_utils.py`` function
``make_initial_condition(config, ...)`` provides the common finalization
boundary. Its required arrays are already converted proper-code arrays:

* ``boundary_proper_code`` has ``grid_cells + 1`` entries;
* ``rho_proper_code``, ``vel_proper_code``, ``temp_proper_code``, and
  ``mu_dimensionless`` have one entry per active cell; and
* optional ``area_proper_code`` contains one custom cell-area value per active
  cell and is used to build ``volume_proper_code``.

The helper constructs ``Rsim(config["par"])``, initializes typed mesh/fluid
geometry, builds conserved mass/momentum/energy fields, validates finite and
positive active-cell state plus EOS consistency, removes setup ghost cells,
and returns ``Rsim.FromComponents(...)``. A successful HDF5 write alone is not
an IC validation; active cells must pass these checks before the run starts.

``make_initial_condition`` currently implements the proper-code contract only.
Cosmological examples must build their explicit comoving/supercomoving mesh
and fluid fields and serialize the matching typed state; they must not use the
proper-code helper as a generic adapter. Likewise, physical YAML quantities
must be converted with ``quantity_to_value`` or ``.to_value`` before becoming
NumPy arrays—``float(quantity)`` is not a unit conversion.

After construction, write a returned writer with ``writer.write(filename)`` or
an assembled state with ``radhydropy.io.writehdf5``. For readback, call
``radhydropy.io.loadhdf5(config, filename)`` and use its ``*_radarray`` views
for dimensional mesh and fluid data.
When a provenance group is present, ``readhdf5`` restores it as
``par.provenance``; a subsequent snapshot write can reuse that metadata.
Avoid ad-hoc ``SimpleNamespace``/dynamic containers and direct snapshot
``h5py`` reads in active example workflows.

Practical Notes
---------------

The example YAML files typically point
``par.simulation.initial_condition_filename`` at a file named
``InitialCondition.hdf5`` inside the example directory. The same file layout is
used by the output snapshot reader, so an output file can be reloaded with the
same HDF5 structure.
