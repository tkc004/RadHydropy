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
``fluid.rho_comoving_code`` in the runtime code-unit system. The loader
requires ``Header.attrs["CodeUnits"]`` to be present and raises an error if it
is missing.

Reading and Writing
-------------------

Use :class:`radhydropy.initial_condition_writer.InitialConditionWriter` for
the preferred writer-backed path. Write the returned writer with
``writer.write(filename, validate=True)`` and load the initial condition or a
snapshot through :func:`radhydropy.io.loadhdf5` with the complete nested
configuration. The loader uses the header ``CodeUnits`` block to recover the
runtime unit system and returns typed mesh/fluid fields.

An already assembled typed ``Rsim`` state may still be serialized with
:func:`radhydropy.io.writehdf5`; this is a separate state-serialization path,
not the preferred example IC-construction boundary.

IC builder contract
-------------------

The example-side ``build_initial_condition(config)`` function is the boundary
between a nested YAML configuration and a typed runtime state. It receives the
complete configuration mapping, including ``par``, ``initial_condition``, and
``example`` and selects the appropriate sections internally. A typical runner
follows this pattern:

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

Physical YAML quantities must be converted with ``quantity_to_value`` or
``.to_value`` before becoming NumPy arrays—``float(quantity)`` is not a unit
conversion. Cosmological examples retain their explicit comoving or
supercomoving representation through IC construction and serialization.

After construction, write a returned writer with ``writer.write(filename)`` or
an assembled state with ``radhydropy.io.writehdf5``. For readback, call
``radhydropy.io.loadhdf5(config, filename)`` and use its ``*_radarray`` views
for dimensional mesh and fluid data.
When a provenance group is present, ``loadhdf5`` restores it as
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
