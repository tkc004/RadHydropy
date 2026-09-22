Reproducible Runs
=================

Reproducibility in RadHydropy has two parts: preserve the inputs and code
identity, and preserve the generated state and diagnostics. The HDF5 writer
stores the first part under ``Header/Provenance`` so an initial condition and
its numbered snapshots can be traced back to the same run configuration.

What to preserve
----------------

Record these items for every published or validated run:

* the RadHydropy Git commit and whether the worktree was dirty;
* the original YAML text and the effective configuration after runtime or
  output overrides;
* the SHA-256 hashes of those two YAML documents;
* the SHA-256 hash of ``InitialCondition.hdf5``;
* the names and hashes of external inputs such as PIE, CHIANTI, and generated
  cosmological correlation tables; and
* the final time, snapshot filenames, figures, CSV/NPZ diagnostics, and any
  non-fatal warnings.

The source YAML and effective YAML are stored as datasets. Their hashes and
the remaining run metadata are stored as attributes:

.. list-table:: ``Header/Provenance`` contents
   :header-rows: 1
   :widths: 34 66

   * - Entry
     - Meaning
   * - ``source_config_yaml``
     - Original YAML text supplied by the user.
   * - ``effective_config_yaml``
     - Complete configuration used by the run after overrides; private keys
       beginning with ``_`` are omitted.
   * - ``source_config_sha256`` / ``effective_config_sha256``
     - SHA-256 hashes of the two stored YAML documents.
   * - ``schema_version``
     - Version of the caller’s provenance record.
   * - ``source_config_filename``
     - Original configuration filename, when available.
   * - ``git_commit`` / ``git_dirty``
     - Code revision and whether uncommitted changes were present.
   * - ``initial_condition_sha256``
     - SHA-256 hash of the serialized initial-condition file.

Attach provenance at the writer boundary
-----------------------------------------

The writer accepts a mapping containing the original YAML and effective
configuration. The following pattern shows the important boundary; example
builders may differ in how they construct ``writer``:

.. code-block:: python

   from hashlib import sha256
   from pathlib import Path

   source_yaml_path = Path("sodshock1d.yaml")
   ic_filename = Path("InitialCondition.hdf5")
   provenance = {
       "source_config_yaml": source_yaml_path.read_text(),
       "effective_config": config,
       "schema_version": 1,
       "source_config_filename": str(source_yaml_path),
       "git_commit": "<40-character commit>",
       "git_dirty": False,
   }

   # build_initial_condition(config) returns an InitialConditionWriter.
   writer = build_initial_condition(config)
   writer.provenance = provenance
   writer.write(ic_filename, validate=True)

   provenance["initial_condition_sha256"] = sha256(
       ic_filename.read_bytes(),
   ).hexdigest()

The initial-condition hash is computed after the file is written. Attach the
completed mapping to the runtime parameters before producing numbered output
so subsequent snapshots carry the same provenance:

.. code-block:: python

   sim = Rsim(config["par"])
   sim.par.provenance = provenance
   sim.RunAll()

``InitialConditionWriter.write(..., validate=True)`` is the preferred writer
boundary, ``Rsim.RunAll()`` owns normal snapshot scheduling, and
``radhydropy.io.write_snapshot_hdf5()`` preserves ``par.provenance`` when a
live state must be serialized directly.

Obtaining Git metadata
----------------------

Populate ``git_commit`` and ``git_dirty`` from the exact checkout used for the
run. For example:

.. code-block:: bash

   git rev-parse HEAD
   test -z "$(git status --porcelain)"; echo $?

The second command prints ``0`` for a clean worktree and ``1`` when changes are
present. A dirty run is still reproducible when the complete diff is archived
with the run record; do not label it as source-identical to the recorded
commit.

Hashing external inputs and artifacts
-------------------------------------

Hash every external table and generated input used by the selected YAML. From
the repository root, for example:

.. code-block:: bash

   shasum -a 256 \
      ../metal_pie_table/metal_pie_hm12_total.h5 \
      ../CHIANTI_11.0.2_database/cooling_tables/chianti_cie_ion_fractions.h5 \
      ../CHIANTI_11.0.2_database/cooling_tables/chianti_cooling_table.h5

For cosmological correlation workflows, include the generated correlation HDF5
table as well. Record the exact path, table-generation command, and hash; a
missing or regenerated table can change the initial condition without changing
the YAML.

The run record should also list the final time and every expected artifact,
including ``InitialCondition.hdf5``, ``Output_*.hdf5``, figures, CSV/NPZ audit
files, and warning policies such as ``c2ray_nonconvergence: warn``. The
validation checklist in :doc:`validation` provides the minimum reporting
fields.

Reading provenance back
-----------------------

Load the HDF5 file through the public loader. The provenance mapping is
restored as ``snapshot.par.provenance``:

.. code-block:: python

   import radhydropy.io as rio

   snapshot = rio.loadhdf5(config, "Output_001.hdf5")
   provenance = snapshot.par.provenance
   print(provenance["git_commit"])
   print(provenance["effective_config_sha256"])

Use ``h5py`` only when inspecting the serialized file layout itself. Analysis
should use the restored typed ``RadArray`` fields from the loaded snapshot.
