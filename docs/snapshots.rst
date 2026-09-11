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

Dimensional datasets currently store numerical values in their physical cgs
units, with the unit string in the ``units`` attribute. Representation metadata
is stored in attributes such as ``quantity``, ``representation``,
``coordinate_frame``, ``physical_relation``, and ``scale_factor_power`` where
applicable. ``Header/CodeUnits`` supplies the conversion to runtime code units.
For example, ``Energy_code`` is stored in ``erg`` and uses
``UnitMass * UnitVelocity**2`` as its code-unit scale, while
``vel_proper_code`` is stored in ``cm/s`` and uses ``UnitVelocity``. The
``_code`` suffix identifies the runtime field restored by ``readhdf5``; it
does not currently mean that the on-disk numerical values are code-unit
values. Cosmological fields identify their proper, comoving, or
supercomoving representation in the dataset name and metadata.

When a snapshot is reloaded, :func:`radhydropy.io.readhdf5` uses the required
``Header/CodeUnits`` block and the representation metadata to convert the
stored values and restore typed runtime fields such as
``fluid.rho_proper_code`` or ``fluid.rho_comoving_code``.

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

Use :func:`radhydropy.io.readhdf5` with ``Rsim(config["par"])`` to reload a
snapshot into a parameter, mesh, and fluid object. This is the same function
used to load the initial-condition file.

Practical Notes
---------------

The bundled examples usually write an initial snapshot at index ``000`` and
then continue with numbered outputs as the run advances. The same HDF5 layout
lets you post-process a snapshot with the plotting helpers or restart a run
from a saved state.
