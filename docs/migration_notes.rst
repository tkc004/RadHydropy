Migration Notes
===============

This page collects development-facing migration records that affect example
workflows and public boundaries.

Current workflow direction
--------------------------

Maintained examples should follow the initial-condition-driven lifecycle:

.. code-block:: text

   nested YAML
       |
       v
   InitialConditionWriter -> writer.write(validate=True)
       |
       v
   Rsim(config["par"]).RunAll()
       |
       v
   HDF5 snapshots -> radhydropy.io.loadhdf5() -> typed RadArray diagnostics

Use ``radhydropy.io/`` as the persistence boundary. Use
``writehdf5()`` for an initial-condition file and
``write_snapshot_hdf5()`` for direct runtime snapshot serialization. Example
diagnostics should consume restored typed fields, especially the
``*_radarray`` views, instead of reading HDF5 datasets directly.

Cosmological workflows must preserve the structured ``par.cosmology``
container, use representation-specific runtime fields, and keep
``cosmological``, ``cosmological_expansion``, and
``supercomoving_coordinates`` consistent for supercomoving runs.

For current behavior, consult :doc:`architecture`, :doc:`snapshots`,
:doc:`initial_conditions`, and :doc:`validation`.
