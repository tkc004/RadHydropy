I/O
===

The public I/O façade
---------------------

Import the public HDF5 and scheduling interfaces from ``radhydropy.io``. The
implementation is split across focused modules, but callers should use this
façade so the public API remains stable:

.. code-block:: python

   import radhydropy.io as rio

The main HDF5 functions have distinct responsibilities:

* :func:`radhydropy.io.writehdf5` prepares and writes an initial-condition
  file from an assembled runtime state.
* :func:`radhydropy.io.write_snapshot_hdf5` writes an already-prepared runtime
  state directly as a snapshot.
* :func:`radhydropy.io.loadhdf5` loads either an initial condition or snapshot
  and restores its typed runtime state and metadata.

Initial-condition writer
------------------------

.. autofunction:: radhydropy.io.writehdf5

``writehdf5`` is the compatibility façade for serializing an assembled state
as an initial-condition file. The preferred example boundary is
``InitialConditionWriter.write(..., validate=True)``; see
:doc:`../initial_conditions` for the writer contract.

Runtime snapshot writer
-----------------------

.. autofunction:: radhydropy.io.write_snapshot_hdf5

Use ``write_snapshot_hdf5`` when a live, already-prepared ``Rsim`` state must
be serialized directly. Normal runs should let ``Rsim.RunAll()`` schedule
snapshots through ``par.output``.

HDF5 loader
-----------

.. autofunction:: radhydropy.io.loadhdf5

``radhydropy.io.loadhdf5(config, filename)`` is the public object-returning loader for
initial-condition files and simulation snapshots. ``config`` must be the
complete nested configuration, not only ``config["par"]``:

.. code-block:: python

   import radhydropy.io as rio
   from example_utils import load_nested_example_config

   config = load_nested_example_config("sodshock1d.yaml")
   snapshot = rio.loadhdf5(config, "Output_001.hdf5")

The returned object is an ``Rsim`` with restored ``par``, ``mesh``, and
``fluid`` components. If the file contains live dark-matter shells, it also
has a typed ``dark_matter`` analysis component with ``*_radarray`` fields;
the mutable solver shell object remains available as
``snapshot.par.dark_matter``. The loader validates the file header against
the configured coordinate system, grid size, code-unit system, cosmology,
and field representations before returning the object. See
:doc:`../snapshots` for the field layout and analysis examples.

Other public I/O interfaces
---------------------------

.. automodule:: radhydropy.io
   :members:
   :exclude-members: loadhdf5, writehdf5, write_snapshot_hdf5
