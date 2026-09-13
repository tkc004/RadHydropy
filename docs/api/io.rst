I/O
===

``loadhdf5``
------------

.. autofunction:: radhydropy.io.loadhdf5

``loadhdf5(config, filename)`` is the public object-returning loader for
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

All I/O members
---------------

.. automodule:: radhydropy.io
   :members:
   :exclude-members: loadhdf5
