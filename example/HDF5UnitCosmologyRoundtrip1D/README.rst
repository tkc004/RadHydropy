HDF5 unit and cosmology round trip
==================================

This standalone example only writes initial-condition HDF5 files and reads
them back. It checks two proper, non-cosmological unit systems and one
cosmological/supercomoving unit system.

Run it with::

    python hdf5_unit_cosmology_roundtrip1d.py

Temporary output files are created unless ``run()`` is called with an explicit
``output_directory``.
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/HDF5UnitCosmologyRoundtrip1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
