HDF5 unit and cosmology round trip
==================================

This standalone example only writes initial-condition HDF5 files and reads
them back. It checks two proper, non-cosmological unit systems and one
cosmological/supercomoving unit system.

Run it with::

    python hdf5_unit_cosmology_roundtrip1d.py

Temporary output files are created unless ``run()`` is called with an explicit
``output_directory``.
