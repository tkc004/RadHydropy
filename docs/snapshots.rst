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

* ``Coordinate_System``
* ``Number_Grids``
* ``Time``
* ``BoxSize``

The ``Data`` group stores the evolved fluid fields:

* ``Boundary``
* ``Density``
* ``Velocity``
* ``Temperature``
* ``Mol_weight``
* ``NeutralFraction`` when hydrogen thermo-chemistry is active
* ``PhotonNumberDensity`` when radiative transfer is active

As with the initial-condition file, unit-bearing datasets store the unit string
in a ``units`` attribute.

Reading Snapshot Files
----------------------

Use :func:`radhydropy.io.readhdf5` to reload a snapshot into a parameter,
mesh, and fluid object. This is the same function used to load the initial
condition file.

Practical Notes
---------------

The bundled examples usually write an initial snapshot at index ``000`` and
then continue with numbered outputs as the run advances. The same HDF5 layout
lets you post-process a snapshot with the plotting helpers or restart a run
from a saved state.
