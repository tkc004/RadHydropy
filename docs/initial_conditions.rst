Initial-Condition Files
=======================

RadHydropy uses a compact HDF5 layout for initial-condition files. The
bundled example scripts generate ``InitialCondition.hdf5`` from ``ICparams``
before launching a run.

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
arrays when the file is loaded. In practice this means ``fluid.rho`` is read
back as a plain array in the runtime code-unit system. ``readhdf5`` now
requires ``Header.attrs["CodeUnits"]`` to be present and raises an error if it
is missing.

Reading and Writing
-------------------

Use :func:`radhydropy.io.writehdf5` to write an initial-condition file and
:func:`radhydropy.io.readhdf5` to load it into a simulation. The helper
functions preserve the units attached to the stored quantities, and the reader
uses the header ``CodeUnits`` block to recover the runtime unit system.

Practical Notes
---------------

The example YAML files typically point ``ICfilename`` at a file named
``InitialCondition.hdf5`` inside the example directory. The same file layout is
used by the output snapshot reader, so an output file can be reloaded with the
same HDF5 structure.
