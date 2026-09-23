Getting Started
===============

This page takes you from a fresh checkout to a completed simulation and a
loaded HDF5 snapshot. RadHydropy supports Python 3.10 and newer.

Installation
------------

Create an environment, clone the repository, and install RadHydropy in
editable mode:

.. code-block:: bash

   git clone https://github.com/tkc004/RadHydropy.git
   cd RadHydropy
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

Install the optional test and documentation dependencies when developing:

.. code-block:: bash

   python -m pip install -e ".[test,docs]"

Optional physics data
---------------------

The core solver and the basic examples do not require external data. PIE and
CHIANTI/CIE examples use large HDF5 tables maintained separately in the
`RadhydropyData <https://github.com/tkc004/RadhydropyData>`_ repository. The
tables are stored with Git LFS, so install Git LFS before cloning the data
repository:

.. code-block:: bash

   cd ..
   git lfs install
   git clone https://github.com/tkc004/RadhydropyData.git
   cd RadhydropyData
   git lfs pull
   cd ../RadHydropy

The maintained example configurations conventionally resolve data from
directories next to the ``RadHydropy`` checkout:

.. code-block:: text

   parent-directory/
   ├── RadHydropy/
   ├── metal_pie_table/
   └── CHIANTI_11.0.2_database/

The PIE tables include ``metal_pie_hm12_total.h5``,
``metal_pie_hm12_metals.h5``, and ``metal_pie_table_Z1_metals.h5``. CIE
examples require ``CHIANTI_11.0.2_database/cooling_tables/`` with at least
``chianti_cie_ion_fractions.h5`` and ``chianti_cooling_table.h5``. If the
cloned data repository uses a different layout, either copy or link these
directories beside ``RadHydropy`` or override the YAML paths
``metal_pie_table_filename``, ``cie_ion_fraction_table``, and
``cie_cooling_table``. CHIANTI table-generation tools also accept an explicit
database location through ``XUVTOP`` or ``--xuvtop``.

Verify that Git LFS downloaded real HDF5 files, rather than pointer files,
before running a data-dependent example:

.. code-block:: bash

   test -s ../metal_pie_table/metal_pie_hm12_total.h5
   test -s ../CHIANTI_11.0.2_database/cooling_tables/chianti_cie_ion_fractions.h5
   test -s ../CHIANTI_11.0.2_database/cooling_tables/chianti_cooling_table.h5

See :doc:`thermo_chemistry` for network-specific table parameters and
:doc:`troubleshooting` for missing-input diagnostics.

First run
---------

Run the Sod shock-tube example from its own directory:

.. code-block:: bash

   cd example/SodShock1D
   python sodshock1d.py

The runner builds ``InitialCondition.hdf5``, evolves the problem, and writes
``Output_*.hdf5`` files and ``SodShock1D.jpg``. Example directories are the
intended launch locations because their scripts import local ``tools.py``
helpers and resolve output paths relative to the current directory.

To run a configuration variant, pass its YAML explicitly:

.. code-block:: bash

   cd ../StellarWindBubble1D
   python stellar_wind_bubble1d.py \
      --config stellar_wind_bubble1d_no_metal.yaml

See :doc:`example_matrix` for more examples and :doc:`troubleshooting` if the
runner cannot find a helper, configuration, or generated input.

Minimal Python workflow
-----------------------

The bundled examples use one nested configuration with ``par``,
``initial_condition``, and ``example`` sections. The ``par.units.CodeUnits``
mapping is required. The following is the smallest complete workflow; run it
from ``example/SodShock1D`` so that the example helper imports resolve:

.. _canonical-configuration-example:

Canonical configuration example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use explicit ``{value, unit}`` mappings for physical YAML values. This compact
configuration shows the required structure and unit-system boundary:

.. code-block:: yaml

   par:
     simulation:
       coordinate_system: cartesian
       final_time: {value: 2.0, unit: s}
     mesh:
       grid_cells: 100
       ghost_cells: 2
     hydrodynamics:
       eos_type: polytropic
       gamma: 1.4
       CFL: 0.1
     units:
       CodeUnits:
         name: cgs_unit_system
         InternalUnitSystem:
           UnitMass_in_cgs: 1.0
           UnitLength_in_cgs: 1.0
           UnitVelocity_in_cgs: 1.0
           UnitCurrent_in_cgs: 1.0
           UnitTemp_in_cgs: 1.0
   initial_condition:
     box_size_proper: {value: 1.0, unit: cm}
     rho_proper: {value: 1.0, unit: g/cm**3}
     vel_proper: {value: 0.0, unit: cm/s}
     temperature_proper: {value: 1.0, unit: K}
     mean_molecular_weight: 1.0
   example: {}

Use the complete example YAML as the source of truth for a real run; this
snippet is intentionally minimal and illustrative.

.. code-block:: python

   from pathlib import Path

   import radhydropy.io as rio
   import tools as example_tools
   from example_utils import load_nested_example_config
   from radhydropy.rsim import Rsim

   config = load_nested_example_config(Path("sodshock1d.yaml"))

   writer = example_tools.build_initial_condition(config)
   writer.write(
       config["par"]["simulation"]["initial_condition_filename"],
       validate=True,
   )

   sim = Rsim(config["par"])
   sim.RunAll()

``InitialConditionWriter`` validates and writes the initial condition. ``Rsim``
restores that state and owns the standard evolution and output schedule.
Physical YAML quantities use explicit ``{value, unit}`` mappings and should
remain unit-bearing until they cross into a numerical code-unit operation.

Snapshot inspection
-------------------

Reload a snapshot with the same complete nested configuration using
``radhydropy.io.loadhdf5()``:

.. code-block:: python

   snapshot = rio.loadhdf5(config, "Output_001.hdf5")

   radius = snapshot.mesh.boundary_radarray
   density = snapshot.fluid.rho_radarray
   density_cgs = density.to_cgs()

The ``*_radarray`` fields are typed, unit-bearing views suitable for plotting
and physical analysis. Numerical solver fields use explicit representation
names such as ``rho_proper_code`` or ``rho_comoving_code``. For example:

.. code-block:: python

   from radhydropy.analysis import rplot1d

   rplot1d(snapshot, yquan="rho")

For snapshots containing live dark matter, use typed shell views such as
``snapshot.dark_matter.radius_radarray``. Do not infer physical units from a
``_code`` suffix alone; the snapshot field metadata and ``CodeUnits`` header
are authoritative. See :doc:`snapshots` for the file contract and
:doc:`initial_conditions` for the initial-condition writer boundary.

Next steps
----------

* Read :doc:`quickstart` for output scheduling and controlled stepping.
* Read :doc:`parameters` for the nested runtime configuration.
* Browse :doc:`example_matrix` to choose a physics-focused example.
* Use :doc:`troubleshooting` for common setup and runtime failures.
