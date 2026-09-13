Quickstart
==========

RadHydropy examples use three nested YAML sections and a validated HDF5
initial-condition file:

``par``
   Runtime, mesh, solver, output, and code-unit settings.
``initial_condition``
   Physical inputs used to construct the initial state.
``example``
   Plotting, comparison, and other workflow-specific settings.

Every maintained example provides a complete configuration with a mandatory
``par.units.CodeUnits`` block.

Run an example
--------------

The quickest way to verify an installation is to run the Sod shock-tube
example from its own directory:

.. code-block:: bash

   cd example/SodShock1D
   python sodshock1d.py

The script builds an initial condition, runs the solver, and writes HDF5
snapshots and a figure in the example directory. Examples with variants accept
an explicit configuration, for example:

.. code-block:: bash

   cd example/StellarWindBubble1D
   python stellar_wind_bubble1d.py \
      --config stellar_wind_bubble1d_no_metal.yaml

Run examples from their own directories so relative configuration paths,
output directories, and helper imports resolve correctly.

The standard workflow
---------------------

The example lifecycle is:

.. code-block:: text

   nested YAML
       |
       v
   InitialConditionWriter -- writer.write(validate=True) --> InitialCondition.hdf5
       |
       v
   Rsim(config["par"]).RunAll() --> Output_*.hdf5
       |
       v
   loadhdf5(config, snapshot) --> typed runtime state and *_radarray views

The builder prepares the initial condition; ``Rsim`` evolves it; ``loadhdf5``
reloads an IC or snapshot for inspection or restart.

Minimal runner
--------------

This is the complete pattern used by the bundled examples. Run the snippet
from ``example/SodShock1D``:

.. code-block:: python

   from pathlib import Path

   import radhydropy.io as rio
   from example_utils import load_nested_example_config
   from radhydropy.rsim import Rsim
   from radhydropy.units import CodeUnits
   import tools as et

   config = Path("sodshock1d.yaml")
   config_data = load_nested_example_config(config)
   config_data["_code_units"] = CodeUnits.from_mapping(
       config_data["par"]["units"]["CodeUnits"]
   )

   writer = et.build_initial_condition(config_data)
   writer.write(
       config_data["par"]["simulation"]["initial_condition_filename"],
       validate=True,
   )

   sim = Rsim(config_data["par"])
   sim.RunAll()

The builder receives the complete nested configuration and normally returns an
``InitialConditionWriter``. Physical values remain unit-bearing until they are
assigned to the writer. The writer validates the active mesh and fluid state
before serialization.

Inspect a snapshot
------------------

Load snapshots with the complete nested configuration. Use typed
``*_radarray`` fields for dimensional mesh and fluid data, and convert to
plain numerical values only at an explicit plotting or numerical boundary:

.. code-block:: python

   import radhydropy.io as rio
   from radhydropy.analysis import rplot1d

   snapshot = rio.loadhdf5(config_data, "Output_001.hdf5")
   radius_proper_unyt = snapshot.mesh.boundary_radarray
   density_proper_unyt = snapshot.fluid.rho_radarray
   density_cgs_g_cm3 = density_proper_unyt.to_cgs()

   rplot1d(snapshot, yquan="rho")

The canonical runtime fields, such as ``rho_proper_code`` or
``rho_comoving_code``, are numerical solver state. Their suffix identifies the
representation; do not replace it with generic names such as ``density`` in
new diagnostics.

Configuration rules
-------------------

Physical YAML values use ``{value, unit}`` mappings:

.. code-block:: yaml

   par:
     simulation:
       final_time: {value: 1.0, unit: s}
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
     temperature_proper: {value: 1.0e4, unit: K}

Use explicit representation names for physical values, such as
``rho_proper``, ``temperature_proper``, ``time_cosmic``, and
``radius_outer_comoving``. In Python, convert unit-bearing values with
``quantity_to_value`` or ``.to_value``; ``float(quantity)`` is not a unit
conversion.

The most important runtime owners are:

* ``par.simulation``: run name, coordinate system, IC filename, and final time;
* ``par.mesh``: grid size, ghost cells, and geometry;
* ``par.hydrodynamics``: EOS, ``gamma``, CFL, and reconstruction ``order``;
* ``par.boundary``: Cartesian or spherical boundary condition;
* ``par.timestep``: minimum and maximum timestep controls;
* ``par.output``: output directory, filename prefix, and cadence or time list;
* ``par.units.CodeUnits``: the required internal unit system.

Maintained spherical examples use ``OutflowSph``. The
``StellarWindBubble1D`` no-metal configuration is validated with
hydrodynamics ``order: 0``.

Advanced control
----------------

Use ``Step`` for one controlled update and ``Evolve`` for a custom evolution
loop:

.. code-block:: python

   result = sim.Step(mode="hydro_sources")
   print(result["dt"])

   counters = sim.Evolve(
       final_time=sim.par.simulation.final_time,
       mode="hydro_sources",
   )
   print(counters)

Available modes are ``"hydro"``, ``"sources"``, and
``"hydro_sources"``. For fixed-density Stromgren-style tests,
``Rsim.EvolveStaticThermochemistry(...)`` advances thermo-chemistry and
radiative-transfer sources without a hydrodynamic flux update.

To schedule explicit output times, set ``par.output.time_list_filename`` to a
text file whose first non-empty line is the time unit and whose remaining lines
are output times. Include ``par.simulation.final_time`` when the final state
should be written.

See :doc:`initial_conditions` for the HDF5 IC contract,
:doc:`snapshots` for output files, :doc:`parameters` for the complete nested
runtime reference, and :doc:`examples` for runnable workflows.
