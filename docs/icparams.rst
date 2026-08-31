Initial-Condition Parameters
============================

The ``initial_condition`` block defines the initial fluid state that gets
written to ``InitialCondition.hdf5`` before the run starts. It is separate
from the runtime ``par`` block and from the example-only ``example`` block.
The exact keys depend on the example, but common names include:

* ``grid_cells``: number of active grid cells.
* ``coordinate_system``: geometry used to build the mesh.
* ``box_size``: physical size of the domain.
* ``current_time``: initial simulation time stored in the HDF5 file.
* ``initial_density``, ``initial_velocity``, ``initial_temperature``, and
  ``mean_molecular_weight``: initial density,
  velocity, temperature, and mean molecular weight.
* ``rhoratio`` and ``tempratio``: optional profile-shaping parameters used by
  some example setups, such as shock tubes.

Unit-bearing values are written as ``value`` / ``unit`` pairs in YAML. The
shared ``example_utils.load_nested_example_config`` helper converts them to
``unyt`` quantities before the IC builder writes the initial-condition file.

Example YAML
------------

The Advection1D example uses a nested ``initial_condition`` block like this:

.. code-block:: yaml

   initial_condition:
     grid_cells: 100
     coordinate_system: cartesian
     box_size:
       value: 4.0
       unit: cm
     current_time:
       value: 0.0
       unit: s
     initial_density:
       value: 1.0
       unit: g/cm**3
     initial_velocity:
       value: 0.0
       unit: km/s
     initial_temperature:
       value: 1.5506894880146205e-08
       unit: K
     mean_molecular_weight: 1.0
     rhoratio: 0.1
     tempratio: 0.8

These values are loaded with
``example_utils.load_nested_example_config`` and written to
``InitialCondition.hdf5`` before the simulation starts.

Example-only settings
---------------------

The optional ``example`` block contains values used only by the example
runner, such as plot filenames, output indices, analytic comparison choices,
and plotting cadence. These values must not be added to ``par`` because they
do not affect the solver runtime configuration. For example:

.. code-block:: yaml

   example:
     plot:
       filename: Advection1D.jpg
       markevery: 10
     output_indices: [0, 5]
