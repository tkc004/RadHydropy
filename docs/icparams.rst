Initial-Condition Parameters
============================

The ``initial_condition`` block defines the initial fluid state that gets
written to ``InitialCondition.hdf5`` before the run starts. It is separate
from the runtime ``par`` block and from the example-only ``example`` block.
The exact keys depend on the example. Current examples use explicit role and
representation names, rather than generic ``initial_*`` aliases. Common
patterns include:

* ``grid_cells``: number of active grid cells.
* ``coordinate_system``: geometry used to build the mesh.
* ``box_size_proper`` or ``box_size_comoving``: domain size.
* ``time_proper`` or ``time_cosmic``: initial time.
* ``rho_proper``, ``vel_proper``, and ``temperature_proper`` for proper-frame
  fluid states.
* ``rho_comoving``, ``vel_supercomoving``, and related cosmological names for
  expanding-background states.
* ``mean_molecular_weight`` and example-specific ratios or profile controls.

Role-specific geometric parameters follow the same convention, for example
``radius_inner_proper``, ``radius_outer_proper``, ``radius_core_proper``, and
``radius_injection_proper``. Do not reintroduce generic keys such as
``initial_radius``, ``inner_radius``, or ``outer_radius``.

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
     box_size_proper:
       value: 4.0
       unit: cm
     time_proper:
       value: 0.0
       unit: s
     rho_proper:
       value: 1.0
       unit: g/cm**3
     vel_proper:
       value: 0.0
       unit: km/s
     temperature_proper:
       value: 1.5506894880146205e-08
       unit: K
     mean_molecular_weight: 1.0
     density_ratio: 0.1
     temperature_ratio: 0.8

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
