Quickstart
==========

RadHydropy runs from a YAML example configuration plus an HDF5
initial-condition file. The high-level :class:`radhydropy.rsim.Rsim` class
reads the initial condition, prepares mesh and fluid state, advances the
solver, and writes HDF5 outputs.

The runtime requires a ``CodeUnits`` block in the nested ``par`` section. This is
mandatory: the current workflow does not fall back to cgs. Example
configurations define an internal unit system with ``InternalUnitSystem`` and
RadHydropy converts the mesh, fluid, gravity, and source-term inputs into that
code-unit system at startup. This keeps the hot paths in a consistent internal
unit space even when the YAML files are written in physical units. Example
helpers can accept ``unyt`` objects at the boundary, but they must use explicit
``.to_value(...)`` or ``quantity_to_value(...)`` conversions before repeated
evaluation; do not rely on ``float(quantity)``.

Minimum Runner
--------------

.. code-block:: python

   from pathlib import Path

   import radhydropy.io as rio
   from example_utils import load_nested_example_config
   from radhydropy.rsim import Rsim
   from radhydropy.units import CodeUnits
   import tools as et

   config = Path("example/SodShock1D/sodshock1d.yaml")
   config_data = load_nested_example_config(config)
   config_data["_code_units"] = CodeUnits.from_mapping(
       config_data["par"]["units"]["CodeUnits"]
   )
   initial = et.build_initial_condition(config_data)
   rio.writehdf5(
       initial,
       config_data["par"]["simulation"]["initial_condition_filename"],
   )

   sim = Rsim(config_data["par"])
   sim.RunAll()

This is the same pattern used by the bundled example scripts: load the YAML
file, generate ``InitialCondition.hdf5`` from the nested
``initial_condition`` section with the mandatory ``CodeUnits`` attached, then
launch the run with ``Rsim``. The helper resolves paths against the example
directory.
Gravity examples such as the hydrostatic point-mass and ballistic-infall
benchmarks follow the same pattern but also pass ``CodeUnits`` into their
analytic gravity helpers so the internal math stays float-first.

Initial-condition builder contract
-----------------------------------

Every example builder receives the complete nested ``config`` mapping. The
builder reads runtime settings from ``config["par"]`` and physical IC inputs
from ``config["initial_condition"]``; it must not receive a projected ``par``
mapping or legacy ``icparams``/``runparams`` arguments. The normal sequence is:

1. ``load_nested_example_config`` loads the YAML and converts
   ``{value, unit}`` mappings to ``unyt`` quantities.
2. ``CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])`` creates
   the configured conversion system.
3. ``build_initial_condition(config)`` converts physical inputs explicitly,
   constructs a typed ``Rsim`` state, and validates the active cells.
4. ``radhydropy.io.writehdf5`` serializes that typed state with the code-unit
   metadata in the HDF5 header.
5. ``Rsim(config["par"])`` loads and runs the same nested runtime model.

Basic proper-coordinate hydro examples may delegate their final assembly to
``example/basic_hydro_utils.py:make_initial_condition``. Its profile arrays
must already be named and converted as ``boundary_proper_code``,
``rho_proper_code``, ``vel_proper_code``, ``temp_proper_code``, and
``mu_dimensionless``. It owns ghost-cell setup, typed geometry, conserved-state
construction, active-cell/EOS consistency checks, and
``Rsim.FromComponents(...)`` serialization. Cosmological examples must build
their explicit comoving/supercomoving fields instead of using this proper-code
helper as an adapter.

Strict unit formatting
----------------------

Physical YAML values always use ``{value, unit}``; do not write bare physical
floats. YAML keys remain semantic, for example ``rho_proper``,
``temperature_proper``, ``time_cosmic``, and ``radius_outer_comoving``. Runtime
values identify their representation and units, such as
``rho_proper_code``, ``vel_supercomoving_code``, or
``temperature_cgs_K``. A unit-bearing Python value additionally ends in
``_unyt``. Convert with ``quantity_to_value`` or ``.to_value`` before passing
values to NumPy, EOS, geometry, or solver calls; ``float(quantity)`` is not a
unit conversion.

Runtime Parameters
------------------

Start with the bundled Sod-shock YAML configuration:

.. code-block:: yaml

   par:
     simulation:
       name: SodShock1d
       initial_condition_filename: InitialCondition.hdf5
       coordinate_system: cartesian
       final_time: {value: 1.0, unit: s}
     mesh:
       ghost_cells: 2
       area_proper: {value: 1.0, unit: cm**2}
     hydrodynamics:
       eos_type: polytropic
       gamma: 1.4
       CFL: 0.1
       order: 1
     boundary:
       condition: Periodic
     timestep:
       dtmin: {value: 2.0e-8, unit: s}
       dtmax: {value: 2.0e-1, unit: s}
     output:
       directory: .
       filename_prefix: Output
       cadence: {value: 0.1, unit: s}
     diagnostics:
       verbose: 0
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
     grid_cells: 1000
     coordinate_system: cartesian
     box_size_proper: {value: 4.0, unit: cm}
     time_proper: {value: 0.0, unit: s}
     rho_proper: {value: 1.0, unit: g/cm**3}
     vel_proper: {value: 0.0, unit: km/s}
     temperature_proper: {value: 1.5506894880146205e-08, unit: K}
     mean_molecular_weight: 1.0
     density_ratio: 0.1
     temperature_ratio: 0.8
   example:
     output_index: 2
     plot_filename: SodShock1D.jpg

The complete nested ``par`` block controls the solver and run lifecycle. Use
the current names below; older flat names such as ``timesim``, ``nogrid``, and
``boundcond`` are not accepted:

* ``par.simulation.name`` and ``par.simulation.initial_condition_filename``
  identify the run and IC file.
* ``par.simulation.coordinate_system`` and ``par.simulation.final_time``
  select geometry and the stopping time.
* ``par.mesh.grid_cells``, ``par.mesh.ghost_cells``, and
  ``par.mesh.area_proper`` define the mesh.
* ``par.hydrodynamics.eos_type``, ``gamma``, ``CFL``, and ``order`` define the
  fluid update.
* ``par.boundary.condition`` selects ``Periodic``, ``Open``, ``Reflecting``,
  ``OpenSph``, ``InflowSph``, or ``OutflowSph``.
* ``par.timestep.dtmin`` and ``par.timestep.dtmax`` constrain the step size.
* ``par.output.directory``, ``filename_prefix``, ``cadence`` (or
  ``time_interval``), and optional ``time_list_filename`` control saved
  snapshots; ``directory`` is an
  example-workflow output location.
* ``par.units.CodeUnits`` is mandatory and defines the internal unit system.

Unit-bearing values use ``{value, unit}`` mappings. Workflow-only values such
as plot names, output indices, comparison settings, and convergence controls
belong under ``example`` rather than ``par``. See :doc:`parameters` for the
complete runtime parameter reference.

See :doc:`initial_conditions` for a standalone description of the initial-condition
parameters used by the bundled YAML examples.

To use explicit output times instead of a fixed cadence, set
``par.output.time_list_filename`` to a txt file whose first non-empty line is
the time unit and whose remaining lines are the output times. Include
``par.simulation.final_time`` if you want the final state written as an output
snapshot. For example,
the bundled example configs typically point to files such as ``output_times.txt``:

.. code-block:: text

   yr
   0.0
   1.0e4
   2.0e4

Stepping API
------------

The high-level runner also exposes a canonical stepping interface through
:meth:`radhydropy.rsim.Rsim.Step` and :meth:`radhydropy.rsim.Rsim.Evolve`.
This keeps hydrodynamics, source terms, and output scheduling on a single code
path.

Use :meth:`radhydropy.rsim.Rsim.Step` for one controlled update:

.. code-block:: python

   dt = sim.Step(mode="hydro_sources")["dt"]

Available ``mode`` values are:

* ``"hydro"`` for a finite-volume hydrodynamic step only;
* ``"sources"`` for thermo-chemistry and radiative-transfer sources only; and
* ``"hydro_sources"`` for the coupled update used by the standard run loop.

For hydro-only steps, ``hydro_integrator="ssprk2"`` enables the optional
second-order SSP Runge-Kutta update.

Use :meth:`radhydropy.rsim.Rsim.Evolve` to advance until a target time:

.. code-block:: python

   counters = sim.Evolve(
       final_time=sim.par.simulation.final_time,
       mode="hydro_sources",
   )
   print(counters["hydro_steps"], counters["source_steps"])

The main runner helper remains :meth:`radhydropy.rsim.Rsim.Run`.

For fixed-density Stromgren-style tests, use
:meth:`radhydropy.rsim.Rsim.EvolveStaticThermochemistry`, which evolves the
static thermo-chemistry/radiative-transfer state without a hydrodynamic flux
update. See :doc:`thermo_chemistry` for a standalone description of the
thermo-chemistry solver and its example workflows.

See :doc:`hydrodynamics` for a standalone description of the finite-volume
Euler update, reconstruction order, fluxes, and boundary handling.
See :doc:`boundary_conditions` for a standalone description of the supported
boundary-condition modes and the geometry-specific ghost-cell treatment.
See :doc:`initial_conditions` for the HDF5 structure used to build
``InitialCondition.hdf5``.
See :doc:`snapshots` for the HDF5 structure written by output snapshots.

Plotting Output
---------------

After a run, load an output file and plot a fluid quantity with
:func:`radhydropy.analysis.rplot1d`:

.. code-block:: python

   from radhydropy.analysis import rplot1d
   import radhydropy.io as rio

   rio.readhdf5(sim.par, sim.mesh, sim.fluid, "Output_001.hdf5")
   rplot1d(sim, yquan="rho")
