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
helpers can still accept ``unyt`` objects at the boundary, but they should
move to code units or plain floats internally before repeated evaluation.

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
   par_config = config_data["par"]
   config_data["_code_units"] = CodeUnits.from_mapping(
       par_config["units"]["CodeUnits"]
   )
   ric = et.build_initial_condition(config_data)
   rio.writehdf5(ric, par_config["simulation"]["initial_condition_filename"])

   sim = Rsim(par_config)
   sim.RunAll()

This is the same pattern used by the bundled example scripts: load the YAML
file, generate ``InitialCondition.hdf5`` from the nested
``initial_condition`` section with the mandatory ``CodeUnits`` attached, then
launch the run with ``Rsim``. The helper resolves paths against the example
directory.
Gravity examples such as the hydrostatic point-mass and ballistic-infall
benchmarks follow the same pattern but also pass ``CodeUnits`` into their
analytic gravity helpers so the internal math stays float-first.

Runtime Parameters
------------------

Start with the bundled Sod-shock YAML configuration:

.. literalinclude:: ../example/SodShock1D/sodshock1d.yaml
   :language: yaml

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
  snapshots; ``savedir`` is an
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
