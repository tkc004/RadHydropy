Quickstart
==========

RadHydropy runs from a YAML example configuration plus an HDF5
initial-condition file. The high-level :class:`radhydropy.rsim.Rsim` class
reads the initial condition, prepares mesh and fluid state, advances the
solver, and writes HDF5 outputs.

The runtime now expects a ``CodeUnits`` block in ``runparams``. Example
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
   from radhydropy.example_config import load_example_parameters
   from radhydropy.rsim import Rsim
   import tools as et

   config = Path("example/SodShock1D/sodshock1d.yaml")
   runparams, ICparams = load_example_parameters(config)

   ric = et.Simwrap(ICparams)
   rio.writehdf5(ric, runparams["ICfilename"])

   sim = Rsim(runparams)
   sim.RunAll()

This is the same pattern used by the bundled example scripts: load the YAML
file, generate ``InitialCondition.hdf5`` from ``ICparams``, then launch the run
with ``Rsim``. The helper resolves relative ``ICfilename``, ``outdir``,
``outputtimefilename``, and ``savedir`` paths against the example directory.
Gravity examples such as the hydrostatic point-mass and ballistic-infall
benchmarks follow the same pattern but also pass ``CodeUnits`` into their
analytic gravity helpers so the internal math stays float-first.

Run Parameters
--------------

The ``runparams`` block controls how the runner loads the problem and writes
outputs. The minimum keys used by the bundled examples are:

* ``simname``: label shown in logs and filenames.
* ``ICfilename``: path to the HDF5 initial-condition file to read or write.
* ``outdir`` and ``outfileprefix``: where numbered HDF5 outputs are written.
* ``savedir``: directory for any plots or derived figures saved by the
  example script.
* ``coordsys``: geometry, usually ``cartesian`` or ``spherical``.
* ``EOStype`` and ``gamma``: equation-of-state settings.
* ``timesim``: final simulation time.
* ``outdeltatime`` or ``outputtimefilename``: fixed output cadence or explicit
  output times.
* ``CFL``, ``order``, ``dtmin``, and ``dtmax``: timestep and reconstruction
  controls.
* ``boundcond``: boundary condition name.

Units can be written inline in the YAML file using ``value`` and ``unit``
fields, as in ``timesim`` and ``outdeltatime`` in the bundled examples. See
:doc:`parameters` for the complete runtime parameter reference.

See :doc:`icparams` for a standalone description of the initial-condition
parameters used by the bundled YAML examples.

To use explicit output times instead of a fixed cadence, set
`outputtimefilename` to a txt file whose first non-empty line is the time unit
and whose remaining lines are the output times. Include the final simulation
time if you want the last state written as an output snapshot. For example,
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

   counters = sim.Evolve(final_time=sim.par.timesim, mode="hydro_sources")
   print(counters["hydro_steps"], counters["source_steps"])

The convenience wrappers remain available and now call the same canonical
implementation:

* :meth:`radhydropy.rsim.Rsim.RunOneStep`
* :meth:`radhydropy.rsim.Rsim.RunHydroStep`
* :meth:`radhydropy.rsim.Rsim.Run`

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
