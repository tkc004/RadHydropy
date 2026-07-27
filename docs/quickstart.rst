Quickstart
==========

RadHydropy runs from a YAML example configuration plus an HDF5
initial-condition file. The high-level :class:`radhydropy.rsim.Rsim` class
reads the initial condition, prepares mesh and fluid state, advances the
solver, and writes HDF5 outputs.

YAML-Driven Example Runner
--------------------------

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

The bundled example scripts follow the same pattern: load the YAML file,
generate ``InitialCondition.hdf5`` from ``ICparams``, then launch the run with
``Rsim``. The helper resolves relative ``ICfilename``, ``outdir``,
``outputtimefilename``, and ``savedir`` paths against the example directory.

To use explicit output times instead of a fixed cadence, set
`outputtimefilename` to a txt file whose first non-empty line is the time unit
and whose remaining lines are the output times. For example, the bundled
example configs typically point to files such as ``output_times.txt``:

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
update.

Initial-Condition Files
-----------------------

Initial-condition files use a compact HDF5 layout:

* ``Header`` contains ``Coordinate_System``, ``Number_Grids``, ``Time``, and
  ``BoxSize``.
* ``Data`` contains ``Boundary``, ``Density``, ``Velocity``, ``Temperature``,
  and ``Mol_weight``. The optional ``NeutralFraction`` dataset stores
  ``xHI = nHI / nH`` for hydrogen thermo-chemistry runs.
* Datasets with units store the unit string in a ``units`` attribute.

Use :func:`radhydropy.io.writehdf5` to write this layout and
:func:`radhydropy.io.readhdf5` to load it. The bundled examples include small
wrapper classes that create initial conditions before launching a run.

Plotting Output
---------------

After a run, load an output file and plot a fluid quantity with
:func:`radhydropy.analysis.rplot1d`:

.. code-block:: python

   from radhydropy.analysis import rplot1d
   import radhydropy.io as rio

   rio.readhdf5(sim.par, sim.mesh, sim.fluid, "Output_001.hdf5")
   rplot1d(sim, yquan="rho")
