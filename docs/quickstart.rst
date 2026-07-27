Quickstart
==========

RadHydropy runs from an HDF5 initial-condition file plus a run-parameter
dictionary. The high-level :class:`radhydropy.rsim.Rsim` class reads the initial
condition, prepares mesh and fluid state, advances the solver, and writes HDF5
outputs.

Minimal Runner
--------------

.. code-block:: python

   import unyt
   from radhydropy.rsim import Rsim

   runparams = {
       "simname": "SodShock1d",
       "ICfilename": "InitialCondition.hdf5",
       "outdir": ".",
       "outfileprefix": "Output",
       "coordsys": "cartesian",
       "EOStype": "polytropic",
       "gamma": 1.4,
       "timesim": 1.0 * unyt.s,
       "outdeltatime": 0.1 * unyt.s,
       "outputtimefilename": None,
       "CFL": 0.1,
       "boundcond": "Periodic",
       "order": 1,
       "dtmin": 2.0e-8 * unyt.s,
       "dtmax": 2.0e-1 * unyt.s,
   }

   sim = Rsim(runparams)
   sim.RunAll(outputtime=1)

To use explicit output times instead of a fixed cadence, set
`outputtimefilename` to a txt file whose first non-empty line is the time unit
and whose remaining lines are the output times. For example:

.. code-block:: text

   yr
   0.0
   1.0e4
   2.0e4

Stepping API
------------

The high-level runner now exposes a canonical stepping interface through
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
* :meth:`radhydropy.rsim.Rsim.RunCoupledHydroSourceStep`
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
