Structure and Runtime Flow
==========================

RadHydropy is organized around a typed initial-condition boundary, a
representation-aware runtime state, and a single run coordinator:

.. code-block:: text

   nested YAML
       |
       v
   InitialConditionWriter
       |
       v
   validated HDF5 initial condition
       |
       v
   Rsim → Mesh + Fluid + Solver + source modules
       |
       v
   HDF5 snapshots and diagnostics

The example workflow is the recommended way to connect these components. See
:doc:`quickstart` for a short runnable version.

Physics modules
---------------

The runtime combines the following physics and numerical components:

* **Mesh and geometry** — one-dimensional Cartesian and spherical finite-volume
  meshes with active cells and ghost cells.
* **Fluid and EOS** — primitive and conserved gas fields, ideal-gas pressure,
  temperature, energy, and sound speed.
* **Hydrodynamics** — finite-volume Euler updates, Riemann fluxes,
  reconstruction, positivity protection, and optional dual-energy evolution.
* **Boundary conditions** — periodic, open, reflecting, spherical inflow, and
  spherical outflow ghost-cell updates.
* **Gravity** — external fields, gas self-gravity, pressure-supported cores,
  and cosmological gravity.
* **Dark matter** — live spherical dark-matter shells, shell crossing, and
  gas/dark-matter gravity coupling.
* **Thermo-chemistry** — hydrogen, H/He, CIE, PIE, Compton, and source-solver
  options for heating, cooling, and ionization.
* **Radiative transfer** — one-dimensional long-characteristic transport,
  instantaneous updates, and causal C²-Ray updates.
* **Radiation pressure** — momentum and energy deposition from absorbed photons.
* **Cosmology** — Einstein--de Sitter and flat matter--Lambda backgrounds with
  physical or supercomoving representations.

Detailed equations and parameter descriptions are maintained on the subsystem
pages listed in :ref:`next-steps`.

Configuration hierarchy
-----------------------

Each maintained example has one complete nested YAML configuration:

.. code-block:: text

   config
   ├── par
   │   ├── simulation
   │   ├── mesh
   │   ├── hydrodynamics
   │   ├── boundary
   │   ├── timestep
   │   ├── output
   │   ├── units
   │   ├── diagnostics
   │   ├── gravity
   │   ├── cosmology
   │   ├── dark_matter
   │   ├── thermochemistry
   │   ├── chemistry
   │   └── radiation
   ├── initial_condition
   └── example

``par`` controls runtime behavior. ``initial_condition`` contains physical
inputs used to build the initial state. ``example`` contains plotting,
comparison, and other workflow controls. Physical values use explicit
``{value, unit}`` mappings, and ``par.units.CodeUnits`` is required.

Initial-condition flow
----------------------

The initial-condition stage converts unit-bearing physical inputs into typed
runtime fields:

.. code-block:: text

   load_nested_example_config(config_filename)
       |
       v
   CodeUnits.from_mapping(par.units.CodeUnits)
       |
       v
   build_initial_condition(config)
       |
       v
   InitialConditionWriter.radarray / radquantity
       |
       v
   writer.write(filename, validate=True)
       |
       v
   InitialCondition.hdf5

The writer keeps physical values unit-bearing while constructing geometry and
primitive fields, derives the typed runtime state, and validates active-cell
geometry, finite values, positivity, and EOS consistency before writing.

Runtime startup order
---------------------

After the initial-condition file is written, the runner starts the simulation
through the following high-level sequence:

1. Load the complete nested configuration and construct ``Rsim(config["par"])``.
2. Load the HDF5 initial condition through ``loadhdf5(config, filename)``.
3. Restore ``CodeUnits`` and field representation metadata from the HDF5
   header.
4. Build the typed mesh and ghost-cell geometry.
5. Initialize primitive, conserved, EOS, and optional dual-energy fields.
6. Initialize configured gravity, dark-matter, thermo-chemistry, and radiation
   state.
7. Enter the selected run mode and schedule outputs.

Cosmological startup also restores the cosmology context and its clocks. When
``par.cosmology.supercomoving_coordinates`` is enabled, the runtime assigns the
matching comoving coordinate and supercomoving field representations.

One coupled timestep
---------------------

The standard ``hydro_sources`` path combines the following operations at a
high level:

.. code-block:: text

   choose timestep
       ↓
   refresh boundary ghost cells
       ↓
   finite-volume hydrodynamic flux update
       ↓
   gravity and other configured source updates
       ↓
   radiative-transfer update, when enabled
       ↓
   thermo-chemistry heating, cooling, and ionization
       ↓
   radiation-pressure source, when enabled
       ↓
   synchronize primitive/conserved state, clocks, and diagnostics
       ↓
   write scheduled snapshot

The exact source path depends on the selected network and transport scheme.
For example, C²-Ray uses causal source-to-cell ordering, while source-only
benchmarks use ``mode="sources"`` without a hydrodynamic flux update.

Run modes and APIs
------------------

The main execution APIs are:

``Rsim.RunAll()``
   Run the configured workflow and scheduled outputs.
``Rsim.Run(mode="sources")``
   Run source updates without a hydrodynamic flux update, as used by the
   Uniform EdS thermochemistry benchmark.
``Rsim.Step(mode=...)``
   Perform one controlled ``hydro``, ``sources``, or ``hydro_sources`` step.
``Rsim.Evolve(final_time=..., mode=...)``
   Advance through repeated controlled steps until a target time.
``Rsim.EvolveStaticThermochemistry(...)``
   Evolve fixed-density thermo-chemistry and radiative-transfer state without
   hydrodynamic fluxes.

Representations and units
-------------------------

The runtime distinguishes physical and cosmological representations. Examples
and diagnostics should preserve that distinction in field names:

``*_radarray``
   Typed, unit-bearing dimensional views for mesh and fluid data.
``*_proper_code``
   Numerical proper-coordinate solver fields.
``*_comoving_code``
   Numerical comoving-coordinate fields.
``*_supercomoving_code``
   Numerical supercomoving fields, such as peculiar velocity or pressure.

Use the typed RadArray views for plotting and physical diagnostics. Convert to
plain values only at an explicit numerical boundary using ``to_value`` or
``quantity_to_value``.

Snapshots and diagnostics
-------------------------

The output relationship is:

.. code-block:: text

   typed runtime state
       |
       v
   HDF5 snapshot with CodeUnits and representation metadata
       |
       v
   loadhdf5(config, filename)
       |
       v
   typed RadArray fields → analysis and plots

See :doc:`snapshots` for the HDF5 layout and provenance fields. Example
diagnostics should use physical active cells rather than ghost cells and retain
explicit representation and unit names.

.. _next-steps:

Next steps
----------

* :doc:`quickstart` — run an example and inspect a snapshot.
* :doc:`parameters` — configure runtime groups and common options.
* :doc:`initial_conditions` — construct and validate HDF5 initial conditions.
* :doc:`snapshots` — read output files and provenance metadata.
* :doc:`hydrodynamics` — finite-volume equations, fluxes, and reconstruction.
* :doc:`thermo_chemistry` — thermal and chemical source networks.
* :doc:`radiative_transfer` — ray tracing, spectra, and C²-Ray.
* :doc:`radiation_pressure` — absorbed-photon momentum sources.
* :doc:`gravity` and :doc:`dark_matter` — gravitational components.
* :doc:`cosmology` — supercomoving variables and cosmological backgrounds.
* :doc:`examples` — runnable workflows and detailed example pages.
