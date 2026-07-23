# RadHydropy

RadHydropy is a Python package for idealized one-dimensional hydrodynamics
simulations. It is designed for Cartesian and spherical test problems, uses
`unyt` quantities for physical units, and reads/writes simulation state through
HDF5 files.

The code currently provides:

- mesh setup with ghost cells for Cartesian and spherical coordinates;
- primitive and conserved fluid state handling;
- ideal-gas pressure, temperature, energy-density, and sound-speed helpers;
- optional subcycled hydrogen cooling with implicit neutral-fraction evolution;
- optional one-dimensional long-characteristic radiative transfer coupled to
  photon number density;
- finite-volume updates with GLF/Rusanov interface fluxes;
- periodic, open, reflecting, spherical open, inflow, and outflow boundaries;
- HDF5 input/output helpers; and
- plotting utilities for one-dimensional outputs.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd RadHydropy
python -m pip install -e .
```

The core package depends on `numpy`, `h5py`, and `unyt`. These are installed
automatically from `pyproject.toml`.

For development and documentation work, install the optional extras:

```bash
python -m pip install -e ".[test,docs]"
```

Run the test suite with:

```bash
pytest
```

## Quick Start

The fastest way to try the code is to run one of the bundled examples. For
example, the Sod shock tube setup creates an initial-condition file, runs the
simulation, writes `Output_*.hdf5` files, and plots the result:

```bash
cd example/SodShock1D
python sodshock1d.py
```

Most examples follow the same pattern:

1. define run parameters;
2. create or load an HDF5 initial-condition file;
3. construct `Rsim`;
4. call `RunAll()`; and
5. inspect or plot the output files.

The runner also exposes lower-level stepping methods when an example or test
needs finer control:

- `Step(mode="hydro")` advances only the finite-volume hydrodynamics update.
- `Step(mode="sources")` advances thermo-chemistry and radiative-transfer
  sources without a hydro flux update.
- `Step(mode="hydro_sources")` performs the standard coupled update.
- `Evolve(final_time=...)` loops over `Step(...)` and returns counters for the
  number of hydro and source updates.

The older convenience methods `RunOneStep()`, `RunHydroStep()`,
`RunCoupledHydroSourceStep()`, and `EvolveCoupledHydroSources()` remain
available and now delegate to the same shared stepping path.

## Minimal Simulation Runner

RadHydropy simulations are driven by a parameter dictionary and an HDF5
initial-condition file. A typical runner looks like this:

```python
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
    "CFL": 0.1,
    "boundcond": "Periodic",
    "order": 1,
    "dtmin": 2.0e-8 * unyt.s,
    "dtmax": 2.0e-1 * unyt.s,
}

sim = Rsim(runparams)
sim.RunAll(outputtime=1)
```

The `ICfilename` file must already exist. You can create it with
`radhydropy.io.writehdf5`, as shown in the scripts under `example/`.

If you want manual control over the evolution loop, use the canonical stepping
API directly:

```python
step = sim.Step(mode="hydro_sources")
print(step["dt"], step["hydro_steps"], step["source_steps"])

counters = sim.Evolve(final_time=sim.par.timesim, mode="hydro_sources")
print(counters)
```

For fixed-density thermo-chemistry tests such as the static Stromgren sphere,
`Rsim.EvolveStaticThermochemistry(...)` evolves the source terms without a
hydrodynamic update.

## Initial-Condition File Format

Initial-condition and output files use a compact HDF5 structure:

- `Header`
  - `Coordinate_System`
  - `Number_Grids`
  - `Time`
  - `BoxSize`
- `Data`
  - `Boundary`
  - `Density`
  - `Velocity`
  - `Temperature`
  - `Mol_weight`
  - `NeutralFraction` (optional; used by hydrogen thermo-chemistry)

Datasets with physical units store the unit string in a `units` attribute.

## Project Layout

```text
radhydropy/
  analysis.py   plotting helpers
  eos.py        equation-of-state setup
  fluid.py      primitive fluid state helpers
  io.py         HDF5 read/write helpers
  mesh.py       mesh and geometry setup
  params.py     default runtime parameters
  radiative_transfer.py optional long-characteristic photon transport
  rsim.py       high-level simulation runner
  solver.py     finite-volume update operations
  utils.py      numerical and thermodynamic utilities

example/        runnable example problems
tests/          unit tests
docs/           Sphinx documentation
```

## Documentation

Build the Sphinx documentation from the project root:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` after the build finishes. You can also run
`make html` from inside the `docs/` directory.
