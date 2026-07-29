# RadHydropy

RadHydropy is a Python package for idealized one-dimensional hydrodynamics
simulations. It is designed for Cartesian and spherical test problems, uses a
shared internal code-unit system for runtime calculations, and reads/writes
simulation state through HDF5 files.

The code currently provides:

- mesh setup with ghost cells for Cartesian and spherical coordinates;
- primitive and conserved fluid state handling;
- ideal-gas pressure, temperature, energy-density, and sound-speed helpers;
- a finite-volume hydrodynamics solver with GLF/Rusanov interface fluxes;
- boundary-condition handling for periodic, open, reflecting, spherical open,
  inflow, and outflow modes;
- an optional chemistry composition selector with hydrogen microphysics now
  organized under `radhydropy/chemistry_species/`, alongside implicit
  neutral-fraction evolution and source-term subcycling;
- optional one-dimensional long-characteristic radiative transfer coupled to
  photon number density;
- HDF5 input/output helpers; and
- plotting utilities for one-dimensional outputs.

The runtime expects a ``CodeUnits`` block in the run parameters. Physical
inputs are converted to that internal unit system at initialization, so the
solver and source terms can work in a consistent code-unit space while the
example YAML files still use readable physical units.

Full documentation: https://tkc004.github.io/RadHydropy/

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
example, the Sod shock tube setup loads a YAML configuration, builds the
initial-condition file, runs the coupled hydrodynamics update, writes
`Output_*.hdf5` files, and plots the result:

```bash
cd example/SodShock1D
python sodshock1d.py
```

Most examples follow the same pattern:

1. load `runparams` and `ICparams` from the example YAML file;
2. create or load an HDF5 initial-condition file from `ICparams`;
3. construct `Rsim` with the runtime parameters, including `CodeUnits`;
4. call `RunAll()`; and
5. inspect or plot the output files.

The bundled YAML files define the internal unit system with a block like:

```yaml
CodeUnits:
  name: galactic_unit_system
  InternalUnitSystem:
    UnitMass_in_cgs:     4.92e31
    UnitLength_in_cgs:   3.08567758e21
    UnitVelocity_in_cgs: 1.0e5
    UnitCurrent_in_cgs:  1.0
    UnitTemp_in_cgs:     1.0
```

That block is required for the current runtime path. The example loaders and
startup conversion step use it to convert mesh, fluid, gravity, and source-term
inputs once at initialization.

The runner also exposes lower-level stepping methods when an example or test
needs finer control:

- `Step(mode="hydro")` advances only the finite-volume hydrodynamics update.
- `Step(mode="sources")` advances thermo-chemistry and radiative-transfer
  sources without a hydrodynamic flux update.
- `Step(mode="hydro_sources")` performs the standard coupled update used by
  the bundled examples.
- `Evolve(final_time=...)` loops over `Step(...)` and returns counters for the
  number of hydro and source updates.

The older convenience methods `RunOneStep()` and `RunHydroStep()` remain
available and now delegate to the same shared stepping path.

## Minimal Simulation Runner

RadHydropy runs from a YAML example configuration plus an HDF5
initial-condition file. The high-level `Rsim` class reads the initial
condition, prepares mesh and fluid state, advances the solver, and writes HDF5
outputs.

```python
from pathlib import Path

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
import tools as et
import matplotlib.pyplot as plt

config = Path("example/SodShock1D/sodshock1d.yaml")
runparams, ICparams = load_example_parameters(config)

ric = et.Simwrap(ICparams)
rio.writehdf5(ric, runparams["ICfilename"])

sim = Rsim(runparams)
sim.RunAll()

rio.readhdf5(sim.par, sim.mesh, sim.fluid, "Output_001.hdf5")
rplot1d(sim, yquan="rho")
plt.show()
```

This is the same pattern used by the bundled example scripts: load the YAML
file, generate ``InitialCondition.hdf5`` from ``ICparams``, then launch the run
with ``Rsim``. The helper resolves relative ``ICfilename``, ``outdir``,
``outputtimefilename``, and ``savedir`` paths against the example directory.
The plotting step reloads the first output snapshot and renders the density
profile with the built-in plotting helper.

To use explicit output times instead of a fixed cadence, set
`outputtimefilename` to a txt file whose first non-empty line is the time unit
and whose remaining lines are the output times. For example, the bundled
example configs typically point to files such as ``output_times.txt``:

```text
yr
0.0
1.0e4
2.0e4
```


If you want manual control over the evolution loop, use the canonical stepping
API directly:

```python
step = sim.Step(mode="hydro_sources")
print(step["dt"], step["hydro_steps"], step["source_steps"])

counters = sim.Evolve(final_time=sim.par.timesim, mode="hydro_sources")
print(counters)
```

For fixed-density thermo-chemistry tests such as the static Stromgren sphere,
`Rsim.EvolveStaticThermochemistry(...)` evolves the thermo-chemistry and
radiative-transfer source terms without a hydrodynamic update.

## Project Layout

```text
radhydropy/
  analysis.py            plotting helpers
  eos.py                 equation-of-state helpers
  example_config.py      example YAML parameter loaders
  fluid.py               primitive and conserved fluid state helpers
  gravity.py             gravity helpers and potentials
  chemistry.py           composition selector for chemistry modules
  chemistry_species/      species microphysics helpers
  io.py                  HDF5 read/write helpers
  mesh.py                mesh and geometry setup
  params.py              default runtime parameters
  radiative_transfer.py  one-dimensional radiative-transfer routines
  rsim.py                high-level simulation runner
  solver.py              finite-volume update operations
  thermo_chemistry.py    thermo-chemistry source terms
  thermo_networks/       thermo-network implementations
  utils.py               numerical and thermodynamic utilities

example/                 runnable example problems
tests/                   unit tests
docs/                    Sphinx documentation
```

## Documentation

The rendered docs include the installation guide, quickstart, example gallery,
and API reference, plus standalone pages for the main simulation subsystems:

- [Installation guide](docs/installation.rst)
- [Quickstart](docs/quickstart.rst)
- [Initial-condition parameters](docs/icparams.rst)
- [Hydrodynamics solver](docs/hydrodynamics.rst)
- [Gravity](docs/gravity.rst)
- [Thermo-chemistry solver](docs/thermo_chemistry.rst)
- [Boundary conditions](docs/boundary_conditions.rst)
- [Radiative transfer](docs/radiative_transfer.rst)
- [Examples](docs/examples.rst)
- [API reference](docs/api/index.rst)

For HTML documentation builds, see the
[installation guide](docs/installation.rst).
