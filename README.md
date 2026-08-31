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

The runtime requires a ``CodeUnits`` block in the run parameters, and the same
unit system must be written into the HDF5 header. This is compulsory: RadHydropy
does not fall back to cgs for current example workflows. Physical inputs are
converted to that internal unit system at initialization, so the solver and
source terms can work in a consistent code-unit space while the example YAML
files still use readable physical units. Example-side helper functions can
still present ``unyt``-friendly interfaces, but they should convert to code
units or plain floats internally on hot paths.

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

Most migrated examples follow the same pattern:

1. load the nested `par`, `initial_condition`, and `example` sections from
   the example YAML file;
2. convert YAML `{value, unit}` entries to `unyt` quantities with
   `example_utils.load_nested_example_config()`;
3. create an HDF5 initial-condition file from `initial_condition`;
4. construct `Rsim` with nested runtime parameters;
5. call `RunAll()`; and
6. inspect or plot the output files.

The bundled YAML files define the internal unit system under
`par.units.CodeUnits`:

```yaml
par:
  units:
    CodeUnits:
      name: galactic_unit_system
      InternalUnitSystem:
        UnitMass_in_cgs: 4.92e31
        UnitLength_in_cgs: 3.08567758e21
        UnitVelocity_in_cgs: 1.0e5
        UnitCurrent_in_cgs: 1.0
        UnitTemp_in_cgs: 1.0
```

That block is required for the current runtime path and must also be stored in
the HDF5 header. The example loaders and startup conversion step use it to
convert mesh, fluid, gravity, and source-term inputs once at initialization.
Example helpers such as gravity profiles and hydrostatic reference solutions
should accept ``unyt`` quantities at the script boundary but evaluate in code
units or floats internally.

The runner also exposes lower-level stepping methods when an example or test
needs finer control:

- `Step(mode="hydro")` advances only the finite-volume hydrodynamics update.
- `Step(mode="sources")` advances thermo-chemistry and radiative-transfer
  sources without a hydrodynamic flux update.
- `Step(mode="hydro_sources")` performs the standard coupled update used by
  the bundled examples.
- `Evolve(final_time=...)` loops over `Step(...)` and returns counters for the
  number of hydro and source updates.

## Minimal Simulation Runner

RadHydropy runs from a YAML example configuration plus an HDF5
initial-condition file. The high-level `Rsim` class reads the initial
condition, prepares mesh and fluid state, advances the solver, and writes HDF5
outputs.

```python
from pathlib import Path

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
from example_utils import load_nested_example_config
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import tools as et
import matplotlib.pyplot as plt

config = Path("example/SodShock1D/sodshock1d.yaml")
config_data = load_nested_example_config(config)
par_config = config_data["par"]
icparams = config_data["initial_condition"]
code_units = CodeUnits.from_mapping(par_config["units"]["CodeUnits"])

ric = et.Simwrap(icparams, code_units=code_units)
rio.writehdf5(ric, par_config["simulation"]["initial_condition_filename"])

sim = Rsim(par_config)
sim.RunAll()

rio.readhdf5(sim.par, sim.mesh, sim.fluid, "Output_001.hdf5")
rplot1d(sim, yquan="rho")
plt.show()
```

This is the same pattern used by the migrated example scripts: load the YAML
file, generate ``InitialCondition.hdf5`` from ``initial_condition`` using the
unit system under ``par.units.CodeUnits``, then launch the run with ``Rsim``.
The helper converts ``{value, unit}`` mappings to ``unyt`` quantities and
resolves paths against the example directory.
The plotting step reloads the first output snapshot and renders the density
profile with the built-in plotting helper.

To use explicit output times instead of a fixed cadence, set
`outputtimefilename` to a txt file whose first non-empty line is the time unit
and whose remaining lines are the output times. Include the final simulation
time if you want the last state written as an output snapshot. For example,
the bundled example configs typically point to files such as ``output_times.txt``:

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
  rsim/                  high-level simulation runner and execution components
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
- [Initial-condition and example parameters](docs/icparams.rst)
- [Hydrodynamics solver](docs/hydrodynamics.rst)
- [Gravity](docs/gravity.rst)
- [Thermo-chemistry solver](docs/thermo_chemistry.rst)
- [Boundary conditions](docs/boundary_conditions.rst)
- [Radiative transfer](docs/radiative_transfer.rst)
- [Examples](docs/examples.rst)
- [API reference](docs/api/index.rst)

For HTML documentation builds, see the
[installation guide](docs/installation.rst).
