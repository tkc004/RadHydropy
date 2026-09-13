# RadHydropy

RadHydropy is a one-dimensional finite-volume hydrodynamics package for
Cartesian and spherical test problems. It supports ideal-gas fluid evolution,
gravity, thermo-chemistry, radiative transfer, cosmological coordinates, and
HDF5-based initial conditions and snapshots.

The main user entry point is the example workflow:

```text
nested YAML → InitialConditionWriter → HDF5 initial condition
            → Rsim → HDF5 snapshots → loadhdf5 and typed RadArray views
```

## Installation

Install the package in editable mode:

```bash
git clone <repository-url>
cd RadHydropy
python -m pip install -e .
```

For tests and documentation, install the optional dependencies:

```bash
python -m pip install -e ".[test,docs]"
```

## Run your first example

Run examples from their own directories so relative configuration paths and
helper imports resolve correctly:

```bash
cd example/SodShock1D
python sodshock1d.py
```

The script builds an HDF5 initial condition, runs the solver, and writes
`Output_*.hdf5` files and a figure in the example directory.

Examples with variants accept an explicit configuration:

```bash
cd example/StellarWindBubble1D
python stellar_wind_bubble1d.py \
  --config stellar_wind_bubble1d_no_metal.yaml
```

Other useful starting points are:

```bash
cd example/UniformEdSThermochemistry1D
python uniform_eds_thermochemistry1d.py

cd ../StaticStromgrenSphere1D
python static_stromgren_sphere1d.py
```

## Minimal Python workflow

The following is the standard pattern used by the bundled examples. Run it
from `example/SodShock1D`:

```python
from pathlib import Path

import radhydropy.io as rio
from example_utils import load_nested_example_config
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import tools as example_tools

config = Path("sodshock1d.yaml")
config_data = load_nested_example_config(config)
config_data["_code_units"] = CodeUnits.from_mapping(
    config_data["par"]["units"]["CodeUnits"]
)

writer = example_tools.build_initial_condition(config_data)
writer.write(
    config_data["par"]["simulation"]["initial_condition_filename"],
    validate=True,
)

sim = Rsim(config_data["par"])
sim.RunAll()

snapshot = rio.loadhdf5(config_data, "Output_001.hdf5")
radius = snapshot.mesh.boundary_radarray
density = snapshot.fluid.rho_radarray
```

`InitialConditionWriter` constructs and validates the initial state. `Rsim`
evolves it. `loadhdf5` reloads an initial condition or snapshot and exposes
typed `*_radarray` views for dimensional mesh and fluid data.

## Configuration model

Every maintained example uses one complete nested YAML configuration:

| Section | Purpose |
| --- | --- |
| `par` | Runtime, mesh, solver, output, and code-unit settings |
| `initial_condition` | Physical inputs used to build the initial state |
| `example` | Plotting, comparison, and workflow-specific settings |

The `par.units.CodeUnits` block is required. Physical YAML values use explicit
`{value, unit}` mappings, for example:

```yaml
par:
  simulation:
    final_time: {value: 1.0, unit: s}
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
  temperature_proper: {value: 1.0e4, unit: K}
```

Use explicit representations in configuration and diagnostics, such as
`rho_proper`, `temperature_proper`, `rho_proper_code`, and
`rho_comoving_code`. Convert unit-bearing Python values with
`quantity_to_value` or `.to_value()`.

Maintained spherical examples use `OutflowSph`. The
`StellarWindBubble1D` no-metal configuration is validated with hydrodynamics
`order: 0`.

## Development and validation

Run the full test suite with:

```bash
python -m pytest
```

For example or configuration changes, also run:

```bash
python -m pytest -q tests/test_example_alignment.py
python -m sphinx -b html docs /tmp/radhydropy-docs
git diff --check
```

## Project layout

```text
RadHydropy/
  radhydropy/             solver package and runtime APIs
    rsim/                 run orchestration and lifecycle
    solver/               finite-volume updates and source stepping
    thermo_networks/      hydrogen, H/He, CIE, PIE, Compton, and C²-Ray
    chemistry_species/    species microphysics
    initial_condition_writer.py
                          validated initial-condition construction
    radarray.py           representation-aware dimensional fields
    units.py              code-unit definitions and conversions
    cosmology.py          cosmological models and scale factors
    params.py             nested runtime parameters and defaults
    example_config.py     nested YAML configuration loading
    io.py                 HDF5 initial-condition and snapshot I/O
  example/                runnable examples and configurations
    example_utils.py      shared example loader and helpers
    <ExampleName>/        scripts, YAML, plots, and diagnostics
  docs/                   Sphinx documentation and example pages
  tests/                  unit and regression tests
    test_example_alignment.py
                          nested example/configuration audit
  tools/                  spectrum-generation and supporting tools
  .github/workflows/      CI and documentation deployment
  .codex/skills/          repository-local RadHydropy guidance
  pyproject.toml          package metadata and dependencies
  README.md               project overview and usage guide
```

## Documentation

- [Installation guide](docs/installation.rst)
- [Quickstart](docs/quickstart.rst)
- [Architecture and runtime flow](docs/architecture.rst)
- [Initial conditions](docs/initial_conditions.rst)
- [Snapshots](docs/snapshots.rst)
- [Hydrodynamics](docs/hydrodynamics.rst)
- [Gravity](docs/gravity.rst)
- [Thermo-chemistry](docs/thermo_chemistry.rst)
- [Radiative transfer](docs/radiative_transfer.rst)
- [Examples](docs/examples.rst)
- [API reference](docs/api/index.rst)

The rendered documentation is available at
<https://tkc004.github.io/RadHydropy/>.
