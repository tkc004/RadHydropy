# CosmologicalVirialShock1D `Rsim.RunAll()` Migration Plan

> **Status: archival — migration complete.**
>
> This document records the design and acceptance criteria for the completed
> migration. It is retained for historical context and is not an active work
> plan or the authoritative API reference. For current behavior, see
> `docs/architecture.rst`, `docs/snapshots.rst`, and `docs/validation.rst`.

## Purpose

Migrate `CosmologicalVirialShock1D` from its example-specific evolution loop to
the standard RadHydropy workflow:

```python
writer.write(config["par"]["simulation"]["initial_condition_filename"])
mainrun = Rsim(config["par"])
mainrun.RunAll(...)
```

The standard runner should own initial-condition loading, runtime setup,
timestep scheduling, HDF5 output, and simulation completion. The example
should provide only its cosmological physics hooks and diagnostic analysis.

The plan was originally design-only. The migration has since been completed;
the remaining sections describe the former implementation, the intended
callback contracts, and the historical acceptance criteria.

## Current situation

The main workflow is currently implemented in:

```text
example/CosmologicalVirialShock1D/cosmological_gas_correlation_z100.py
```

Its `run()` function currently performs all of the following:

1. Loads and normalizes the nested YAML configuration.
2. Builds the correlation-function initial condition.
3. Builds the live dark-matter shell object.
4. Writes an initial HDF5 file.
5. Calls `rio.loadhdf5()` manually.
6. Reconstructs mesh and fluid runtime state manually.
7. Restores and verifies cosmological clocks manually.
8. Configures gravity and the live dark-matter object manually.
9. Updates cosmological boundary conditions before every step.
10. Updates thermochemistry before every step.
11. Calls `GetStepTime()` and `sim.Step()` in a custom loop.
12. Decides independently when diagnostics should be saved.
13. Writes custom HDF5 snapshot files.
14. Reloads each snapshot with `loadhdf5()`.
15. Reads dark-matter RadArrays from the reloaded snapshot.
16. Builds gas, dark-matter, virial, energy, and angular-momentum histories.
17. Writes several `.npz` files and plots after the loop.

The implementation already has the correct typed snapshot interface:

```python
snapshot.dark_matter.radius_radarray
snapshot.dark_matter.radial_velocity_radarray
snapshot.dark_matter.dark_matter_mass_radarray
snapshot.dark_matter.softening_radquantity
```

The migration must preserve this analysis boundary.

## Existing core capabilities

The relevant existing paths are:

```text
radhydropy/rsim/core.py
radhydropy/rsim/evolution.py
radhydropy/rsim/output.py
radhydropy/output.py
radhydropy/io.py
```

Important existing behavior:

- `Rsim.RunAll()` reads the configured initial-condition file, initializes
  mesh/fluid state, and calls `Rsim.Run()`.
- `Rsim.Run()` already uses `sim.Evolve()`.
- `sim.Evolve()` already supports `history_callback` and `output_callback`.
- The HDF5 output path already serializes runtime state through the direct
  HDF5 serializer.
- HDF5 output already serializes a `DarkMatter` group when
  `sim.par.dark_matter` is present.
- `loadhdf5()` already reconstructs the mutable dark-matter solver object and
  the typed `dark_matter_radarrays` analysis view.

The main missing capability is exposing suitable lifecycle callbacks through
`Run()` and `RunAll()`, and invoking snapshot diagnostics after serialization.

## Target architecture

The example should eventually have this shape:

```python
def run(config_filename=DEFAULT_CONFIG):
    config = load_nested_example_config(config_filename)

    writer = build_initial_condition(config)
    writer.write(config["par"]["simulation"]["initial_condition_filename"])

    mainrun = Rsim(config["par"])
    diagnostics = CosmologicalVirialShockDiagnostics(config)

    mainrun.RunAll(
        before_step_callback=diagnostics.before_step,
        history_callback=diagnostics.on_step,
        snapshot_callback=diagnostics.on_snapshot,
    )

    diagnostics.finalize(mainrun)
```

The exact names may be adjusted during implementation, but the responsibilities
must remain separate:

| Hook | Timing | Responsibility |
| --- | --- | --- |
| `before_step_callback` | Before timestep estimation | Update boundary/source state required by the next timestep estimate |
| `history_callback` | After each accepted step | Collect per-step energy and conservation diagnostics |
| `snapshot_callback` | After HDF5 serialization | Reload persisted state and collect RadArray-based profiles |
| `finalize_callback` or explicit `diagnostics.finalize()` | After the run | Write histories, plots, validation summaries, and reports |

## Proposed `Rsim` API changes

### `Rsim.Run()` and `Rsim.RunAll()`

Extend the existing methods with callback parameters rather than creating a
second example-specific runner:

```python
mainrun.RunAll(
    outputtime=0,
    mode="hydro_sources",
    advect_chemistry=True,
    stop_condition=None,
    step_backend=None,
    step_backend_kwargs=None,
    before_step_callback=None,
    history_callback=None,
    snapshot_callback=None,
)
```

The implementation should avoid adding an unrelated configuration wrapper or
an override object. These are execution hooks, not alternate parameter
sources.

### Callback forwarding

Forward the callbacks through the existing call chain:

```text
Rsim.RunAll()
  -> rsim.evolution.RunAll()
  -> Rsim.Run()
  -> rsim.evolution.Run()
  -> sim.Evolve()
```

`history_callback` can be forwarded directly to `sim.Evolve()`.

`before_step_callback` requires a deliberate insertion point. It must run
before `sim.GetStepTime()` if boundary or source updates affect the timestep
estimate. A callback invoked only inside `step_backend` would be too late.

`snapshot_callback` belongs to the output layer because it must receive the
actual filename written by the output writer.

## Snapshot callback contract

The preferred callback signature is:

```python
snapshot_callback(sim, snapshot_filename, output_index)
```

The callback must run after the HDF5 serializer has completed. It may then do:

```python
snapshot = rio.loadhdf5(config, snapshot_filename)
dark_matter = snapshot.dark_matter
```

The callback must use the restored typed fields for dimensional analysis:

```python
radius_radarray = dark_matter.radius_radarray
velocity_radarray = dark_matter.radial_velocity_radarray
mass_radarray = dark_matter.dark_matter_mass_radarray
softening_radquantity = dark_matter.softening_radquantity
```

The mutable object at `snapshot.par.dark_matter` remains reserved for restart
or solver behavior. It must not become the example-facing analysis source.

### Initial output

The initial HDF5 output must also trigger `snapshot_callback`. This ensures the
diagnostic history contains the same initial state as later output times.

### Output filename ownership

The output writer should return or expose the filename it writes. The callback
must not reconstruct the filename from `directory`, `filename_prefix`, and an
index because custom output writers or future naming changes would make that
fragile.

## Initial-condition migration

The current workflow writes the IC before attaching the live dark-matter shell
object. This must be changed so the initial HDF5 file contains the `DarkMatter`
group before `Rsim.RunAll()` reads it.

Target order:

```python
writer = build_initial_condition(config)
writer.simulation.par.dark_matter = make_dark_matter(config)
writer.write(initial_condition_filename)

mainrun = Rsim(config["par"])
```

The initial HDF5 read should then reconstruct:

```python
mainrun.par.dark_matter
mainrun.par.dark_matter_radarrays
```

The initial-condition builder must continue to use the current nested
configuration and unit-bearing IC boundary. Do not reintroduce manual copies
of cosmological flags, clocks, or representation conversions.

## Cosmological boundary and source hooks

The custom loop currently performs operations before each timestep that are
not diagnostic-only. These must be preserved through a pre-step hook.

The hook should perform, in the existing order:

1. Read the current supercomoving time.
2. Convert to cosmic time and scale factor through the configured cosmology.
3. Update time-dependent thermochemistry parameters.
4. Update the outer cosmological boundary state.
5. Preserve the outer background reservoir.
6. Apply the solver boundary condition.
7. Rebuild the conserved state if the current workflow requires it.
8. Return control to the runner before `GetStepTime()`.

This ordering must not be changed casually. The current example uses the
boundary state to influence timestep estimation.

The hook should not write files or build plots. Those belong to snapshot and
finalization diagnostics.

## Output cadence migration

The current example has an example-local profile cadence. The migrated path
should use the standard output cadence so HDF5 snapshots and diagnostics always
refer to the same times.

Preferred YAML structure:

```yaml
par:
  output:
    directory: outputs_correlation_gas_tvir1e3_z15_gas1024
    filename_prefix: CosmologicalGasCorrelationTvir1e3Z15
    cadence: {value: 0.1, unit: kpc/(km/s)}
```

The example should not independently decide when to call a second
`save_snapshot()` function. If different cadences are later required, add an
explicit scheduler feature rather than restoring a private evolution loop.

## Diagnostic accumulator design

Move the current `save_snapshot()` logic into a dedicated example diagnostic
component, preferably in:

```text
example/CosmologicalVirialShock1D/diagnostics.py
```

Possible structure:

```python
class CosmologicalVirialShockDiagnostics:
    def __init__(self, config):
        self.config = config
        self.gas_profiles = []
        self.dark_matter_profiles = []
        self.radius_history = []
        self.energy_history = []
        self.snapshot_files = []

    def before_step(self, sim): ...

    def on_step(self, sim): ...

    def on_snapshot(self, sim, snapshot_filename, output_index): ...

    def finalize(self, sim): ...
```

The component may be split into smaller helpers if the existing file becomes
too large. Keep plotting functions separate from the callback that extracts
state.

### Snapshot extraction

`on_snapshot()` should:

1. Load the written HDF5 file with `rio.loadhdf5()`.
2. Read gas data from restored `*_radarray` views where dimensional analysis
   is required.
3. Read dark-matter shell data from `snapshot.dark_matter` RadArrays.
4. Convert representations explicitly with `to_proper()` or
   `to_comoving()`.
5. Read the softening length from `softening_radquantity`.
6. Build the softened shell density profile using the same effective radius as
   the shell force law:

   ```python
   radius_effective_comoving = radius_comoving + softening_comoving
   ```

7. Append only representation-qualified fields to the diagnostic history.

### Per-step extraction

Energy closure, boundary flux, gravitational work, thermochemistry changes,
and angular-momentum conservation are per-step diagnostics. They should be
collected by `on_step()` rather than recomputed from HDF5 snapshots unless the
diagnostic specifically requires persisted output.

### Finalization

`finalize()` should contain the existing post-loop work:

- Convert lists to arrays.
- Pad variable-length shell histories.
- Write the main `.npz` history.
- Write dark-matter density `.npz` data.
- Write energy-audit `.npz` data.
- Write per-cell/per-shell energy history.
- Generate all figures.
- Validate dark-matter mass conservation.
- Validate energy and angular-momentum residuals.
- Print output paths and summary values.

## Softened dark-matter density requirements

The dark-matter density plot must remain based on persisted snapshot data and
must use the snapshot softening length.

Required source fields:

```python
dark_matter.radius_radarray
dark_matter.dark_matter_mass_radarray
dark_matter.softening_radquantity
```

The density-bin center must use the effective softened coordinate consistent
with the solver:

```python
radius_effective = radius + softening
```

The raw shell coordinate may also be retained in the diagnostic output, but
the plotted density profile must use the softened coordinate.

The saved dark-matter diagnostic data should include an explicit field such as:

```text
softening_comoving_code
```

Do not silently use `sim.par.dark_matter.softening` when the callback is
analyzing a reloaded snapshot. The persisted snapshot’s typed softening field
is the authoritative analysis input.

## Files expected to change during implementation

### Core execution files

```text
radhydropy/rsim/core.py
radhydropy/rsim/evolution.py
radhydropy/rsim/output.py
radhydropy/io.py
```

Only the callback plumbing and output-callback ordering should change in core.
Do not alter solver operator ordering or cosmological conversion rules.

### Example files

```text
example/CosmologicalVirialShock1D/cosmological_gas_correlation_z100.py
example/CosmologicalVirialShock1D/virial_shock_tools.py
example/CosmologicalVirialShock1D/diagnostics.py
```

The existing large script may initially retain plotting helpers while its
execution loop is removed. Extract a new diagnostics module only if that keeps
the migration readable.

### Configuration and documentation

```text
example/CosmologicalVirialShock1D/*.yaml
example/CosmologicalVirialShock1D/README.md
docs/snapshots.rst
docs/api/io.rst
```

Update YAML only where necessary to use the standard output cadence and to
ensure dark matter is part of the serialized IC. Document that `RunAll()` now
writes snapshots and that diagnostics consume those snapshots through
`loadhdf5()`.

## Implementation sequence

### Step 1: add callback plumbing without changing the example

- Extend `Rsim.Run()` and `Rsim.RunAll()` signatures.
- Forward `history_callback` through the runner.
- Add `before_step_callback` at the correct pre-timestep location.
- Add `snapshot_callback` to the HDF5 output path.
- Ensure the callback receives the actual filename and output index.
- Keep existing behavior unchanged when callbacks are `None`.

### Step 2: add core callback tests

Test:

- callback order;
- initial snapshot callback;
- output snapshot callback;
- callback filename existence;
- no callback behavior regression;
- cosmological clock consistency during callback execution.

### Step 3: serialize dark matter in the IC

- Attach the initial shell object before `writer.write()`.
- Verify that the initial HDF5 file contains `DarkMatter`.
- Verify `Rsim.RunAll()` reconstructs the mutable shell state.
- Verify the typed dark-matter RadArray view is present after load.

### Step 4: migrate boundary and source preparation

- Extract the current pre-step updates into a callback or dedicated physics
  component.
- Confirm that it executes before timestep estimation.
- Remove duplicated startup-clock restoration from the example once
  `RunAll()` owns startup.

### Step 5: migrate snapshot diagnostics

- Move `save_snapshot()` logic into the snapshot callback.
- Reload each written HDF5 file.
- Use typed gas and dark-matter analysis views.
- Preserve softened dark-matter density behavior.
- Preserve all current history keys and units.

### Step 6: migrate finalization and plots

- Move NPZ writing and plot generation into the diagnostic finalizer.
- Keep plotting functions unchanged where possible.
- Make all output paths derive from `par.output` and the active configuration.

### Step 7: remove the custom evolution loop

- Replace the example-specific `while` loop with `mainrun.RunAll()`.
- Remove manual calls to `GetStepTime()` and `Step()`.
- Remove manual snapshot scheduling.
- Remove duplicated HDF5 writing and startup logic.

### Step 8: validate numerical equivalence

Run the old and migrated workflows with the same configuration and compare:

- final cosmic time;
- number of accepted hydro steps;
- dark-matter substep count;
- total dark-matter mass;
- softened density profiles;
- virial and shock radii;
- energy closure residual;
- angular-momentum residual;
- generated snapshot count.

Only remove the old path after these comparisons pass within documented
tolerances.

## Testing plan

Run focused tests after each stage:

```bash
python -m pytest -q tests/test_io.py tests/test_dark_matter.py
python -m pytest -q tests/test_example_alignment.py
```

Run the example with a short final time first, then the production YAML:

```bash
python example/CosmologicalVirialShock1D/cosmological_gas_correlation_z100.py \
  --config example/CosmologicalVirialShock1D/cosmological_virial_shock1d_smoke.yaml

python example/CosmologicalVirialShock1D/cosmological_gas_correlation_z100.py \
  --config example/CosmologicalVirialShock1D/cosmological_gas_correlation_tvir1e3_z15.yaml
```

Inspect one generated snapshot with:

```python
snapshot = rio.loadhdf5(config, filename)
dark_matter = snapshot.dark_matter

print(dark_matter.radius_radarray)
print(dark_matter.dark_matter_mass_radarray)
print(dark_matter.softening_radquantity)
```

The final test set should include the full relevant core suite and the example
workflow tests.

## Acceptance criteria

The migration is complete when all of the following hold:

- The example uses `InitialConditionWriter`, `Rsim(config["par"])`, and
  `RunAll()`.
- The example has no private evolution `while` loop.
- HDF5 output scheduling is owned by the standard runner.
- Snapshot diagnostics run after HDF5 serialization.
- Dark-matter diagnostics use restored typed RadArrays.
- The dark-matter density plot uses the persisted softening length and the
  softened effective radius.
- Cosmological boundary updates still occur before timestep estimation.
- Per-step energy and angular-momentum audits remain available.
- Existing plots and `.npz` outputs remain available with explicit field
  names.
- The production `tvir1e3_z15` example completes successfully.
- Focused and full relevant tests pass.
- No legacy serialization wrapper or example-specific compatibility adapter is
  introduced.
