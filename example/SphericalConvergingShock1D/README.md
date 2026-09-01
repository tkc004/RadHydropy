# Spherical converging-flow shock benchmark

This is a gravity-free spherical Euler test. A uniform gas initially moves
inward, reflects at the origin and outer wall, and focuses into a central
converging shock.

The benchmark has no simple closed-form post-shock profile because the shock
trajectory is spherical and time-dependent. Its solver-independent
expectations are:

- the central density and temperature rise after focusing;
- the total gas mass remains constant under reflecting boundaries;
- total kinetic plus thermal energy remains constant to time-integration and
  spatial-discretization accuracy.

Run it from this directory with:

```bash
python spherical_converging_shock1d.py --config spherical_converging_shock1d.yaml
```

Use `--riemann-solver HLLC` to run the same case with HLLC; omit it for the
Rusanov baseline.

Add `--dual-energy` to evolve the thermal-energy density separately and use it
for pressure recovery when kinetic energy dominates. It can be combined with
HLLC:

```bash
python spherical_converging_shock1d.py \
  --config spherical_converging_shock1d.yaml \
  --riemann-solver HLLC \
  --dual-energy
```

The script writes `SphericalConvergingShock1D.jpg` and checks all three
expectations. The initial `order: 1` setting is the current MUSCL spatial
reconstruction and is intentionally kept fixed when comparing Riemann
solvers or dual-energy variants.

## Cold spherical-flow settings

The reference YAML enables dual energy because spherical convergence makes
the thermal energy a small residual of total energy minus kinetic energy.
The configured controls have distinct jobs:

- `dual_energy_eta1: 1e-3` selects `InternalEnergy` when the conservative
  thermal fraction is too small for reliable pressure recovery.
- `dual_energy_eta2: 1e-1` controls when the independently evolved thermal
  state may be synchronized back to conservative `Energy`.
- `dual_energy_consistency_factor: 1e-1` rejects a dual-energy update that
  drops by more than a factor of ten when conservative `E-K` cannot recover
  it, preventing artificial temperature dips.
- `dual_energy_pressure_floor: 1e-20` is used only when both thermal-energy
  estimates are invalid; its injected energy is tracked by the solver.
- `dual_energy_entropy_limiter: false` disables the experimental entropy
  limiter for this reference configuration. Positivity limiting and the
  dual-energy consistency recovery remain enabled.

These settings preserve the conservative total-energy field while using the
separately evolved thermal state to avoid zero-temperature artifacts caused
by spherical E-K cancellation. `riemann_solver: Rusanov` is retained as the
robust reference flux for this cold, strongly converging flow.
