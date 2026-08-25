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
