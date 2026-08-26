# Cosmological converging-wall shock

This controlled companion to `CosmologicalVirialShock1D` places a cold,
supersonic shell in an otherwise vacuum spherical domain and sends it inward
onto a finite wall at `rmin=1 kpc`. It uses the same physical profile and
shock diagnostics needed to interpret the cosmological correlation run.
The run uses `riemann_solver: HLLC` and `order: 0`.  Only the inner ghost cells
are reflected (`rho` and `p` copied, `u` negated); the outer ghost cells are
outflow.  The finite wall face therefore has nonzero area and its Riemann flux
is retained.

Run from this directory with:

```bash
python cosmological_converging_shock1d.py --config cosmological_converging_shock1d.yaml
```

The script writes `outputs_cosmological_converging/CosmologicalConvergingShock1D.jpg` and an NPZ
containing density, velocity, pressure, temperature, entropy, and the inner
wall mass/momentum/energy flux at every step.

To repeat the test with the more diffusive Rusanov solver, use
`python cosmological_converging_shock1d.py --riemann-solver Rusanov`; its
diagnostics are written under `outputs_cosmological_converging_Rusanov/`.
