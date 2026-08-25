# Spherical shell reflecting-wall shock

This short validation problem places a cold, supersonic shell in an otherwise
vacuum spherical domain and sends it inward onto a finite wall at `rmin=1 kpc`.
The run uses `riemann_solver: HLLC` and `order: 0`.  Only the inner ghost cells
are reflected (`rho` and `p` copied, `u` negated); the outer ghost cells are
outflow.  The finite wall face therefore has nonzero area and its Riemann flux
is retained.

Run from this directory with:

```bash
python spherical_shell_reflecting_wall1d.py --config spherical_shell_reflecting_wall1d.yaml
```

The script writes `outputs/SphericalShellReflectingWall1D.jpg` and an NPZ
containing density, velocity, pressure, temperature, entropy, and the inner
wall mass/momentum/energy flux at every step.

To repeat the test with the more diffusive Rusanov solver, use
`python spherical_shell_reflecting_wall1d.py --riemann-solver Rusanov`; its
diagnostics are written under `outputs_Rusanov/`.
