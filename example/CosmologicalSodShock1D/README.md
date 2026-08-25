# Cosmological Sod shock tube

This test evolves a standard Sod discontinuity in supercomoving variables in
an Einstein--de Sitter background. With no gravity or source terms, the
supercomoving shock is the ordinary Sod problem, while the physical fields
scale with the known expansion factors.

Run with:

```bash
python cosmological_sod_shock1d.py --config cosmological_sod_shock1d.yaml
```

The test uses a periodic box, so it contains the primary Sod front and its
periodic image. It checks mass and supercomoving total-energy conservation and
verifies shock heating. HLLC and dual energy can be selected independently:

```bash
python cosmological_sod_shock1d.py --riemann-solver HLLC --dual-energy
```
