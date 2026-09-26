# High-Mach advection tests

The runner supports a uniform periodic state, a pressure-equilibrated density
contact, and a vacuum expansion. The vacuum case can be run without the
independent dual-energy variable:

```bash
python high_mach_advection1d.py \
  --config high_mach_vacuum_expansion1d.yaml \
  --without-dual-energy
```

This comparison is written to
`outputs_high_mach_vacuum_expansion1d_no_dual_energy/`. The tested run
completed successfully with relative total-energy change about `1.7e-16`,
with no fallback, synchronization, or pressure-floor events.

To keep dual-energy evolution enabled but force pressure selection to use only
the conservative `E-K` estimate, run:

```bash
python high_mach_advection1d.py \
  --config high_mach_vacuum_expansion1d.yaml \
  --conservative-pressure
```

This sets `dual_energy_pressure_selection: conservative` and writes to
`outputs_high_mach_vacuum_expansion1d_conservative_pressure/`. In the vacuum
test, the run reaches an extremely dilute cell where cancellation in `E-K`
produces an enormous sound speed and collapses the timestep. This failure is
the expected demonstration of why the independent thermal-energy pressure
selection is needed near vacuum.
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/HighMachAdvection1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
