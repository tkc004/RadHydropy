# Coupled gas/dark-matter linear-growth check

This inexpensive diagnostic evolves the same LCDM correlation-function shape
as the virial-shock experiment, but reduces its target enclosed overdensity to
`0.001`. The default uses 64 Eulerian gas cells and 64 live dark-matter shells;
the gas uses a coupled dual-energy update because the flow is cold and
supersonic. It has a spherical origin, negligible positive gas pressure, and
no central thermalization.

Run it from the repository root with:

```bash
python example/CosmologicalVirialShock1D/cosmological_gas_dm_linear_growth.py
```

The default uses cumulative shell mass interpolated linearly in enclosed
volume when evaluating the force on gas cells. This is the continuum control.
Run the current production step-function shell force with:

```bash
python example/CosmologicalVirialShock1D/cosmological_gas_dm_linear_growth.py \
  --raw-shell-force
```

The raw control writes to `outputs_linear_growth_raw_shell_force` so the two
results do not overwrite one another.

For a matched resolution-convergence run, use for example:

```bash
python example/CosmologicalVirialShock1D/cosmological_gas_dm_linear_growth.py \
  --resolution 256
```

This writes to `outputs_linear_growth_256` (and appends
`_raw_shell_force` for the raw control). In cold supersonic cells, total
energy is dominated by kinetic energy, so `E - K` loses precision. The dual
energy variable advects thermal energy independently and supplies the pressure
when that subtraction is unreliable, while total energy remains the
conservative quantity.

The calculation stops at cosmic time `0.05` by default, while the perturbation
is linear. It aborts if a shell crossing is predicted within the next hydro
step, if shell identities exchange, or if shell radii cease to be strictly
ordered.

The diagnostic solver records the smallest paired-face positivity factor on
every hydro step. These factors are local: a restrictive face no longer
multiplies the fluxes on unrelated faces, so gravity cannot accelerate gas
while freezing the entire Euler mass update. A zero local factor means that
one face was suppressed; the fitted gas density and velocity amplitudes are
the meaningful acceptance checks.

It uses a uniform origin-centred mesh for this calibration. That keeps the
geometric truncation error of a very large logarithmic innermost cell from
overwhelming the deliberately small linear perturbation.

`outputs_linear_growth/CosmologicalGasDMLinearGrowth.npz` stores the full
history of enclosed gas, dark-matter, and analytic growing-mode overdensities,
as well as gas and shell peculiar velocities and their analytic references.
The accompanying figure and text report summarize least-squares growing-mode
amplitudes over 20--200 comoving kpc. Shells are placed at volume-centred
radii with cell-integrated masses, which prevents the homogeneous half-shell
quadrature error from overwhelming the deliberately small perturbation.
