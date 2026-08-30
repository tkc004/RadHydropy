# CosmologicalVirialShock1D

This example is a controlled EdS spherical-collapse experiment patterned on the
cosmological part of Birnboim & Dekel (2003). It evolves comoving Eulerian gas
with live collisionless dark-matter shells, then repeats the same initial
conditions with either adiabatic gas or time-dependent radiative cooling. The
radiative case uses CIE cooling while `z > 10`, then switches to the HM12
UV-background PIE table at `z <= 10`.

Before the UV background turns on, the initial gas is cold CIE-era gas; after
the turn-on epoch, the initial temperature is obtained by solving the HM12 PIE
table for heating = cooling. The gas and dark matter start with the
growing-mode inward peculiar velocity. The disc radius is a
diagnostic centrifugal radius, estimated from `r = j^2/(G M_target)`; it is not
a resolved rotating disc in this 1-D spherical model.

Run with:

```bash
python cosmological_virial_shock1d.py
```

For a fast end-to-end check, use
`cosmological_virial_shock1d_smoke.yaml`.

To calibrate the halo mass without gas or shell-crossing noise, run:

```bash
python cosmological_dark_matter_only.py
```

This evolves the target (10^{12}M_\odot) Lagrangian top-hat boundary directly,
and reports numerical versus analytic turnaround and virial scales. The
resulting calibration figure is `outputs/CosmologicalTopHatDarkMatterOnly.jpg`.

To generate the correlation-function initial condition from the (z=100)
configuration without evolving the system, run:

```bash
python generate_cosmological_correlation_ic.py
```

This writes `outputs_correlation/InitialCondition.hdf5` and a diagnostic plot
of the correlation-shaped density contrast and quiet-Hubble/peculiar velocity
components. The table radius is in Mpc/(h), while simulation coordinates are
converted from the code length unit before interpolation.

To re-read and verify the stored HDF5 fields, run:

```bash
python plot_cosmological_correlation_ic.py
```

The production configuration uses 1024 live dark-matter shells. This is
important for the wide logarithmic radial domain: with 64 shells the target
mass falls in the outermost perturbed shell and shell-crossing errors delay
turnaround substantially. The live-shell integrator resolves crossings with
substeps while still advancing the complete requested cosmological timestep.

The dark-matter-only run also writes density snapshots at selected cosmic
times to `outputs/CosmologicalDarkMatterOnlyDensityProfiles.jpg`. The shell
masses are accumulated into common logarithmic proper-radius bins before
computing the density; raw binned profiles are saved in
`CosmologicalDarkMatterOnlyDensityProfiles.npz`.

To run the separate 1024-shell adiabatic gas experiment from the same z=100
correlation-function IC, run:

```bash
python cosmological_adiabatic_gas_correlation.py
```

It uses 128 Eulerian gas cells and assigns the gas `f_b rho_m` and the live
dark matter `(1-f_b) rho_m`, so the homogeneous matter density is not counted
twice. It saves the evolving physical gas-density profiles and the live-DM
`r_200` diagnostic in
`outputs_correlation_gas_adiabatic/AdiabaticGasDensityProfiles.npz`, together
with `AdiabaticGasDensityProfiles.jpg`. The plot uses comoving radius on its
x-axis and marks each profile's corresponding proper virial radius converted
to comoving coordinates. The adiabatic control starts with
`T = 2.7255(1+z) K` and a configurable residual post-recombination electron
fraction (`2e-4` by default), then evolves Compton CMB heating/cooling at the
instantaneous redshift. Atomic recombination, collisional ionization, UV
heating, and radiative cooling are disabled in this control.

The reusable linear-theory tools in `RadHydropy/tools/` can generate a
sigma8-normalized LCDM power spectrum and its tabulated correlation function
when called from the repository root:

```python
from tools.lcdm_correlation import generate_lcdm_correlation_table
generate_lcdm_correlation_table(
    "outputs_correlation/lcdm_linear_correlation.h5"
)
```

`linear_correlation_from_power_spectrum` performs the exact Fourier-Bessel
integral for any supplied tabulated linear `P(k)`. The built-in spectrum uses
the analytic Eisenstein--Hu no-wiggle transfer shape; CAMB/CLASS output can be
passed to the same integral when BAO-accurate correlation structure is needed.

For comparison with the Bertschinger/Fillmore--Goldreich similarity solution,
the enclosed initial overdensity in this correlation-function IC is locally
well represented by `delta M / M propto M^-s` with `s ~= 0.2` around the target
halo scale. For `s < 2/3`, the nonlinear collapsed dark-matter profile has the
similarity slope

```text
rho_DM propto r^(-9 s / (1 + 3 s)) ~= r^-1.125.
```

The saved profiles in
`outputs_correlation/CosmologicalDarkMatterOnlyDensityProfiles.npz` give an
outer-halo fit of approximately `rho_DM propto r^-1.2` over
`1.2 r_200 < r < 3 r_200`, broadly consistent with this prediction. The
linear far-field overdensity instead scales approximately as `r^(-3 s)`, or
`r^-0.6`; that is not the nonlinear halo slope.

For the correlation-gas run, the gas mesh has a finite reflecting inner wall
at `inner_wall_radius_comoving: 5.0` kpc. This is about 0.05 kpc proper at the
z=100 start and avoids the singular zero-area origin. The energy audit records
the wall momentum and energy fluxes.

The requested figure is `outputs/CosmologicalVirialShock1D.jpg`, with
adiabatic results on top and radiative PIE results below. Each panel plots the
total mass interior to the virial, detected shock, and disc radii. Raw histories
are saved in `outputs/{adiabatic,radiative}/mass_radius_history.npz`.
The production gas-correlation configurations use a volume-smoothed live
dark-matter force (`smooth_dm_force_for_gas: true`). Shell trajectories remain
raw and conserve shell mass; only the enclosed mass sampled by gas cells is
interpolated linearly in `r^3`, removing 64/1024-shell force jumps without
altering the collisionless evolution.
