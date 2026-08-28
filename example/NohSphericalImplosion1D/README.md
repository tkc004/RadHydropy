# Noh spherical implosion

This is a gravity-free spherical hydrodynamics benchmark for the spherical
geometric pressure-work update and dual-energy scheme. Uniform cold gas enters
from the outer `InflowSph` boundary with inward velocity and reflects at the
origin. The converging flow forms a central shock and converts kinetic energy
into thermal energy.

Run it with:

```bash
python noh_spherical_implosion1d.py \
  --config noh_spherical_implosion1d.yaml
```

To run the same resolutions without the independent dual-energy variable:

```bash
python noh_spherical_implosion1d.py \
  --config noh_spherical_implosion1d.yaml \
  --without-dual-energy
```

This comparison writes to `outputs_noh_spherical_implosion1d_no_dual_energy/`.

The YAML runs 128, 256, and 512 cells. The runner checks that each resolution
heats and forms a central shock, then writes:

- `NohSphericalImplosion1D_Profiles.jpg`, final density, temperature, velocity,
  and pressure profiles;
- `NohSphericalImplosion1D_Convergence.jpg`, density-profile convergence toward
  the finest resolution; and
- `NohSphericalImplosion1D_Convergence.npz`, numerical convergence data.

For the idealized cold Noh solution with :math:`\gamma=5/3` and unit inward
velocity, the strong-shock radius is approximately :math:`r_s=t/3`; the small
initial temperature in this test makes that analytic limit a useful reference.
The outer boundary must be inflow, not reflecting: a reflecting outer wall
would launch a second inward shock and change the benchmark.

The no-dual-energy comparison also forms the central shock at all three
resolutions for this setup. Since the initial Mach number is only about nine,
it is a spherical-geometry and pressure-work comparison rather than a severe
dual-energy cancellation stress test; use the HighMachAdvection1D examples
for that purpose.
