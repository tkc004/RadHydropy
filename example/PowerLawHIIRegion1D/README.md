# Analytic power-law H II region

This standalone example reproduces the thin ionization-front construction for
the spherical cloud model of Franco, Tenorio-Tagle & Bodenheimer (1990),
*ApJ*, 349, 126.

The cloud has a constant-density core and an envelope

\[
n_H(r)=n_c \quad (r<r_c),\qquad
n_H(r)=n_c(r/r_c)^{-w} \quad (r\ge r_c).
\]

It uses `n_c = 10^6 cm^-3`, `r_c = 2.1e16 cm`, and a constant source rate
`Q = 5e49 s^-1`, matching the parameters used in the paper's spherical
examples. The formation phase is computed from photon conservation. For
`w <= 3/2`, the matched D-type expansion approximation is continued after the
front slows to approximately `2 c_i`. For `3/2 < w < 3`, the paper's
approximately linear champagne expansion is used. The `w=3` fit and the
`w>3` strong-shock approximation use the paper's fitted exponents, including
`delta = 2.8 + 0.55*(w-3)`. The champagne equations are evaluated in their
absolute time coordinate; the plot lower limit of `R = 1e17 cm` hides only
the smaller-radius portion.

Run:

```bash
python power_law_hii_region_analytic.py
```

The output is `PowerLawHIIRegion1D.jpg`. This script intentionally does not
import or call RadHydropy functions; it is an analytic reference for a future
hydrodynamic example.

## RadHydropy comparison

The `w=1` runtime comparison uses the spherical long-characteristic solver,
hydrogen photoionization, hydrodynamics, and the piecewise isothermal closure
used by `HIIRegionExpansion1D`. Neutral cells are held at 100 K and ionized
cells at `1e4 K`; thermal coupling is disabled so the prescribed ionized
temperature is maintained:

```bash
python power_law_hii_region_radhydropy.py \
  --config power_law_hii_region_radhydropy.yaml
```

It writes the IC and numbered output snapshots, then reloads those snapshots
and creates `PowerLawHIIRegion1D_RadHydroVsAnalytic.jpg`. The `w=1` runtime
comparison covers 100--39,811 yr using 256 cells; the standalone analytic
script can still be plotted through 100,000 yr. It also creates
`PowerLawHIIRegion1D_Profiles.jpg`, showing density and radial-velocity
profiles. Each snapshot time uses the same line style in both panels.

The same runtime can be run independently for `w=1.5` with:

```bash
python power_law_hii_region_radhydropy.py \\
  --config power_law_hii_region_w1p5.yaml
```

This uses separate `w1p5` IC/snapshot files and writes
`PowerLawHIIRegion1D_w1p5_RadHydroVsAnalytic.jpg` and
`PowerLawHIIRegion1D_w1p5_Profiles.jpg`. Its dedicated output schedule ends
at 19,953 yr (`log10(t/yr)=4.3`) and uses 1024 cells. Its isothermal EOS
threshold is `xHII > 0.5`: those cells are set to `1e4 K`, while the rest
remain at 100 K.

The same setup is also available for `w=1.4`:

```bash
python power_law_hii_region_radhydropy.py \\
  --config power_law_hii_region_w1p4.yaml
```
