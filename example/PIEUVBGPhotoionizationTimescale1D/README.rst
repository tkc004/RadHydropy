PIEUVBGPhotoionizationTimescale1D
=================================

This test evolves uniform, optically thin one-cell gas parcels with the HM12
``pie_uvbg_cooling`` network at redshift ``z=4``. It covers initial
temperatures ``1e3``, ``1e4``, ``2e4``, and ``1e5 K`` and densities ``0.1``,
``1``, and ``10 cm^-3``. Each case is followed for ten prescribed
photoionization timescales.

The PIE table assumes ionization equilibrium and does not evolve a neutral
fraction. Therefore the photoionization timescale is an explicit diagnostic
timescale, not a rate calculated by this network. The script finds the
zero-net-heating temperature from the HM12 table and reports the temperature
error after one and ten photoionization timescales.

Run with::

   python pie_uvbg_photoionization_timescale_1d.py

The case snapshots are written below ``outputs/nH_*_T_*/`` and the combined
temperature/error plots are written separately to
``outputs/PIEUVBGPhotoionizationTimescale1D_nH_*.jpg``. Their horizontal axes
show physical time in years on a logarithmic scale.
