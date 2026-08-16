H/He multifrequency static Strömgren sphere
============================================

This example is based on ``MultiFrequencyRadiativeTransferSph1D`` and evolves
gas with hydrogen mass fraction ``X=0.75`` and helium mass fraction
``Y=0.25`` for ``100 Myr``. The hydrogen number density is
``10^-3 cm^-3`` and the source emits ``5*10^48`` ionizing photons per second
with a ``10^5 K`` blackbody spectrum.

It uses the five-group spectrum file
``tools/radiation_spectrum_generator/radiation_spectrum_BB100000K_5groups_HHe.h5``
and the ``hydrogen_helium`` thermo-chemistry network. The initial temperature
is ``100 K`` with neutral H and He.

Run from this directory with::

   python multifrequency_radiative_transfer_sph1d_hhe_100myr.py

The output is written to ``Output_000.hdf5`` and
``MultiFrequencyRadiativeTransferSph1D_HHe_100Myr.jpg``. To compare all five ion
fractions with the supplied reference profiles, run::

   python compare_snapshot_with_references.py

This writes ``HHe_multifrequency_snapshot_vs_reference.jpg``. The snapshot
contains 128 radial cells; the reference profiles and snapshot are both at
100 Myr.

The causal H/He C²-Ray option is configured separately in
``multifrequency_radiative_transfer_sph1d_hhe_100myr_c2ray.yaml``. Run it with::

   python multifrequency_radiative_transfer_sph1d_hhe_100myr.py \
     --config multifrequency_radiative_transfer_sph1d_hhe_100myr_c2ray.yaml

It writes ``Output_C2Ray_000.hdf5`` and
``MultiFrequencyRadiativeTransferSph1D_HHe_100Myr_C2Ray.jpg``. Compare its
snapshot with the same references using::

   python compare_snapshot_with_references.py \
     --snapshot Output_C2Ray_000.hdf5 \
     --figure HHe_multifrequency_c2ray_snapshot_vs_reference.jpg

The C²-Ray local solver uses all five spectral groups; the fourth and fifth
groups provide the photons capable of ionizing He II and producing He III.
