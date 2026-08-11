H/He multifrequency static Strömgren sphere: 30,000 K
======================================================

This example is based on ``MultiFrequencyRadiativeTransferSph1D_HHe_100Myr``
but uses a ``30,000 K`` blackbody spectrum. It evolves gas with
``X=0.75``, ``Y=0.25``, ``nH=10^-3 cm^-3``, and a source rate of
``5*10^48`` ionizing photons per second for ``100 Myr``.

It uses the five-group spectrum file
``tools/radiation_spectrum_generator/radiation_spectrum_BB30000K_5groups_HHe.h5``.

Run from this directory with::

   python multifrequency_radiative_transfer_sph1d_hhe_30000k_100myr.py
