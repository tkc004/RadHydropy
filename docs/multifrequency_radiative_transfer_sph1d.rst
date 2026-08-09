Multifrequency Radiative Transfer Spherical Example
====================================================

The example is located at
``example/MultiFrequencyRadiativeTransferSph1D``. It evolves a static,
uniform pure-hydrogen medium with spherical long-characteristic transport,
hydrogen thermo-chemistry, photoheating, and cooling.

Running the example
-------------------

From the example directory::

   python multifrequency_radiative_transfer_sph1d.py \
     --config multifrequency_radiative_transfer_sph1d.yaml

The default setup uses three groups with edges
``[13.6, 24.6, 54.4, 10000] eV`` and a ``10^5 K`` blackbody spectrum. The
simulation has ``n_H = 10^-3 cm^-3``, 512 spherical cells, and evolves to
100 Myr.

HDF5 spectrum input
-------------------

The generation utility and its command-line options are documented in
:doc:`radiation_spectrum_generator`.

The YAML parameter ``radiation_spectrum_filename`` points to an HDF5 file.
RadHydropy loads this file during :class:`radhydropy.params.Par` startup and
reloads it after initial-condition HDF5 headers are read, preserving the
physical units of the spectrum data.

The file contains a ``RadiationSpectrum`` group with:

* ``group_edges_eV``;
* ``ionizing_photon_energy_erg``;
* ``star_emission_rates``;
* ``group_sigma_gamma_cm2``;
* ``group_epsilon_gamma_erg``.

The group attributes include the number of radiation groups and group edges,
spectrum type, blackbody temperature, and absorber. A matching file can be
generated with::

   python tools/radiation_spectrum_generator/generate_radiation_spectrum.py

Total source normalization
--------------------------

The optional parameter ``radiation_spectrum_total_photon_rate`` overrides the
total ionizing photon injection rate while preserving the relative group
spectrum::

   radiation_spectrum_total_photon_rate:
     value: 5.0e48
     unit: 1/s

When omitted, the HDF5 file's normalization is retained. The first
``star_emission_rates`` entry is a non-ionizing placeholder; the remaining
entries are the ionizing groups.

Hydrogen network
----------------

The example sets ``hydrogen_alpha_B`` and ``hydrogen_beta`` to ``null`` so the
built-in temperature-dependent rates are used. Collisional ionization and
thermal coupling are enabled. The current network is pure hydrogen; helium
datasets in a spectrum file are ignored.

Outputs and references
----------------------

The run writes ``InitialCondition.hdf5``, ``Output_000.hdf5``, and
``MultiFrequencyRadiativeTransferSph1D.jpg``. The plot shows the neutral
fraction, temperature, and photon number density of each group.

``TTT1D_Stromgren100Myr.txt`` and ``xTT1D_Stromgren100Myr.txt`` contain the
multifrequency comparison profiles. Their radius is normalized by
``r_s = 5.4 kpc``; their second columns contain ``log10(T/K)`` and
``log10(x_HI)``, respectively.
