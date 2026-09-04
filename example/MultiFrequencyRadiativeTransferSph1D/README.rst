Multifrequency Radiative Transfer Spherical Example
====================================================

This example evolves a static, uniform pure-hydrogen medium around a central
source using spherical long-characteristic transport and five ionizing photon
groups:

.. code-block:: text

   [13.6, 24.6, 35.5, 54.4, 75.0, 50000] eV

Run
---

From this directory::

   python multifrequency_radiative_transfer_sph1d.py \
     --config multifrequency_radiative_transfer_sph1d.yaml

The YAML points to the generated spectrum file in
``tools/radiation_spectrum_generator``. Regenerate it from the repository root
with::

   python tools/radiation_spectrum_generator/generate_radiation_spectrum.py

Spectrum HDF5 file
------------------

RadHydropy reads ``radiation_spectrum_filename`` during startup. The HDF5
group ``RadiationSpectrum`` contains:

* ``group_edges_eV``;
* ``ionizing_photon_energy_cgs_erg``;
* ``star_emission_rates``;
* ``group_sigma_gamma_cgs_cm2``;
* ``group_epsilon_gamma_cgs_erg``.

Its attributes include ``number_of_radiation_groups``,
``number_of_group_edges``, spectrum type, blackbody temperature, and absorber.

The optional YAML parameter below rescales all ionizing groups while preserving
their relative spectrum::

   radiation_spectrum_total_photon_rate:
     value: 5.0e48
     unit: 1/s

If omitted, the normalization stored in the HDF5 file is used.

Physics and outputs
-------------------

The example uses temperature-dependent hydrogen recombination and
collisional-ionization rates, photoheating, and cooling. It writes
``InitialCondition.hdf5``, ``Output_000.hdf5``, and
``MultiFrequencyRadiativeTransferSph1D.jpg``. The plot contains neutral
fraction, temperature, and photon number density for every group.

The reference files use radius in units of ``r_s = 5.4 kpc`` and store
``log10(T/K)`` or ``log10(x_HI)`` in their second column.
