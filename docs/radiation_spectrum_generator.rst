Radiation Spectrum Generator
============================

The utility
``tools/radiation_spectrum_generator/generate_radiation_spectrum.py``
generates an HDF5 radiation-spectrum file for RadHydropy's multibin
radiative-transfer solver. It calculates blackbody-weighted group quantities
using Verner (1996) photoionization cross-section fits.

Generating the standard pure-H spectrum
----------------------------------------

From the repository root, run::

   python tools/radiation_spectrum_generator/generate_radiation_spectrum.py

The default output is::

   tools/radiation_spectrum_generator/radiation_spectrum_BB100000K_3groups_HI.h5

The default spectrum uses a ``10^5 K`` blackbody, group edges
``[13.6, 24.6, 54.4, 10000] eV``, and an ionizing injection rate of
``5×10^48 photons/s``.

Custom generation
------------------

The main options are:

.. list-table::
   :header-rows: 1
   :widths: 28 52

   * - Option
     - Meaning
   * - ``--output FILE``
     - Output HDF5 filename.
   * - ``--temperature K``
     - Blackbody temperature.
   * - ``--edges E0 E1 ...``
     - Radiation-group boundaries in eV.
   * - ``--injected-photons-per-second RATE``
     - Total ionizing photon injection rate used to construct the stored
       spectrum normalization.
   * - ``--samples-per-group N``
     - Log-spaced integration samples per group.
   * - ``--verner-file FILE``
     - Verner cross-section fit file.
   * - ``--include-helium``
     - Include He I and He II cross-section and photoheating datasets.

For example::

   python tools/radiation_spectrum_generator/generate_radiation_spectrum.py \
     --output custom_BB80000K_5groups_HI.h5 \
     --temperature 80000 \
     --edges 13.6 20.0 30.0 54.4 100.0 10000.0 \
     --injected-photons-per-second 1.0e49

The command illustrates that arbitrary strictly increasing group edges are
supported.

HDF5 schema
-----------

The file contains a ``RadiationSpectrum`` group with the datasets:

* ``group_edges_eV``;
* ``ionizing_photon_energy_erg``;
* ``star_emission_rates``;
* ``group_sigma_gamma_cm2``; and
* ``group_epsilon_gamma_erg``.

The group attributes include ``number_of_radiation_groups``,
``number_of_group_edges``, ``stellar_spectrum_type``,
``stellar_spectrum_type_name``,
``stellar_spectrum_blackbody_temperature_K``, and ``absorber``.

The generated file can be consumed by the runtime using
``radiation_spectrum_filename``. To override the total rate without
regenerating the file, use ``radiation_spectrum_total_photon_rate`` in YAML;
RadHydropy rescales all ionizing groups by one common factor and preserves the
relative spectrum. See :doc:`radiative_transfer` and
:doc:`multifrequency_radiative_transfer_sph1d` for runtime examples.

For a five-group H/He file with custom edges::

   python tools/radiation_spectrum_generator/generate_radiation_spectrum.py \
     --output tools/radiation_spectrum_generator/radiation_spectrum_BB100000K_5groups_HHe.h5 \
     --edges 13.6 24.6 35.5 54.4 75.0 50000.0 \
     --include-helium

Hydrogen and helium cross-sections
-----------------------------------

The Verner fit file identifies species by atomic number ``Z`` and number of
electrons ``N``. The generator uses the following entries:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30

   * - Species
     - ``(Z, N)``
     - Threshold
   * - H I
     - ``(1, 1)``
     - 13.6 eV
   * - He I
     - ``(2, 2)``
     - 24.6 eV
   * - He II
     - ``(2, 1)``
     - 54.4 eV

For each species and radiation group, the generator samples the blackbody
spectrum on a logarithmic energy grid and evaluates the Verner (1996) fit.
The group cross-section is photon-weighted:

.. math::

   \langle\sigma\rangle_g =
   \frac{\int_g \sigma(E) N_\gamma(E)\,dE}
        {\int_g N_\gamma(E)\,dE}.

The corresponding excess heating energy is weighted by the absorption
probability:

.. math::

   \langle\epsilon\rangle_g =
   \frac{\int_g \sigma(E)N_\gamma(E)
         [E-E_{\rm th}]\,dE}
        {\int_g \sigma(E)N_\gamma(E)\,dE}.

Groups entirely below a species' ionization threshold receive zero
cross-section and zero excess-energy values for that species.

The current runtime network is pure hydrogen, so the standard output writes
the H I datasets:

``group_sigma_gamma_cm2`` and ``group_epsilon_gamma_erg``.

For a future hydrogen-plus-helium network, the HDF5 schema should use explicit
species names, for example:

.. code-block:: text

   group_sigma_gamma_HI_cm2
   group_epsilon_gamma_HI_erg
   group_sigma_gamma_HeI_cm2
   group_epsilon_gamma_HeI_erg
   group_sigma_gamma_HeII_cm2
   group_epsilon_gamma_HeII_erg

This keeps the cross-section and heating data unambiguous when multiple
absorbers are coupled to the same radiation groups.
