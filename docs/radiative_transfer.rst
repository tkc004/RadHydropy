Long-Characteristic Radiative Transfer
======================================

RadHydropy includes an optional one-dimensional long-characteristic ray tracer
for hydrogen ionizing photons. The module is disabled by default and is enabled
with ``radiative_transfer=True``. When enabled, it computes a finite-volume
cell-averaged photon number density and writes it to ``fluid.ngamma`` before
the hydrogen source terms use ``n_gamma`` for photo-ionization and
photo-heating.

The implementation uses the shared internal code-unit helpers in
``radhydropy.units``. Runtime inputs such as boundary fluxes, source photon
rates, and photon densities are converted once into the internal unit system at
startup, then the ray tracer works in that consistent code-unit space.

Governing Idea
--------------

The ray tracer follows photons along a monotonic one-dimensional
characteristic. In a cell with constant hydrogen opacity,

.. math::

   \kappa_i = \sigma_\gamma n_{{\rm H},i} x_i ,

the optical depth across the cell is

.. math::

   \Delta \tau_i = \kappa_i \Delta s_i .

The outgoing photon flux or photon rate is then

.. math::

   F_{i+1/2} = F_{i-1/2}\exp(-\Delta\tau_i)

for Cartesian transport, or

.. math::

   Q_{i+1/2} = Q_{i-1/2}\exp(-\Delta\tau_i)

for spherical transport. Here ``F`` is photon number flux in
``photons cm^-2 s^-1`` and ``Q`` is photon number rate in ``photons s^-1``.

Cartesian Finite Volumes
------------------------

For Cartesian coordinates the module treats the ray as a plane-parallel beam
with constant cell face area. The finite-volume, cell-averaged flux is the
analytic cell average,

.. math::

   \langle F\rangle_i =
   F_{\rm in,i}\frac{1-\exp(-\Delta\tau_i)}{\Delta\tau_i},

with the small-optical-depth limit set to ``F_in``. The photon density coupled
to the chemistry is

.. math::

   n_{\gamma,i} = \frac{\langle F\rangle_i}{c}.

The absorbed photon rate per volume is also reported:

.. math::

   \dot n_{{\rm abs},i} =
   \frac{A_i F_{\rm in,i} - A_{i+1}F_{\rm out,i}}{V_i}.

Spherical Finite Volumes
------------------------

For spherical coordinates the module transports the integrated photon rate
``Q`` so geometric dilution is handled by the face areas rather than by
artificially attenuating the luminosity. The cell-averaged photon density is

.. math::

   n_{\gamma,i} =
   \frac{Q_{\rm in,i}\Delta r_i}{c V_i}
   \frac{1-\exp(-\Delta\tau_i)}{\Delta\tau_i}.

This expression remains finite for the first spherical cell touching
``r = 0``. Face photon fluxes are available from ``Q / (4 pi r^2)`` on
nonzero-radius faces.

Runtime Coupling
----------------

The implementation lives in :mod:`radhydropy.radiative_transfer`. The runner
calls ``Solver.ApplyRadiativeTransfer`` after the hydrodynamic update and
before ``Solver.AddHydrogenSources``. Inside the hydrogen source subcycling,
the ray trace is repeated so the photon field responds to the current neutral
fraction. When ``radiative_transfer=True``, the local analytic
``hydrogen_radiation_evolution`` sink is ignored to avoid double attenuation:
the ray tracer supplies ``n_gamma`` instead.

The optional C²-Ray temporal scheme is selected with
``radiative_transfer_temporal_scheme: c2ray``. It processes cells in causal
source-to-boundary order. For each cell it iterates the time-averaged neutral
fraction, computes the conservative absorbed photon rate, relaxes the local
hydrogen chemistry over the source timestep, and only then passes the outgoing
photon rate to the next cell. The default ``instantaneous`` scheme is
unchanged. The current C²-Ray implementation supports the hydrogen network;
hydrogen-helium runs continue to use the instantaneous scheme.

C²-Ray Temporal Integration
----------------------------

C²-Ray is implemented as a separate source integrator in
:mod:`radhydropy.thermo_networks.c2ray`. The long-characteristic geometry and
finite-volume transport operations are shared with the ordinary ray tracer,
but the temporal update is different: the radiation field and chemistry are
advanced together during one global source timestep.

For a cell traversed by an incoming photon rate ``Q_in``, the integrator uses
the current iterate of the time-averaged neutral fraction ``xbar`` to form

.. math::

   \Delta\tau = \sigma_\gamma n_{\rm H} x_{\rm bar} \Delta s,
   \qquad
   Q_{\rm abs} = Q_{\rm in}(1 - e^{-\Delta\tau}),
   \qquad
   Q_{\rm out} = Q_{\rm in}e^{-\Delta\tau}.

The absorbed rate is conservative: in spherical geometry it is the incoming
minus outgoing photon rate divided by the cell volume. The local hydrogen
equation is then advanced over the complete source timestep using this
absorbed photon rate together with recombination and, when enabled, collisional
ionization. The resulting time-averaged neutral fraction is iterated until it
matches the opacity used for transport. Only after this local iteration has
finished is ``Q_out`` passed to the next cell.

This ordering is essential. It makes the method causal and photon-conserving,
but it also means that cells along one ray cannot be updated independently or
fully vectorized: the outgoing rate from cell ``i`` is the incoming rate for
cell ``i+1``. The independent work inside each cell is the small implicit
neutral-fraction iteration. For a one-dimensional spherical run, the method is
therefore naturally implemented as a fast compiled-style numerical loop over
cells rather than as an all-mesh vector operation.

The implementation supports scalar and multigroup source rates and reuses the
shared transport geometry. The current thermo-chemistry coupling is limited
to ``thermochemistry_network: hydrogen``. The ordinary instantaneous scheme
remains the default and is used for H/He runs.

The C²-Ray controls are:

.. code-block:: yaml

   radiative_transfer_temporal_scheme: c2ray
   radiative_transfer_c2ray_max_iterations: 32
   radiative_transfer_c2ray_tolerance: 1.0e-6
   radiative_transfer_c2ray_relaxation: 1.0
   radiative_transfer_c2ray_nonconvergence: raise

``radiative_transfer_c2ray_relaxation`` is between zero and one; one uses the
new iterate directly, while smaller values under-relax the opacity update.
The nonconvergence policy can be ``warn``, ``raise``, or a silent fallback
after the iteration limit. For production runs, ``raise`` is useful when an
unresolved local timestep should not pass unnoticed.

The complete benchmark workflow is documented in
:doc:`static_stromgren_c2ray_comparison`. It compares a 100-step C²-Ray run
with instantaneous runs using 100, 1,000, 10,000, and 100,000 steps. Its lower
panel uses the 100,000-step instantaneous trajectory as the numerical
reference, rather than the analytic Strömgren solution.

Useful parameters are:

.. list-table::
   :header-rows: 1
   :widths: 32 48 20

   * - Key
     - Meaning
     - Typical unit
   * - ``radiative_transfer``
     - Enable the optional long-characteristic update.
     - boolean
   * - ``radiative_transfer_method``
     - Currently only ``long_characteristics``.
     - string
   * - ``radiative_transfer_boundary_flux``
     - Incident photon number flux used for Cartesian beams or spherical
       boundary illumination.
     - ``cm^-2 s^-1``
   * - ``radiative_transfer_source_photon_rate``
     - Photon number rate for a spherical point/source luminosity. Use this
       when the inner radial face is at ``r = 0``.
     - ``s^-1``
   * - ``radiative_transfer_direction``
     - ``+1`` traces from the left/inner boundary to the right/outer boundary;
       ``-1`` traces in the opposite direction.
     - dimensionless
   * - ``hydrogen_sigma_gamma``
     - Hydrogen photo-ionization cross-section used for opacity.
     - ``cm^2``

Multibin Transport
-------------------

The long-characteristic solver supports multiple photon groups. The number of
groups is ``len(radiation_group_edges_eV) - 1``. For example,

.. code-block:: yaml

   radiation_group_edges_eV: [13.6, 24.6, 54.4, 10000.0]
   radiation_group_sigma_gamma: [2.99e-18, 5.66e-19, 7.84e-20]
   radiation_group_epsilon_gamma: [6.17e-12, 2.81e-11, 7.77e-11]
   radiative_transfer_source_photon_rate_groups: [2.24e48, 2.48e48, 2.94e47]

creates three groups. Cross-sections and excess photoheating energies have one
entry per group. The source-rate and boundary-flux arrays also have one entry
per group. The resulting photon density has shape ``(number_of_groups,
number_of_cells)`` and each group is transported with its own optical depth.

The legacy scalar parameters remain valid. With no group edges, the solver uses
``hydrogen_sigma_gamma``, ``radiative_transfer_source_photon_rate``, and
``radiative_transfer_boundary_flux`` and returns the traditional one-dimensional
photon-density array. A single group is represented by two edges, for example
``radiation_group_edges_eV: [13.6, 10000.0]``.

Radiation Spectrum HDF5 Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spectrum generator and its HDF5 schema are documented in
:doc:`radiation_spectrum_generator`.

For reusable spectra, set ``radiation_spectrum_filename`` to an HDF5 file
containing a ``RadiationSpectrum`` group. RadHydropy reads the group during
``Par`` startup and after an initial-condition HDF5 header is restored. The
standard datasets are:

* ``group_edges_eV``;
* ``ionizing_photon_energy_erg``;
* ``star_emission_rates``;
* ``group_sigma_gamma_cm2``; and
* ``group_epsilon_gamma_erg``.

The group metadata must include ``number_of_radiation_groups`` and
``number_of_group_edges``. The optional
``radiation_spectrum_total_photon_rate`` YAML parameter rescales all ionizing
groups by a common factor while preserving their relative spectrum. If it is
omitted, the normalization stored in ``star_emission_rates`` is used.

Limitations
-----------

This is a one-ray transport update for one-dimensional meshes. It supports
multiple frequency groups but does not include scattering, frequency
redistribution, or angular moments from multiple rays. Group cross-sections are
frequency-averaged inputs; resolving spectral hardening within a group
requires using narrower groups.
