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

Limitations
-----------

This is a one-frequency, one-ray transport update for one-dimensional meshes.
It is intended for idealized finite-volume tests and for coupling an attenuated
photon field to the existing hydrogen chemistry. It does not yet solve a
multi-frequency transfer problem, include scattering, or construct angular
moments from multiple rays.
