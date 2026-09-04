Direct Radiation Pressure
=========================

RadHydropy can convert the momentum of absorbed photons into a gas momentum
source. This is enabled with ``radiation_pressure: true`` and is separate from
the thermo-chemistry update.

Physical model
--------------

For photons absorbed at a rate ``dot{N}_abs`` with photon energy ``E_gamma``,
the deposited momentum rate is

.. math::

   \dot{p}_{\rm rad} = \epsilon\,s\,
   \frac{\dot{N}_{\rm abs} E_\gamma}{c},

where ``epsilon`` is ``radiation_pressure_efficiency`` and ``s`` is the ray
direction (``+1`` or ``-1``). For multiple photon groups, RadHydropy sums the
energy-weighted contribution from every group.

The absorbed rate is a volumetric rate. Therefore the cell acceleration is

.. math::

   a_{\rm rad} = \frac{\dot{p}_{\rm rad}}{\rho},

and the conserved momentum and energy are updated by

.. math::

   \Delta (\rho V v) = \rho a_{\rm rad} V\Delta t,
   \qquad
   \Delta E = \rho v a_{\rm rad} V\Delta t.

The implementation skips cells with zero density rather than dividing by a
small artificial density.

Runtime ordering
----------------

Radiation pressure is applied after the thermo-chemistry source step:

.. code-block:: python

   source_result = self.solver.ApplyThermochemistryFast(
       dt, self.mesh, self.fluid, self.par,
       transport_result=transport_result,
   )
   self.solver.ApplyRadiationPressure(
       dt, self.mesh, self.fluid, self.par, source_result
   )

Thermo-chemistry calculates the absorbed photon rate and does not update gas
momentum. ``ApplyRadiationPressure`` consumes the returned
``absorbed_photon_rate`` and ``photon_energy_cgs_erg`` fields. The existing
radiative-transfer update remains before thermo-chemistry for the ordinary
instantaneous scheme, and the C\ :sup:`2`-Ray temporal scheme remains selected
and handled by its normal source path.

Configuration
-------------

The basic controls are:

.. code-block:: yaml

   radiation_pressure: true
   radiation_pressure_efficiency: 1.0

The efficiency is dimensionless. A value of one transfers all absorbed photon
momentum to the gas; smaller values model incomplete coupling. The source
direction is supplied by the radiation-transport result. Direct absorption
from several groups is combined using their photon energies.

Verification
------------

The unit tests in ``tests/test_radiation_pressure.py`` check:

* one-cell momentum deposition against ``N_abs E_gamma dt / c``;
* skipping zero-density cells;
* reversal of momentum for the opposite ray direction; and
* energy-weighted multigroup deposition.

The clean source-only benchmark in
``example/RadiationPressureDrivenShell1D/`` follows the standard RadHydropy
simulation workflow. It creates an HDF5 initial condition, starts ``Rsim``,
runs the source update, writes numbered snapshots, and compares the shell
momentum with

.. math::

   p_{\rm expected}(t) = \frac{L t}{c}.

Run it with:

.. code-block:: console

   $ cd example/RadiationPressureDrivenShell1D
   $ python thin_shell_ode.py

This benchmark intentionally has no ambient gas pressure, swept-up mass, or
hydrodynamic boundary flux. It is therefore a direct momentum-conservation
test of the RadHydropy radiation-pressure source. The generated figure is
``RadiationPressureDrivenShell1D_ThinShellODE.jpg``.

For a full hydrodynamic radiation-pressure calculation with photoheating, see
the dynamic Strömgren-sphere example:

``example/DynamicStromgrenSpherePhotoheating20pcRadiationPressure1D/``

In that case the total radial gas momentum need not equal ``Lt/c`` because gas
pressure, hydrodynamic fluxes, shell mass changes, and spherical geometry also
contribute to the radial momentum diagnostic.

