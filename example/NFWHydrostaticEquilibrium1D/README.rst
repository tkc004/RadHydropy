NFW Hydrostatic Equilibrium 1D
==============================

This example models gas in hydrostatic equilibrium inside a ``1e8 Msun`` NFW
dark-matter halo. It assumes ``z=0``, an overdensity of ``200`` relative to the
critical density, concentration ``c=10``, ionized-gas mean molecular weight
``mu=0.59``, and a gas mass equal to ``0.157 M_200``.

The gas is isothermal at the halo virial temperature, calculated from

.. math::

   T_{\rm vir} = \frac{\mu m_p}{2 k_B}\frac{G M_{200}}{R_{200}}.

For the NFW potential, the hydrostatic density profile is

.. math::

   \rho_g(r) = \rho_g(r_0)
   \exp\left[-\frac{\mu m_p}{k_B T_{\rm vir}}
   \left(\Phi(r)-\Phi(r_0)\right)\right].

Run from this directory with::

   python nfw_hydrostatic_equilibrium1d.py

The output figure compares the evolved density and radial velocity with the
analytic hydrostatic profile. The NFW halo is supplied as an external gravity
potential; gas self-gravity is not included.
