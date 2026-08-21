NFW Virial Shock with HM12 PIE Cooling
=======================================

This separate experiment evolves gas in a fixed ``3e11 Msun`` NFW halo at
``z=0`` with the Haardt--Madau 2012 photoionization-equilibrium (PIE) heating
and cooling table. The domain extends to approximately ``4 R200``. This mass
is chosen to place the post-shock gas near the strong atomic-cooling regime,
while the ``1e12 Msun`` adiabatic example remains the stable-shock control.

Unlike the uniform-background IC used by the adiabatic benchmark, this PIE
run initializes the baryons with a correlated perturbation profile,

   ``rho_b(r) = rho_mean [1 + delta_floor + delta_R200 (r/R200)^(-alpha)]``.

The default ``delta_R200=8``, ``alpha=1.8``, and ``delta_floor=0.25`` keep the
outer boundary above the cosmic mean while increasing the enclosed overdensity
towards the halo. These parameters are the simplified IC controls for the
linear-correlation-function perturbation described by Birnboim and Dekel; the
fluctuation amplitude sets the collapse strength and timing.

Run it from this directory::

   python nfw_virial_shock_pie1d.py

The run writes ``NFWVirialShockPIE1D.jpg`` and
``NFWVirialShockPIE1D_Stability.txt``. The stability report contains the
shock radius, upstream infall speed and density, post-shock temperature, HM12
cooling and heating rates, ``S_cooling``, ``S_net``, and an estimate of
``gamma_eff``.

The Birnboim--Dekel value ``S=0.0126`` applies to pure cooling. With HM12,
the net pressure evolution and ``gamma_eff`` must also be checked because the
UV background can provide photoheating.

Cooling-table data
------------------

The HM12 and CIE/CHIANTI cooling tables are distributed separately through
Git LFS in
`tkc004/RadhydropyData <https://github.com/tkc004/RadhydropyData>`_. Install
Git LFS and clone that repository if the table is not already available in
the repository's sibling data directories::

   git lfs install
   git clone https://github.com/tkc004/RadhydropyData.git

The ``metal_pie_hm12_total.h5`` file should be placed at the path configured
by ``metal_pie_table_filename``. CIE runs additionally require
``chianti_cie_ion_fractions.h5`` and ``chianti_cooling_table.h5``; these can
be selected with ``cie_ion_fraction_table`` and ``cie_cooling_table``.

Control runs
------------

The same runner includes two comparison configurations::

   python nfw_virial_shock_pie1d.py --config nfw_virial_shock_pie_1e12.yaml
   python nfw_virial_shock_pie1d.py --config nfw_virial_shock_pie_1e11.yaml

The ``1e12`` run is the stable-shock control and the ``1e11`` run is the
likely unstable/no-shock control. They use separate output directories and
write ``NFWVirialShockPIE1D_1e12.*`` and ``NFWVirialShockPIE1D_1e11.*``.
