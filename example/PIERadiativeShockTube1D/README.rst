PIE Radiative Shock Tube
========================

This example is a planar colliding-flow shock tube with HM12
photoionization-equilibrium (PIE) heating and cooling. It is a hydrodynamic
benchmark between a local PIE cooling layer and the more complicated
spherical virial-shock problem.

The experiment tests:

* shock jump conditions against a strong-shock estimate;
* the post-shock cooling layer and its temperature profile;
* the expected cooling length
  ``ell_cool ~= u_post * t_cool``;
* comparison with an adiabatic control;
* density and metallicity dependence of the cooling layer.

The run includes solar-metallicity and ``Z=0.1`` PIE cases at
``nH=1e-3 cm^-3``, a denser solar-metallicity case at ``nH=1e-2 cm^-3``,
and an adiabatic control. The colliding streams have ``100 km/s`` speed and
initial temperature ``1e4 K``.

Run it from this directory::

   python pie_radiative_shock_tube_1d.py

The script writes saved HDF5 snapshots under ``outputs/``, the comparison
figure ``PIERadiativeShockTube1D.jpg``, and
``PIERadiativeShockTube1D_ShockReport.txt``. The report gives the measured
compression, strong-shock expectation, post-shock temperature, and expected
and measured cooling lengths.

The HM12 table is stored with Git LFS in
`tkc004/RadhydropyData <https://github.com/tkc004/RadhydropyData>`_. Install
Git LFS and fetch it if the sibling ``metal_pie_table/`` directory is missing::

   git lfs install
   git clone https://github.com/tkc004/RadhydropyData.git
   cd RadhydropyData
   git lfs pull
