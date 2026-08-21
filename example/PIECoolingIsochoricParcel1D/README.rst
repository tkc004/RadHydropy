PIE Cooling Isochoric Parcel
============================

This one-zone benchmark evolves fixed-density gas with the non-radiative-
transfer HM12 photoionization-equilibrium heating and cooling network. It
tests the thermal source update independently of gravity, advection, and
shock motion.

The example runs four cases at ``z=0``:

* diffuse gas with ``nH=1e-4 cm^-3`` initially at ``1e4 K``;
* diffuse gas with ``nH=1e-4 cm^-3`` initially at ``1e6 K``;
* dense gas with ``nH=1e-2 cm^-3`` initially at ``1e4 K``; and
* dense gas with ``nH=1e-2 cm^-3`` initially at ``1e6 K``.

Run it from this directory::

   python pie_cooling_isochoric_parcel1d.py

The script writes per-case HDF5 snapshots under ``outputs/``, together with
``PIECoolingIsochoricParcel1D.jpg`` and
``PIECoolingIsochoricParcel1D_ThermalReport.txt``. The report compares the
simulated final temperature with the stable HM12 equilibrium temperature and
records the initial net thermal rate, initial thermal timescale, and density
drift.

The HM12 table is stored with Git LFS in
`tkc004/RadhydropyData <https://github.com/tkc004/RadhydropyData>`_. Install
Git LFS and fetch it if the sibling ``metal_pie_table/`` directory is missing::

   git lfs install
   git clone https://github.com/tkc004/RadhydropyData.git
   cd RadhydropyData
   git lfs pull
