PIEUVBGCoolingUniformSph1D
==========================

This example exercises the non-radiative-transfer HM12
``PIEUVBGCoolingNetwork`` in hydrodynamics. It runs two uniform spherical
gas cases at ``z=4``:

* ``nH=1 cm^-3`` receives HM12 photoheating and cooling;
* ``nH=100 cm^-3`` is self-shielded, so only the HM12 cooling rate is used.

Run it from this directory with::

   python pie_uvbg_cooling_uniform_sph1d.py

The script writes case snapshots under ``outputs/`` and the comparison figure
``PIEUVBGCoolingUniformSph1D.jpg``.

The HM12 table is stored with Git LFS in
`tkc004/RadhydropyData <https://github.com/tkc004/RadhydropyData>`_. Download
it before running this example. From the RadHydropy project root::

   cd ..
   git lfs install
   git clone https://github.com/tkc004/RadhydropyData.git
   cd RadhydropyData
   git lfs pull
   cp -R metal_pie_table ../metal_pie_table
   cd ../RadHydropy

Verify the required table exists::

   test -s ../metal_pie_table/metal_pie_hm12_total.h5
