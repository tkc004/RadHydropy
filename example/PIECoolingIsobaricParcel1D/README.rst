PIE Cooling Isobaric Parcel
===========================

This one-zone benchmark evolves HM12 photoionization-equilibrium (PIE) gas
approximately at constant pressure. It integrates the isobaric enthalpy
equation, with ``nH*T`` held fixed, so cooling and heating automatically drive
the density in the opposite direction to the temperature.

The experiment tests:

* thermal instability near rapidly varying portions of the PIE cooling curve;
* density evolution during approximately isobaric cooling or heating;
* the effective equation of state ``gamma_eff = d ln P / d ln rho``;
* the isobaric perturbation growth rate ``d(dT/dt)/dT``.

It runs four ``z=0`` HM12 cases, spanning diffuse and dense gas and hot and
cold starting points. A positive perturbation growth rate in the lower-right
panel indicates local isobaric thermal instability.

Run it from this directory::

   python pie_cooling_isobaric_parcel1d.py

The script writes one saved CSV trajectory per case under ``outputs/``,
``PIECoolingIsobaricParcel1D.jpg``, and
``PIECoolingIsobaricParcel1D_ThermalReport.txt``.
The companion ``PIECoolingIsobaricParcel1D_Rate.jpg`` plots
``abs(Gamma-C)/nH**2`` in ``erg cm^3 s^-1`` along the constant-pressure
cooling curves; solid portions are net heating and dashed portions are net
cooling.

The HM12 table is stored with Git LFS in
`tkc004/RadhydropyData <https://github.com/tkc004/RadhydropyData>`_. Install
Git LFS and fetch it if the sibling ``metal_pie_table/`` directory is missing::

   git lfs install
   git clone https://github.com/tkc004/RadhydropyData.git
   cd RadhydropyData
   git lfs pull
