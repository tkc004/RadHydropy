Static Strömgren C²-Ray comparison
===================================

This example uses the same 256-cell, constant-temperature spherical setup as
``StaticStromgrenSphere1D``. It runs the C²-Ray temporal update with 100 global
steps and compares it with the instantaneous update using 100, 1,000, 10,000,
and 100,000 steps.

Run it from this directory with::

   python static_stromgren_c2ray_comparison.py

The script writes each run below ``comparison_runs/`` and creates
``StaticStromgrenC2RayComparison_IFront.jpg`` plus a CSV containing every
front-history sample.

The upper plot panel contains the front trajectories and the analytic
Strömgren solution. The lower panel plots
``(R - R_100000) / R_100000`` for every numerical case, where ``R_100000`` is
the instantaneous 100,000-step trajectory sampled at the same times. See
``../../docs/static_stromgren_c2ray_comparison.rst`` for the full description.
