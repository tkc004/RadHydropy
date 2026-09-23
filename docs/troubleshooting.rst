Troubleshooting
===============

Most example failures are caused by the launch directory, configuration
paths, missing generated inputs, or an HDF5 file written with an incomplete
unit header. The checks below cover the common cases.

The example cannot import ``tools`` or cannot find a YAML file
--------------------------------------------------------------

Run the command from the example directory, not from the repository root:

.. code-block:: bash

   cd example/SodShock1D
   python sodshock1d.py

Variant configurations must be passed explicitly from that directory:

.. code-block:: bash

   cd example/StellarWindBubble1D
   python stellar_wind_bubble1d.py \
      --config stellar_wind_bubble1d_no_metal.yaml

Relative output directories and local ``tools.py`` imports are resolved from
the current working directory.

The loader reports a missing ``CodeUnits`` header
--------------------------------------------------

Current HDF5 loading requires ``Header.attrs["CodeUnits"]``. Regenerate the
initial condition with the maintained writer workflow instead of manually
copying datasets:

.. code-block:: python

   config = load_nested_example_config("sodshock1d.yaml")
   writer = build_initial_condition(config)
   writer.write("InitialCondition.hdf5", validate=True)

Then load the file with the complete nested configuration using
``radhydropy.io.loadhdf5(config, filename)``.

The run fails with a cosmological startup-time error
------------------------------------------------------

Einstein--de Sitter and other expanding-background runs require a strictly
positive cosmic startup time. Check the selected YAML for a zero-valued
``initial_condition.time_cosmic`` such as:

.. code-block:: yaml

   initial_condition:
     time_cosmic: {value: 0.0, unit: s}

Replace it with the example’s intended positive value. Keep these three flags
consistent for a supercomoving workflow:

.. code-block:: yaml

   par:
     cosmology:
       cosmological: true
       cosmological_expansion: true
       supercomoving_coordinates: true

Do not replace the structured ``par.cosmology`` container with a bare
background model in an initial-condition builder.

The cosmological correlation run cannot find its input table
-------------------------------------------------------------

The correlation-function workflows depend on generated HDF5 tables. Confirm
the configured table exists before starting the expensive evolution. For the
Tvir=1e3, z=15 workflow, use the canonical runner and configuration from its
directory:

.. code-block:: bash

   cd example/CosmologicalVirialShock1D
   python cosmological_gas_correlation_z100.py \
      --config cosmological_gas_correlation_tvir1e3_z15.yaml

If the finite-box correlation table is absent, generate it with the project’s
``tools.lcdm_correlation`` utility. For example, from the repository root:

.. code-block:: python

   from tools.lcdm_correlation import generate_lcdm_correlation_table

   generate_lcdm_correlation_table(
       "example/CosmologicalVirialShock1D/outputs_correlation/"
       "eds_finite_box_1Mpc_linear_correlation.h5"
   )

Use the exact output path expected by the selected YAML. A missing generated
artifact is an input-preparation problem, not necessarily a solver failure.
See :doc:`cosmological_virial_shock1d` for the full correlation workflow.

The run prints C²-Ray non-convergence warnings
-----------------------------------------------

Some H II-region configurations deliberately set
``c2ray_nonconvergence: warn``. Cell-level warnings can therefore appear in a
successful run. Check that the configured final time was reached and that the
expected HDF5 snapshots and figures were written before treating the warning
as a failure.

The output contains unexpected generic field names
---------------------------------------------------

Runtime fields are representation-specific. Use names such as
``rho_proper_code``, ``rho_comoving_code``, ``vel_supercomoving_code``, and
``tau_supercomoving_code``. For dimensional analysis, use typed fields such
as ``snapshot.fluid.rho_radarray`` and
``snapshot.dark_matter.radius_radarray``. Do not infer physical units from a
``_code`` suffix alone; the HDF5 field metadata and ``CodeUnits`` header are
authoritative.

The example runs but produces no useful plot
---------------------------------------------

First confirm that the output file exists in the example directory. Then
check that the plotting configuration is in the nested ``example`` section,
not at the YAML top level. For H II-region plots, for example:

.. code-block:: yaml

   example:
     show_stagnation_radius: true

Regenerate plots from saved HDF5 snapshots when possible. This avoids rerunning
long simulations just to change a diagnostic overlay.

The run is unexpectedly slow or unstable
-----------------------------------------

Use the validated YAML before changing solver defaults. In particular:

* preserve ``positivity_factor_method: invariant_domain`` where it is part of
  a validated example configuration;
* keep the spherical converging-shock benchmark’s configured dual-energy and
  positivity settings when comparing results; and
* compare generated profiles, conservation diagnostics, and admissibility—not
  runtime alone—when changing a limiter or Riemann solver.

Reporting a reproducible failure
---------------------------------

Include the following when reporting an issue:

* the repository commit and Python version;
* the example directory and exact command;
* the YAML configuration or any command-line overrides;
* the first traceback or warning, rather than only the final summary; and
* whether ``InitialCondition.hdf5`` and the expected ``Output_*.hdf5`` files
  were generated.
