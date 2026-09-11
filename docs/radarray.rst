RadArray and RadQuantity
========================

``RadArray`` and ``RadQuantity`` are the representation-aware unit
boundaries used by RadHydropy examples. They extend ``unyt`` arrays and
scalars with:

* the configured :class:`radhydropy.units.CodeUnits`;
* a canonical :class:`radhydropy.field_metadata.FieldSpec`; and
* a :class:`radhydropy.cosmology_context.CosmologyContext` when proper,
  comoving, or supercomoving conversion is relevant.

``RadArray`` is for mesh and cell-wise values. ``RadQuantity`` is its scalar
counterpart, commonly used for a box size or another scalar initial-condition
input.

Creating values with ``InitialConditionWriter``
------------------------------------------------

An initial-condition builder receives the complete nested ``config`` mapping.
After constructing the configured writer, use ``radarray`` and ``radquantity``
with unit-bearing values. The writer infers the canonical primitive field from
the value's dimensions and from whether the configured run is proper or
cosmological:

.. code-block:: python

   from radhydropy.initial_condition_writer import InitialConditionWriter
   from radhydropy.units import CodeUnits

   units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
   writer = InitialConditionWriter(
       par_config=config["par"],
       code_units=units,
   )
   writer.box_size = writer.radquantity(
       config["initial_condition"]["box_size_proper"]
   )
   writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
   writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
   writer.fluid.vel_radarray = writer.radarray(velocity_proper_unyt)
   writer.fluid.temp_radarray = writer.radarray(temperature_proper_unyt)
   writer.write("InitialCondition.hdf5")

The input values must carry units. Use ``.to_value(units.<dimension>_unit)``
when an explicitly converted numerical code array is required, but do not
construct a unitless array and attach units later. A time quantity or another
dimension that does not identify one canonical primitive field is rejected.

For cosmological runs, the writer creates ``boundary_comoving_code``,
``rho_comoving_code``, ``vel_supercomoving_code``,
``temp_supercomoving_code``, and ``pre_supercomoving_code`` views according to
the configured schema. A supercomoving velocity conversion also requires
``x_comoving_code`` because it includes the Hubble-flow term.

Reading values after ``loadhdf5``
---------------------------------

Always use the restored ``*_radarray`` views for dimensional data after
loading an initial condition or snapshot:

.. code-block:: python

   import radhydropy.io as rio

   sim = rio.loadhdf5(config, "Output_001.hdf5")
   boundary_comoving_unyt = sim.mesh.boundary_radarray
   density_comoving_unyt = sim.fluid.rho_radarray
   temperature_supercomoving_unyt = sim.fluid.temp_radarray

These views preserve units, field metadata, representation, and cosmology.
Use explicit conversions for analysis:

.. code-block:: python

   density_cgs_g_cm3 = density_comoving_unyt.to_cgs()
   density_code = density_comoving_unyt.value

The ``*_code`` attributes, such as ``rho_comoving_code``, are plain numeric
solver state. They remain useful inside solver-facing code, but example
readers, plotters, and diagnostics should use the RadArray views instead.

Representation conversion
--------------------------

Non-velocity fields can be converted between the proper and cosmological
representations while retaining their metadata:

.. code-block:: python

   rho_proper_unyt = density_comoving_unyt.to_proper()
   rho_comoving_unyt = rho_proper_unyt.to_comoving()

For velocity, provide the matching comoving coordinate array:

.. code-block:: python

   velocity_proper_unyt = sim.fluid.vel_radarray.to_proper(
       x_comoving_code=sim.mesh.x_comoving_code,
   )

Adding or subtracting RadArray/RadQuantity values requires matching
representations and cosmology contexts. Multiplication and division produce a
derived field with explicit dimensional metadata.

API reference
-------------

.. autoclass:: radhydropy.radarray.RadArray
   :members:

.. autoclass:: radhydropy.radarray.RadQuantity
   :members:

