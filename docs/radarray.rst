RadArray and RadQuantity
========================

RadHydropy uses two metadata-carrying types at the dimensional-data boundary:

``RadArray``
   A subclass of ``unyt_array`` for mesh and cell-wise values.
``RadQuantity``
   A scalar counterpart for values such as a box size or scalar field input.

They contain more than numerical values and units. Each object carries:

* the configured :class:`radhydropy.units.CodeUnits`;
* a :class:`radhydropy.field_metadata.FieldSpec` describing quantity,
  representation, coordinate frame, and storage relation; and
* a :class:`radhydropy.cosmology_context.CosmologyContext` containing the
  scale factor, equation-of-state index, cosmology, and Hubble parameter when
  cosmological conversion is relevant.

This distinction is important:

.. code-block:: text

   RadArray = metadata-aware unyt_array
       |
       |  .to_proper(), .to_comoving(), arithmetic
       |  preserve or derive representation metadata
       v
   RadArray

   .to_cgs() or .to_value(...)
       |
       v
   ordinary unyt array or NumPy array for analysis

Do not name a value obtained directly from ``*_radarray`` as ``*_unyt``. Use
``*_radarray`` to show that it still carries RadHydropy metadata. Reserve
``*_unyt`` for ordinary unit-bearing input values or results after an explicit
conversion boundary.

Creating RadArray values
------------------------

Initial-condition builders normally create RadArray values through
``InitialConditionWriter``. The input to the factory must already be
unit-bearing:

.. code-block:: python

   from radhydropy.initial_condition_writer import InitialConditionWriter
   from radhydropy.units import CodeUnits

   units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
   writer = InitialConditionWriter(
       par_config=config["par"],
       code_units=units,
       ic_config=config["initial_condition"],
   )
   writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
   writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
   writer.fluid.vel_radarray = writer.radarray(velocity_proper_unyt)
   writer.fluid.temp_radarray = writer.radarray(temperature_proper_unyt)
   writer.write("InitialCondition.hdf5", validate=True)

The writer selects the canonical field from the configured geometry and the
input dimensions. Keep physical input values unit-bearing until assignment;
do not construct a unitless NumPy array and attach units afterward.

For cosmological configurations, the writer creates representation-aware fields
such as ``boundary_comoving_code``, ``rho_comoving_code``,
``vel_supercomoving_code``, ``temp_supercomoving_code``, and
``pre_supercomoving_code``. A velocity conversion between proper and
supercomoving representations also requires the matching ``x_comoving_code``
because the Hubble-flow term depends on position.

Reading values after ``loadhdf5``
---------------------------------

``loadhdf5`` returns an ``Rsim`` whose mesh and fluid accessors expose actual
``RadArray`` objects:

.. code-block:: python

   import radhydropy.io as rio
   from radhydropy.radarray import RadArray

   sim = rio.loadhdf5(config, "Output_001.hdf5")
   boundary_radarray = sim.mesh.boundary_radarray
   density_radarray = sim.fluid.rho_radarray
   temperature_radarray = sim.fluid.temp_radarray

   assert isinstance(boundary_radarray, RadArray)
   assert density_radarray.representation in {"proper", "comoving"}
   cosmology_context = density_radarray.cosmology

The returned objects preserve their field metadata and cosmology context. The
neutral accessor names ``boundary_radarray``, ``rho_radarray``, and
``temp_radarray`` do not mean that the values are always proper: inspect
``.representation``, ``.field_spec``, and ``.cosmology`` when the distinction
matters.

The canonical plain runtime fields are separate numerical solver state. For
example, a cosmological simulation has fields such as
``rho_comoving_code`` and ``vel_supercomoving_code``. Those fields do not
carry units or RadArray metadata and are intended for solver internals.

Representation conversion
--------------------------

Convert representations while the value is still a RadArray. The conversion
returns another RadArray with an updated ``FieldSpec`` and the same compatible
cosmology context:

.. code-block:: python

   density_proper_radarray = density_radarray.to_proper()
   density_comoving_radarray = density_proper_radarray.to_comoving()

The supported conversions include proper/comoving radius and density,
proper/supercomoving temperature and pressure, and proper/supercomoving
velocity. Velocity conversion requires a comoving position:

.. code-block:: python

   velocity_proper_radarray = sim.fluid.vel_radarray.to_proper(
       x_comoving_code=sim.mesh.x_comoving_code,
   )

The conversion uses the scale factor and, for velocity, the Hubble parameter
stored in ``RadArray.cosmology``. It is therefore not equivalent to changing
the displayed unit with ``.to(...)``.

Analysis conversions
---------------------

Use ``to_value`` when a plotting or numerical library needs a NumPy array:

.. code-block:: python

   radius_kpc = boundary_radarray.to_value("kpc")
   density_g_cm3 = density_radarray.to_value("g/cm**3")
   temperature_K = temperature_radarray.to_value("K")

Use ``to_cgs`` when an ordinary cgs ``unyt_array`` is useful:

.. code-block:: python

   density_cgs_unyt = density_radarray.to_cgs()

These methods are explicit analysis boundaries. Their results no longer carry
the complete RadHydropy field and cosmology metadata, so perform
``to_proper`` or ``to_comoving`` before calling them if a representation
conversion is needed later.

RadArray arithmetic
-------------------

Addition and subtraction require matching representations and compatible
cosmology contexts. This prevents accidentally combining proper and comoving
fields:

.. code-block:: python

   # Valid: result remains a RadArray with compatible metadata.
   density_sum_radarray = density_radarray + density_radarray

   # Convert first when the representations differ.
   density_sum_radarray = (
       density_radarray.to_proper() + density_comoving_radarray.to_proper()
   )

Multiplication and division create a derived RadArray with combined dimensions
and an explicit derived field specification. Use this for dimensional analysis
while retaining the representation context. Convert to ordinary unyt or NumPy
values only after the derived quantity is complete.

Inspecting metadata
-------------------

The most useful metadata attributes are:

``array.field_spec``
   Canonical quantity, dimensions, representation, coordinate frame, storage
   convention, and physical relation.
``array.representation``
   Short representation name, such as ``proper``, ``comoving``, or
   ``supercomoving``.
``array.cosmology``
   Immutable scale-factor, gamma, cosmology, and Hubble-parameter context.
``array.code_units``
   The configured RadHydropy code-unit system.

For a loaded snapshot, dataset-level storage metadata is also available from
``sim.par.field_metadata``:

.. code-block:: python

   for field_name, metadata in sim.par.field_metadata.items():
       print(
           field_name,
           metadata.get("storage_unit"),
           metadata.get("representation"),
           metadata.get("coordinate_frame"),
       )

Use this metadata rather than inferring a physical representation from a
dataset name or assuming that every unit-bearing field is stored in cgs.

API reference
-------------

.. autoclass:: radhydropy.radarray.RadArray
   :members:

.. autoclass:: radhydropy.radarray.RadQuantity
   :members:

See also :doc:`snapshots` for HDF5 field formats and
:doc:`cosmology` for the physical meaning of supercomoving variables.
