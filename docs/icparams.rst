Initial-Condition Parameters
============================

The ``ICparams`` block defines the initial fluid state that gets written to
``InitialCondition.hdf5`` before the run starts. The exact keys depend on the
example, but the common ones are:

* ``nogrid``: number of active grid cells.
* ``coordsys``: geometry used to build the mesh.
* ``boxsize``: physical size of the domain.
* ``time``: initial simulation time stored in the HDF5 file.
* ``rhoini``, ``vini``, ``tempini``, and ``muini``: initial density,
  velocity, temperature, and mean molecular weight.
* ``rhoratio`` and ``tempratio``: optional profile-shaping parameters used by
  some example setups, such as shock tubes.

As with ``runparams``, unit-bearing values are written as ``value`` /
``unit`` pairs in YAML. The example scripts convert ``ICparams`` into the
initial-condition file before calling :class:`radhydropy.rsim.Rsim`.

Example YAML
------------

The Sod shock example uses a compact ``ICparams`` block like this:

.. code-block:: yaml

   ICparams:
     nogrid: 1000
     coordsys: cartesian
     boxsize:
       value: 4.0
       unit: cm
     time:
       value: 0.0
       unit: s
     rhoini:
       value: 1.0
       unit: g/cm**3
     vini:
       value: 0.0
       unit: km/s
     tempini:
       value: 1.5506894880146205e-08
       unit: K
     muini: 1.0
     rhoratio: 0.1
     tempratio: 0.8

These values are loaded with :func:`radhydropy.example_config.load_example_parameters`
and written to ``InitialCondition.hdf5`` before the simulation starts.
