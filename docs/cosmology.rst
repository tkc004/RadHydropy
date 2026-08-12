Supercomoving Cosmology
========================

RadHydropy's cosmological mode uses supercomoving variables following the
formulation of Martel and Shapiro and its use in Gnedin et al. The stored
coordinate is the comoving radius ``x`` and the stored time is the
supercomoving time ``tau`` defined by

.. math::

   d\tau = dt/a^2.

For the Einstein--de Sitter background,

.. math::

   a(t) \propto t^{2/3}, \qquad
   \varrho=a^3\rho, \qquad
   u=Hax+v/a.

Here ``varrho`` is the comoving density and ``v`` is the supercomoving
peculiar velocity. For an ideal gas, the stored temperature and pressure use
the corresponding supercomoving scaling:

.. math::

   \tilde p=a^{3\gamma}p, \qquad
   \tilde E=\frac{1}{2}\varrho v^2+rac{\tilde p}{\gamma-1}.

For ``gamma=5/3``, a homogeneous adiabatically expanding gas has constant
stored density, velocity, pressure, and energy.

Modified equations
------------------

The physical Euler equations use proper position ``r``, cosmic time ``t``,
proper velocity ``u``, density ``rho``, pressure ``p``, and total energy
density ``E``. Under the transformations above, the continuity equation is

.. math::

   \frac{\partial\varrho}{\partial\tau}
   +\nabla_x\cdot(\varrho v)=0.

There is no explicit ``3 H rho`` dilution term. The momentum equation becomes

.. math::

   \frac{\partial(\varrho v)}{\partial\tau}
   +\nabla_x\cdot\left(\varrho vv+\tilde p I\right)
   =-\varrho\nabla_x\varphi.

The usual Hubble-drag term is absorbed into the definition of ``v``. For
``gamma=5/3``, the total-energy equation is

.. math::

   \frac{\partial\tilde E}{\partial\tau}
   +\nabla_x\cdot\left[(\tilde E+\tilde p)v\right]
   =-\varrho v\cdot\nabla_x\varphi.

For a general adiabatic index, an additional expansion source remains:

.. math::

   (5-3\gamma)a^2H\frac{\tilde p}{\gamma-1}.

It vanishes for the ``gamma=5/3`` Bertschinger gas benchmark.

The Poisson equation is the principal place where cosmology remains explicit:

.. math::

   \nabla_x^2\varphi
   =4\pi G a\left(\varrho-\bar{\varrho}\right),

where ``bar(varrho)`` is the constant comoving background density. In
spherical symmetry,

.. math::

   g_{\rm sc}(x)=-\frac{G a\,\Delta M(<x)}{x^2},
   \qquad
   \Delta M(<x)=4\pi\int_0^x
   [\varrho(x')-\bar{\varrho}]x'^2\,dx'.

The homogeneous background is therefore subtracted from the local collapse
force; it drives the global scale factor instead.

Spherical finite-volume form
----------------------------

The supercomoving spherical update uses the usual finite-volume form in
``(x, tau)``:

.. math::

   \frac{dU_i}{d\tau}=-\frac{A_{i+1/2}F_{i+1/2}-A_{i-1/2}F_{i-1/2}}{V_i}+S_i,

with fluxes

.. math::

   F_\rho=\varrho v,\qquad
   F_m=\varrho v^2+\tilde p,\qquad
   F_E=(\tilde E+\tilde p)v.

The spherical pressure-geometry source uses supercomoving pressure:

.. math::

   S_{m,i}=\tilde p_i
   \frac{A_{i+1/2}-A_{i-1/2}}{V_i}.

The CFL timestep is measured in supercomoving time and uses the
supercomoving sound speed ``tilde(c_s) = a c_s``:

.. math::

   \Delta\tau=C_{\rm CFL}\frac{\Delta x}{|v|+\tilde c_s}.

Physical output recovery
------------------------

The physical fields reconstructed from a supercomoving snapshot are

.. math::

   r=ax,\qquad
   \rho=\frac{\varrho}{a^3},\qquad
   u=Hax+\frac{v}{a},\qquad
   p=\frac{\tilde p}{a^{3\gamma}}.

For ``gamma=5/3``, the temperature conversion is ``T = tilde(T)/a**2``.
The shared ``example_utils.snapshot_physical_fields`` helper applies these
conversions from the HDF5 cosmology header.

Enable the mode with::

   cosmological_expansion: true
   supercomoving_coordinates: true
   cosmology_type: einstein_de_sitter

The HDF5 ``Header`` records the coordinate, time, velocity, density, pressure,
and temperature representations. Dataset attributes record the physical
conversion relation. The header also contains ``CosmologyType``,
``CosmologyTRef``, ``CosmologyARef``, ``CosmicTime``, ``SupercomovingTime``,
``ScaleFactor``, and ``HubbleParameter``. These same fields are read from an
initial-condition or restart file, so the file's cosmology is available even
when the input YAML only supplies the file path.

Example analysis code should use
``example.example_utils.snapshot_physical_fields``. It reads the metadata and
returns physical boundary, radius, density, velocity, and temperature arrays.
This avoids accidentally treating supercomoving output as physical output.

The current implementation provides the variable transformations and the
ordinary-Euler supercomoving hydro path. Density-contrast Poisson gravity and
cosmological dark-matter shell evolution are subsequent phases.
