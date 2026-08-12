Gas--Dark-Matter Shell Coupling
===============================

This example evolves spherical gas with both gas self-gravity and live
dark-matter shells. The gas density is used to accelerate the shells, while
the shell enclosed mass contributes to the gas gravity source term.

Run from this directory::

   python gas_dark_matter_shell_coupling1d.py \
       --config gas_dark_matter_shell_coupling1d.yaml

The first coupling implementation is intentionally short and diagnostic. It
evolves for approximately half a dynamical time, plots initial and final gas
density profiles, uses synchronized hydro and shell updates, shell-crossing
timestep control, and writes the gas output using the normal RadHydropy workflow.
It also checks conservation of total gas and dark-matter mass. For the default
run, both relative mass errors are at floating-point roundoff level.

Dark-matter restart output is supported by the ``DarkMatter`` HDF5 group, but
automatic reconstruction of a live shell object on restart is not yet provided.
