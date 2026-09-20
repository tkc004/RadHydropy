"""Plotting helpers for RadHydropy outputs."""

import matplotlib.pyplot as plt
import numpy as np
import unyt
from radhydropy.units import code_unit_scales
from radhydropy.runtime_fields import (
    runtime_fields,
    select_fluid_primitive_arrays,
    select_mesh_geometry_arrays,
)

def rplot1d(rsim, yquan=None, showfig=1, showhalf=0, **kwargs):
    """Plot a one-dimensional fluid quantity against cell-center position.

    Parameters
    ----------
    rsim : object
        Simulation-like object with typed runtime fluid and mesh state.
    yquan : str, optional
        Canonical representation-specific fluid field to plot. Defaults to the
        active runtime density field.
    showfig : int, optional
        Show and clear the figure when set to 1.
    showhalf : int, optional
        Limit the x-axis to the left half when 1 or right half when 2.
    **kwargs
        Additional keyword arguments passed to ``matplotlib.pyplot.plot``.
    """
    fields = runtime_fields(rsim.par)
    yquan = fields.density if yquan is None else yquan
    if yquan in {"rho_code", "vel_code", "pre_code", "temp_code"}:
        raise ValueError(f"legacy fluid field {yquan!r} is forbidden")
    runtime_state = getattr(rsim.fluid, "runtime_state", None) or rsim.fluid
    mesh_state = getattr(rsim.mesh, "geometry_state", None)
    if mesh_state is None:
        raise ValueError("rplot1d requires typed mesh geometry state")
    _, boundary_code, _, _, _ = select_mesh_geometry_arrays(mesh_state, rsim.par)
    code_units = getattr(rsim.par, "CodeUnits", None)
    scales = code_unit_scales(code_units)
    xq = (
        0.5 * (np.asarray(boundary_code[1:], dtype=float) + np.asarray(boundary_code[:-1], dtype=float))
        * scales["length_cgs_cm"] * unyt.cm
    )
    yraw = np.asarray(getattr(runtime_state, yquan), dtype=float)
    if yquan == fields.density:
        yq = yraw * scales["density_cgs_g_cm3"] * (unyt.g / unyt.cm**3)
        ylabel = r'$\\rho$'
    elif yquan == fields.velocity:
        yq = yraw * scales["velocity_cgs_cm_s"] * (unyt.cm / unyt.s)
        ylabel = r'$v$'
    elif yquan == fields.pressure:
        yq = yraw * scales["pressure_cgs_erg_cm3"] * (unyt.erg / unyt.cm**3)
        ylabel = r'$P$'
    else:
        yq = yraw
        ylabel = yquan
    xlabel = r'$r$'
    plt.plot(xq,yq,**kwargs)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    if showhalf==1:
        plt.xlim(xmax=0.5*np.amax(xq))
    if showhalf==2:
        plt.xlim(xmin=0.5*np.amax(xq))        
    if showfig==1:
        plt.show()
        plt.clf()
