"""Plotting helpers for RadHydropy outputs."""

import matplotlib.pyplot as plt
import numpy as np
import unyt
from radhydropy.units import code_unit_scales

def rplot1d(rsim, yquan='rho_code',showfig=1,showhalf=0,**kwargs):
    """Plot a one-dimensional fluid quantity against cell-center position.

    Parameters
    ----------
    rsim : object
        Simulation-like object with ``mesh.boundary`` and ``fluid`` attributes.
    yquan : str, optional
        Name of the fluid quantity to plot.
    showfig : int, optional
        Show and clear the figure when set to 1.
    showhalf : int, optional
        Limit the x-axis to the left half when 1 or right half when 2.
    **kwargs
        Additional keyword arguments passed to ``matplotlib.pyplot.plot``.
    """
    code_units = getattr(rsim.par, "CodeUnits", None)
    scales = code_unit_scales(code_units)
    if hasattr(rsim.mesh.boundary, "in_cgs"):
        xb = rsim.mesh.boundary.in_cgs()
        xq = 0.5 * (xb[1:] + xb[:-1])
        yq = getattr(rsim.fluid, yquan).in_cgs()
        xlabel = r'$' + xq.in_cgs().units.latex_repr + '$'
        ylabel = r'$' + yq.in_cgs().units.latex_repr + '$'
    else:
        xb = np.asarray(rsim.mesh.boundary, dtype=float) * scales["length_cm"] * unyt.cm
        xq = 0.5 * (xb[1:] + xb[:-1])
        yraw = np.asarray(getattr(rsim.fluid, yquan), dtype=float)
        if yquan == 'rho_code':
            yq = yraw * scales["density_g_cm3"] * (unyt.g / unyt.cm**3)
            ylabel = r'$\\rho$'
        elif yquan == 'vel_code':
            yq = yraw * scales["velocity_cm_s"] * (unyt.cm / unyt.s)
            ylabel = r'$v$'
        elif yquan == 'pre_code':
            yq = yraw * scales["pressure_erg_cm3"] * (unyt.erg / unyt.cm**3)
            ylabel = r'$P$'
        else:
            yq = yraw
            ylabel = yquan
        xlabel = r'$r$'
    plt.plot(xq,yq,**kwargs)
    if hasattr(xq, "in_cgs"):
        plt.xlabel(r'$'+xq.in_cgs().units.latex_repr+'$',fontsize=24)
    else:
        plt.xlabel(xlabel, fontsize=24)
    if hasattr(yq, "in_cgs"):
        plt.ylabel(r'$'+yq.in_cgs().units.latex_repr+'$',fontsize=24)
    else:
        plt.ylabel(ylabel, fontsize=24)
    if showhalf==1:
        plt.xlim(xmax=0.5*np.amax(xq))
    if showhalf==2:
        plt.xlim(xmin=0.5*np.amax(xq))        
    if showfig==1:
        plt.show()
        plt.clf()
