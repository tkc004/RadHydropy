"""Plotting helpers for RadHydropy outputs."""

import matplotlib.pyplot as plt
import numpy as np
import unyt

def rplot1d(rsim, yquan='rho',showfig=1,showhalf=0,**kwargs):
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
    xb = rsim.mesh.boundary.in_cgs()
    xq = 0.5*(xb[1:]+xb[:-1])
    yq = getattr(rsim.fluid,yquan)
    yq = yq.in_cgs()
    plt.plot(xq,yq,**kwargs)
    plt.xlabel(r'$'+xq.in_cgs().units.latex_repr+'$',fontsize=24)
    plt.ylabel(r'$'+yq.in_cgs().units.latex_repr+'$',fontsize=24)
    if showhalf==1:
        plt.xlim(xmax=0.5*np.amax(xq))
    if showhalf==2:
        plt.xlim(xmin=0.5*np.amax(xq))        
    if showfig==1:
        plt.show()
        plt.clf()
