import matplotlib.pyplot as plt
import numpy as np
import unyt

def rplot1d(rsim, yquan='rho',showfig=1,showhalf=0,**kwargs):
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
