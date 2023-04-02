import matplotlib.pyplot as plt
import numpy as np
import unyt

def rplot1d(rsim, yquan='rho',showfig=1,savefig=0):
    xq = rsim.fluid.mesh.xmesh.to('pc')
    if yquan=='rho':
        yq =  rsim.fluid.rho.in_cgs()
    plt.plot(xq,yq,ls='dotted')
    plt.xlabel(r'$'+xq.to('pc').units.latex_repr+'$',fontsize=24)
    plt.ylabel(r'$'+yq.in_cgs().units.latex_repr+'$',fontsize=24)
    if showfig==1:
        plt.show()
    if savefig==1:
        plt.savefig(rsim.params['savedir']+'/'+rsim.params['simname']+'.png')
    plt.clf()