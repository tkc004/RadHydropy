from radhydropy.rsim import Rsim
import unyt
import os
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams, rc
from radhydropy.analysis import rplot1d
import numpy as np
import radhydropy.utils as ru 
import radhydropy.io as rio
import SedovTaylor_analytic as sa

# Plot parameters
plotparams = {'axes.labelsize': 24,
'axes.titlesize': 24,
'font.size': 24,
'legend.fontsize': 20,
'xtick.labelsize': 15,
'ytick.labelsize': 15,
'xtick.top': True,
'ytick.right': True,
'xtick.bottom': True,
'ytick.left': True,
'xtick.minor.visible': True,
'ytick.minor.visible': True,
'xtick.direction':"in",
'ytick.direction':"in",
'figure.figsize' : (30.45,6.5),
'figure.subplot.left'    : 0.2,
'figure.subplot.right'   : 0.9,
'figure.subplot.bottom'  : 0.2,
'figure.subplot.top'     : 0.85,
'figure.subplot.wspace'  : 0.2,
'figure.subplot.hspace'  : 0.2,
'lines.markersize' : 6,
'lines.linewidth' : 3.,
}
rcParams.update(plotparams)


#to get the current working directory
rundir = os.getcwd()
print('rundir',rundir)


runparams = {
    'simname':'SedovTaylorSph1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outdir':rundir,
    'outfileprefix':'Output', 
    'outdeltatime':1000.0*unyt.s *0.1,
    'savedir':rundir,
    'coordsys':'spherical', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':5./3., # for polytropic, the polytropic index
    'timesim':1000.0*unyt.s, # final simulation time
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'OpenSph',
    'noghost': 10, #number of ghost cells in front (equal number of ghost cell after)
    'verbose':0, # speak out details?
    'order': 0,
    'dtmin': 2.0e-8*unyt.s,
    'dtmax': 2.0e-1*unyt.s,   
}

ICparams = {
    'nogrid':1000, # number of grid points
    'coordsys':"spherical", # coordinate system
    'boxsize':5.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'rhoini': 1.0 * unyt.g/unyt.cm**3,
    'Eini': 1.0e-4 * unyt.erg, # the total energy in central region
    #'Eini': 0.0 * unyt.erg, # the total energy in central region
    'rini': 0.5 * unyt.cm, # radius of central region
    'muini': 1.0,
    "rinj": 0.0*unyt.cm, 
}


class Par():
    def __init__(self) -> None:
        pass

class Mesh():
    def __init__(self) -> None:
        pass

class Fluid():
    def __init__(self) -> None:
        pass

class Simwrap():
    def __init__(self) -> None:
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        # should be read from parameter files instead
        self.par.nogrid = ICparams["nogrid"]
        self.par.coordsys = ICparams["coordsys"]
        self.par.boxsize = ICparams["boxsize"]*np.ones(1)
        self.par.time = ICparams["time"]*np.ones(1)
        Eini = ICparams["Eini"]
         
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.par.boxsize[0]/self.par.nogrid

        # generate initial condition
        self.mesh.boundary = np.linspace(ICparams["rinj"],ICparams["rinj"]+self.par.boxsize[0],self.par.nogrid+1)
        self.mesh.coordinate = 0.5 * (self.mesh.boundary[:-1]+self.mesh.boundary[1:])
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm/unyt.s 
        self.fluid.rho =  ICparams["rhoini"] * np.ones(self.par.nogrid)
        self.mesh.vol = (self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3)*4.0*np.pi/3.0
        # mean molecular weight
        self.fluid.mu = np.ones(self.par.nogrid) * ICparams["muini"] 
        self.fluid.mass = self.fluid.rho*self.mesh.vol 
        #inject Eini to a single particle in the center
        self.fluid.temp = np.ones(self.par.nogrid) * 0.0 * unyt.K
        icut = np.logical_and(self.mesh.coordinate<ICparams['rini'],self.mesh.coordinate>=ICparams['rinj'])
        pre = ICparams["Eini"] / np.sum(self.mesh.vol[icut]) * (runparams['gamma'] - 1.0)
        self.fluid.temp[icut] = ru.CalTemperature(self.fluid.rho[icut],pre,self.fluid.mu[icut])



# I think the analytic solution only work for constant density
# we cannot do density advection
# we cannot do temperature advection neither
# since temperature changes with density
def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rout.fluid.pre = ru.CalPressure(rout.fluid.rho,rout.fluid.temp,rout.fluid.mu)
    nu = 3 # dimension of the problem 
    g  = runparams['gamma'] # polytropic index
    w  = 0.0 # power law slope of the density profile
    E0 = ICparams['Eini']
    A0 = ICparams['rhoini']  # in the case of uniform density
    t = rout.par.time
    r, rho, v, p, Rs = sa.get_blastwave_solution(E0,A0,nu,g,w,t)
    r = np.concatenate((r, unyt.unyt_array([1.0,2]*Rs)))
    rho = np.concatenate((rho, unyt.unyt_array([ICparams['rhoini'],ICparams['rhoini']]))) 
    v = np.concatenate((v,unyt.unyt_array([0.0*unyt.cm/unyt.s, 0.0*unyt.cm/unyt.s])))
    p = np.concatenate((p,unyt.unyt_array([0.0*unyt.dyn/unyt.cm**2, 0.0*unyt.dyn/unyt.cm**2])))
    plt.subplot(1,3,1)
    rplot1d(rout,yquan='pre', showfig=0,**kwargs)
    plt.plot((r ).in_cgs(), (p).in_cgs(), color=kwargs['color'])
    plt.xlim([0,4]) 
    plt.subplot(1,3,2)    
    rplot1d(rout,yquan='vel', showfig=0,**kwargs)
    plt.plot((r ).in_cgs(), (v).in_cgs(), color=kwargs['color'])
    plt.xlim([0,4]) 
    plt.subplot(1,3,3)        
    rplot1d(rout,yquan='rho', showfig=0,**kwargs)
    plt.plot((r ).in_cgs(), (rho).in_cgs(), color=kwargs['color'])
    plt.xlim([0,4]) 




def main():
    ric = Simwrap()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(5,10):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        ReadandPlot(outfilename,ls='none',marker='o', mfc='none', markevery=1,color=next(ax._get_lines.prop_cycler)['color'])
    figure_filename = rundir + '/SedovTaylorSph1D.jpg'
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)

if __name__ == "__main__":
    main()
