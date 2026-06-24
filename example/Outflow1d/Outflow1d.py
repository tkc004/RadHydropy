from radhydropy.rsim import Rsim
import unyt
from radhydropy.analysis import rplot1d
import matplotlib.pyplot as plt
import numpy as np
import radhydropy.utils as ru 
import radhydropy.io as rio
#importing the os module
import os

#to get the current working directory
rundir = os.getcwd()
print('rundir',rundir)


runparams = {
    'simname':'outflow1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outdir':rundir,
    'outfileprefix':'Output', 
    'outdeltatime':2.0*unyt.s *0.1,
    'savedir':rundir,
    'coordsys':'cartesian', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':2.0*unyt.s, # final simulation time
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'OutflowSph',
    'vel_outflow':1.0*unyt.cm/unyt.s,
    'rho_outflow':1.0*unyt.g/unyt.cm**3,
    'temp_outflow':0.0*unyt.K,
    'mu_outflow':1.0,
    'verbose':0, # speak out details?
    'order': 0,
    'noghost':10,
    'dtmin': 2.0e-8*unyt.s,
    'dtmax': 2.0e-2*unyt.s,   
}

ICparams = {
    'nogrid':1000, # number of grid points
    'coordsys':"cartesian", # coordinate system
    'boxsize':2.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'tempini': 0.0 * unyt.K,
    'muini': 1.0,
    'vini':1.0 * unyt.cm/unyt.s, #initial velocity
    'rhoini': 0.001 * unyt.g/ unyt.cm**3, # initial density (of the higher density end)
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
         
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.par.boxsize[0]/self.par.nogrid

        # generate initial condition
        #self.mesh.boundary = np.linspace(-0.5*dx,self.par.boxsize[0]+0.5*dx,self.par.nogrid+1)
        self.mesh.boundary = np.linspace(0.0*self.par.boxsize[0],self.par.boxsize[0],self.par.nogrid+1)
        self.fluid.vel = ICparams["vini"] * np.ones(self.par.nogrid)
        self.fluid.temp = ICparams["tempini"] * np.ones(self.par.nogrid)
        self.fluid.rho = ICparams["rhoini"] * np.ones(self.par.nogrid)
        # mean molecular weight
        self.fluid.mu = ICparams["muini"] * np.ones(self.par.nogrid) # for primordial neutral gas 



# I think the analytic solution only work for constant density
# we cannot do density advection
# we cannot do temperature advection neither
# since temperature changes with density
def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout,yquan='rho',showhalf=0,showfig=0,**kwargs)
    #plt.ylim(ymax=10.1)
    plt.axvline(x = rout.par.time*runparams['vel_outflow'],color=kwargs['color'],ls='dashed')
    #plt.yscale('log')





def main():
    ric = Simwrap()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(0,9,2):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        ReadandPlot(outfilename,ls='none',marker='o', mfc='none', markevery=1,color=next(ax._get_lines.prop_cycler)['color'])
    plt.show()


if __name__ == "__main__":
    main()
