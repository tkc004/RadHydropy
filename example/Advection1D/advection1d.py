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
    'simname':'advection1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outdir':rundir,
    'outfileprefix':'Output', 
    'outdeltatime':1.0*unyt.s *0.1,
    'savedir':rundir,
    'coordsys':'cartesian', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':1.0*unyt.s, # final simulation time
    'ngrid':1000, # number of grid to discretize
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'Periodic',
    'verbose':0, # speak out details?
}

ICparams = {
    'nogrid':1000, # number of grid points
    'coordsys':"cartesian", # coordinate system
    'boxsize':1.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'rhoini':1.0 * unyt.g/ unyt.cm**3, # initial density (of the higher density end)
    'vini':1.0 * unyt.cm/unyt.s, #initial velocity
    'tempini': 1e-10 * unyt.K,
    'muini': 1.0,
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
        rhoini = ICparams["rhoini"]
        vini = ICparams["vini"]
        tempini = ICparams["tempini"]
        muini = ICparams["muini"]

        #check the dimension of the initial condition
        if ru.CheckParamDimen(ICparams) != True:
            raise Exception("%s unit not correctly set in params"%ru.CheckParamDimen(ICparams))
         
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.par.boxsize[0]/self.par.nogrid

        # generate initial condition
        self.mesh.boundary = np.linspace(-dx,self.par.boxsize[0]+dx,self.par.nogrid+3)
        coordinate = 0.5 * (self.mesh.boundary[1:]+self.mesh.boundary[:-1])
        #print('coordinate',coordinate)

        rho = np.ones(self.par.nogrid+2) * rhoini
        self.fluid.vel = np.ones(self.par.nogrid+2) * vini
        self.fluid.temp = np.ones(self.par.nogrid+2) * tempini
        rho[np.logical_or(coordinate<0.25*self.par.boxsize[0], coordinate>0.75*self.par.boxsize[0])] *= 0.5
        self.fluid.rho = rho
        # mean molecular weight
        self.fluid.mu = np.ones(self.par.nogrid+2) * ICparams["muini"] # for primordial neutral gas 


def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    x = np.linspace(0.0,ICparams["boxsize"],ICparams["nogrid"]+2) 
    rho = np.ones(ICparams["nogrid"]+2) * ICparams["rhoini"]
    x1 = 0.25*ICparams["boxsize"]+rout.par.time*ICparams["vini"]
    x2 = 0.75*ICparams["boxsize"]+rout.par.time*ICparams["vini"] 
    if x1>ICparams["boxsize"]:
        x1 -= ICparams["boxsize"] 
    if x2>ICparams["boxsize"]:
        x2 -= ICparams["boxsize"]
    if x2>x1:     
        rho[np.logical_or(x<x1, x>x2)] *= 0.5
    if x1>x2:
        rho[np.logical_and(x>x1, x<x2)] *= 0.5 
    plt.plot(x, rho, color=kwargs['color'],ls='solid')
    rplot1d(rout,showfig=0,**kwargs)



def main():
    ric = Simwrap()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(0,9,8):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        ReadandPlot(outfilename,ls='none',marker='o', mfc='none', markevery=10,color=next(ax._get_lines.prop_cycler)['color'])
    plt.show()    

if __name__ == "__main__":
    main()