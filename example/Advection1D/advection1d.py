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
    'outdeltatime':1.0*unyt.Myr *0.1,
    'savedir':rundir,
    'coordsys':'cartesian', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':1.0*unyt.Myr, # final simulation time
    'ngrid':1000, # number of grid to discretize
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'Periodic',
    'verbose':0, # speak out details?
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
        self.par.nogrid = 1000
        self.par.coordsys = "cartesian"
        self.par.boxsize = 1.0*np.ones(1)*unyt.pc
        self.par.time = np.array([0.0])*unyt.Myr
        rhoini = 1.0 * unyt.mp/ unyt.cm**3
        vini = 1.0 * unyt.km/unyt.s
        tempini = 0.1 * unyt.K

        #check the dimension of the initial condition
        params = {"boxsize":self.par.boxsize, "time":self.par.time, "rhoini":rhoini, "vini":vini, "tempini":tempini}
        ru.CheckParamDimen(params)
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
        self.fluid.mu = np.ones(self.par.nogrid+2) * 1.28 # for primordial neutral gas 


def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout,showfig=0,**kwargs)



def main():
    ric = Simwrap()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll()
    for outindex in range(0,9,2):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        ReadandPlot(outfilename,ls='none',marker='o')
    plt.show()    

if __name__ == "__main__":
    main()