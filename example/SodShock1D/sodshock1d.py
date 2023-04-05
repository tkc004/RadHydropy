from radhydropy.rsim import Rsim
import unyt
from radhydropy.analysis import rplot1d
import numpy as np
import radhydropy.utils as ru 
import radhydropy.io as rio
#importing the os module
import os

#to get the current working directory
rundir = os.getcwd()
print('rundir',rundir)


runparams = {
    'simname':'SodShock1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outfilename':rundir+'/Output.hdf5', 
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


class Par():
    def __init__(self) -> None:
        pass

class Mesh():
    def __init__(self) -> None:
        pass

class Fluid():
    def __init__(self) -> None:
        pass

class InitialCondition():
    def __init__(self) -> None:
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        # should be read from parameter files instead
        self.par.nogrid = 1000
        self.par.coordsys = "cartesian"
        self.par.boxsize = 1.0*np.ones(1)*unyt.cm
        self.par.time = np.array([0.0])*unyt.s
        rhoini = 1.0 * unyt.g/ unyt.cm**3
        uini = 0.0 * unyt.km/unyt.s
        tempini = 1.0 * unyt.g / unyt.cm / unyt.s**2 * (1.28 * unyt.mp) / unyt.kb / (1.0 * unyt.g/unyt.cm**3)

        #check the dimension of the initial condition
        params = {"boxsize":self.par.boxsize, "time":self.par.time, "rhoini":rhoini, "uini":uini, "tempini":tempini}
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
        self.fluid.u = np.ones(self.par.nogrid+2) * uini
        rho[coordinate>0.5*self.par.boxsize[0]] *= 0.1
        self.fluid.rho = rho
        temp = np.ones(self.par.nogrid+2) * tempini
        temp[coordinate>0.5*self.par.boxsize[0]] *= 0.1/0.125
        self.fluid.temp = temp
        # mean molecular weight
        self.fluid.mu = np.ones(self.par.nogrid+2) * 1.28 # for primordial neutral gas 


def main():
    ric = InitialCondition()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll()
    rio.writehdf5(mainrun,runparams['outfilename'])
    rplot1d(mainrun,showfig=1)

if __name__ == "__main__":
    main()