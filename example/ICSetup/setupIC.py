import h5py
import os
import unyt
import numpy as np
import radhydropy.utils as ru
import radhydropy.io as rio

class InitialCondition():
    def __init__(self) -> None:
        # should be read from parameter files instead
        self.nogrid = 1000
        self.coordsys = "cartesian"
        self.boxsize = 1.0*np.ones(1)*unyt.pc
        self.time = np.array([0.0])*unyt.Myr
        rhoini = 1.0 * unyt.mp/ unyt.cm**3
        vini = 1.0 * unyt.km/unyt.s
        tempini = 0.1 * unyt.K

        #check the dimension of the initial condition
        params = {"boxsize":self.boxsize, "time":self.time, "rhoini":rhoini, "vini":vini, "tempini":tempini}
        if ru.CheckParamDimen(params) != True:
            raise Exception("%s unit not correctly set in params"%ru.CheckParamDimen(params))
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.boxsize[0]/self.nogrid

        # generate initial condition
        self.boundary = np.linspace(-dx,self.boxsize[0]+dx,self.nogrid+3)
        coordinate = 0.5 * (self.boundary[1:]+self.boundary[:-1])
        #print('coordinate',coordinate)

        rho = np.ones(self.nogrid+2) * rhoini
        self.vel = np.ones(self.nogrid+2) * vini
        self.temp = np.ones(self.nogrid+2) * tempini
        rho[np.logical_or(coordinate<0.25*self.boxsize[0], coordinate>0.75*self.boxsize[0])] *= 0.5
        self.rho = rho
        # mean molecular weight
        self.mu = np.ones(self.nogrid+2) * 1.28 # for primordial neutral gas 

if __name__ == "__main__":
    ric = InitialCondition()
    rio.writehdf5(ric,"InitialCondition.hdf5")


