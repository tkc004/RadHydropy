import h5py
import os
import unyt
import numpy as np
import radhydropy.utils as ru
import radhydropy.io as rio


# should be read from parameter files instead
nogrid = 1000
coordsys = "cartesian"
boxsize = np.array([1.0])*unyt.pc
time = np.array([0.0])*unyt.Myr
rhoini = 1.0 * unyt.mp/ unyt.cm**3
uini = 1.0 * unyt.km/unyt.s
tempini = 0.1 * unyt.K

params = {"boxsize":boxsize, "time":time, "rhoini":rhoini, "uini":uini, "tempini":tempini}
ru.CheckParamDimen(params)
# boundary points of the mesh
# note that we use first (0) and final (nogrid+1) cells as ghost cells
# to set boundary conditions
dx = boxsize[0]/nogrid

# generate initial condition
boundary = np.linspace(-dx,boxsize[0]+dx,nogrid+3)
coordinate = 0.5 * (boundary[1:]+boundary[:-1])
#print('coordinate',coordinate)

rho = np.ones(nogrid+2) * rhoini
u = np.ones(nogrid+2) * uini
temp = np.ones(nogrid+2) * tempini
rho[np.logical_or(coordinate<0.25*boxsize[0], coordinate>0.75*boxsize[0])] *= 0.5
# mean molecular weight
mu = np.ones(nogrid+2) * 1.28 # for primordial neutral gas 

datadict = {}
datadict['Coordinate_System'] = coordsys
datadict['Number_Grids'] = nogrid 
datadict["Time"] = time 
datadict["BoxSize"] = boxsize 

datadict["Density"] = rho
datadict["Velocity"] = u
datadict["Temperature"] = temp
datadict["Mol_weight"] = mu


rio.WriteIC(datadict,"InitialCondition.hdf5")


