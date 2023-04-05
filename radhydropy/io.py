import h5py
import os
import unyt
import numpy as np
import radhydropy.utils as ru


def WriteIC(datadict,ICdir):
    coordsys = datadict['Coordinate_System']
    nogrid = datadict['Number_Grids']
    time = datadict["Time"]
    boxsize = datadict["BoxSize"]

    rho = datadict["Density"]
    u = datadict["Velocity"]
    temp = datadict["Temperature"]
    mu = datadict["Mol_weight"]

    with h5py.File(ICdir, 'w') as fic:
        # saving initial condition
        # first, save header:
        header = fic.create_group("Header")
        header.attrs['Coordinate_System'] = coordsys
        header.attrs['Number_Grids'] = nogrid
        header.create_dataset("Time",time)
        header["Time"].attrs['units'] = str(time.units)
        header.create_dataset("BoxSize",boxsize)
        header["BoxSize"].attrs['units'] = str(boxsize.units)   

        #second, save mesh and fluid data:
        gdata = fic.create_group("Data")
        gdata.create_dataset("Density", data=rho)
        gdata["Density"].attrs['units'] = str(rho.units)
        gdata.create_dataset("Velocity", data=u)
        gdata["Velocity"].attrs['units'] = str(u.units)   
        gdata.create_dataset("Temperature", data=temp)
        gdata["Temperature"].attrs['units'] = str(temp.units) 
        gdata.create_dataset("Mol_weight", data=mu)