import h5py
import os
import unyt
import numpy as np

with h5py.File('InitialCondition.hdf5', 'r') as fic:
    # saving initial condition
    # first, save header:
    header = fic["Header"]
    coordsys = header.attrs['Coordinate_System']
    nogrid = header.attrs['Number_Grids']
    time = header["Time"][:] 
    time_unit = header["Time"].attrs['units'] 
    time = time * unyt.Unit(time_unit)
    boxsize = header["BoxSize"][:]
    boxsize_unit = header["BoxSize"].attrs['units']
    boxsize = boxsize * unyt.Unit(boxsize_unit) 

    #second, save mesh and fluid data:
    gdata = fic["Data"]
    rho = gdata["Density"][:]
    rho_unit = gdata["Density"].attrs['units']   
    rho = rho * unyt.Unit(rho_unit)  
    u = gdata["Velocity"][:] 
    u_unit = gdata["Velocity"].attrs['units']   
    u = u * unyt.Unit(u_unit)
    temp = gdata["Temperature"][:] 
    temp_unit = gdata["Temperature"].attrs['units'] 
    temp_unit = gdata["Temperature"].attrs['units']   
    temp = temp * unyt.Unit(temp_unit)      
    mu = gdata["Mol_weight"][:]  

print("temp", temp)