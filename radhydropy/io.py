"""HDF5 input and output helpers for simulations."""

import h5py
import os
import unyt
import numpy as np
import radhydropy.utils as ru

def writehdf5(ric,ICfilename):
    """Write simulation state to a RadHydropy HDF5 file.

    The output file contains a ``Header`` group for metadata and a ``Data``
    group for mesh and fluid arrays. Units are stored as HDF5 attributes.
    """
    print("--- writing "+ICfilename+" --- ")
    with h5py.File(ICfilename, 'w') as fic:
        # saving initial condition
        # first, save header:
        header = fic.create_group("Header")
        header.attrs['Coordinate_System'] = ric.par.coordsys
        header.attrs['Number_Grids'] = ric.par.nogrid
        header.create_dataset("Time", data=ric.par.time)
        header["Time"].attrs['units'] = str(ric.par.time.units)
        header.create_dataset("BoxSize", data=ric.par.boxsize)
        header["BoxSize"].attrs['units'] = str(ric.par.boxsize.units)   

        #second, save mesh and fluid data:
        gdata = fic.create_group("Data")
        gdata.create_dataset("Boundary", data=ric.mesh.boundary)
        gdata["Boundary"].attrs['units'] = str(ric.mesh.boundary.units)        
        gdata.create_dataset("Density", data=ric.fluid.rho)
        gdata["Density"].attrs['units'] = str(ric.fluid.rho.units)
        gdata.create_dataset("Velocity", data=ric.fluid.vel)
        gdata["Velocity"].attrs['units'] = str(ric.fluid.vel.units)   
        gdata.create_dataset("Temperature", data=ric.fluid.temp)
        gdata["Temperature"].attrs['units'] = str(ric.fluid.temp.units) 
        gdata.create_dataset("Mol_weight", data=ric.fluid.mu)
        if hasattr(ric.fluid, "xHI"):
            gdata.create_dataset("NeutralFraction", data=ric.fluid.xHI)
        if hasattr(ric.fluid, "ngamma"):
            gdata.create_dataset("PhotonNumberDensity", data=ric.fluid.ngamma)
            gdata["PhotonNumberDensity"].attrs['units'] = str(ric.fluid.ngamma.units)



def readhdf5(par, mesh, fluid, ICfilename): 
    """Read a RadHydropy HDF5 file into parameter, mesh, and fluid objects."""
    print("--- reading "+ICfilename+" --- ")
    with h5py.File(ICfilename, 'r') as fic:
        # saving initial condition
        # first, save header:
        header = fic["Header"]
        coordsys = header.attrs['Coordinate_System']
        if hasattr(par, "coordsys"): 
            if coordsys != par.coordsys:
                raise Exception("Coordinate systems in IC (%s) and run (%s) do not agree!"%(coordsys,par.coordsys))
        else:
            par.coordsys = coordsys
        par.nogrid = header.attrs['Number_Grids']
        time = header["Time"][:] 
        time_unit = header["Time"].attrs['units'] 
        par.time = time * unyt.Unit(time_unit)
        boxsize = header["BoxSize"][:]
        boxsize_unit = header["BoxSize"].attrs['units']
        par.boxsize = boxsize * unyt.Unit(boxsize_unit) 

        #second, save mesh and fluid data:
        gdata = fic["Data"]
        boundary = gdata["Boundary"][:]
        boundary_unit = gdata["Boundary"].attrs['units']   
        mesh.boundary = boundary * unyt.Unit(boundary_unit) 
        rho = gdata["Density"][:]
        rho_unit = gdata["Density"].attrs['units']   
        fluid.rho = rho * unyt.Unit(rho_unit)  
        vel = gdata["Velocity"][:] 
        vel_unit = gdata["Velocity"].attrs['units']   
        fluid.vel = vel * unyt.Unit(vel_unit)
        temp = gdata["Temperature"][:] 
        temp_unit = gdata["Temperature"].attrs['units'] 
        temp_unit = gdata["Temperature"].attrs['units']   
        fluid.temp = temp * unyt.Unit(temp_unit)      
        fluid.mu = gdata["Mol_weight"][:]  
        if "NeutralFraction" in gdata:
            fluid.xHI = gdata["NeutralFraction"][:]
        if "PhotonNumberDensity" in gdata:
            ngamma = gdata["PhotonNumberDensity"][:]
            ngamma_unit = gdata["PhotonNumberDensity"].attrs['units']
            fluid.ngamma = ngamma * unyt.Unit(ngamma_unit)
