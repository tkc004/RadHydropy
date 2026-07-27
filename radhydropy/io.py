"""HDF5 input and output helpers for simulations."""

import h5py
import os
import unyt
import numpy as np
import radhydropy.utils as ru


def _read_quantity(group, name):
    dataset = group[name]
    return np.asarray(dataset[()]) * unyt.Unit(dataset.attrs['units'])


def _read_dataset(group, name):
    return group[name][()]

def writehdf5(ric,ICfilename):
    """Write simulation state to a RadHydropy HDF5 file.

    The output file contains a ``Header`` group for metadata and a ``Data``
    group for mesh and fluid arrays. Units are stored as HDF5 attributes.
    """
    ICfilename = str(ICfilename)
    print(f"--- writing {ICfilename} --- ")
    if hasattr(ric.fluid, "time"):
        output_time = ric.fluid.time
        if hasattr(ric.par, "time"):
            ric.par.time = output_time.copy()
    else:
        output_time = ric.par.time
    with h5py.File(ICfilename, 'w') as fic:
        # saving initial condition
        # first, save header:
        header = fic.create_group("Header")
        header.attrs['Coordinate_System'] = ric.par.coordsys
        header.attrs['Number_Grids'] = ric.par.nogrid
        header.create_dataset("Time", data=output_time)
        header["Time"].attrs['units'] = str(output_time.units)
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
    ICfilename = str(ICfilename)
    print(f"--- reading {ICfilename} --- ")
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
        par.time = _read_quantity(header, "Time")
        fluid.time = par.time.copy()
        par.boxsize = _read_quantity(header, "BoxSize")

        #second, save mesh and fluid data:
        gdata = fic["Data"]
        mesh.boundary = _read_quantity(gdata, "Boundary")
        fluid.rho = _read_quantity(gdata, "Density")
        fluid.vel = _read_quantity(gdata, "Velocity")
        fluid.temp = _read_quantity(gdata, "Temperature")
        fluid.mu = _read_dataset(gdata, "Mol_weight")
        if "NeutralFraction" in gdata:
            fluid.xHI = _read_dataset(gdata, "NeutralFraction")
        if "PhotonNumberDensity" in gdata:
            fluid.ngamma = _read_quantity(gdata, "PhotonNumberDensity")
