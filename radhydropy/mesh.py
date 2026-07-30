"""Mesh construction utilities for one-dimensional simulations."""

import numpy as np
import unyt
from radhydropy.units import _code_units, _to_code_quantity, to_code_value
from radhydropy.arrays import as_named_array


# set up the underlying mesh for fluid
class Mesh:
    """Store cell faces, cell centers, areas, and volumes.

    A ``Mesh`` instance expects its ``boundary`` attribute to be populated with
    physical cell-face locations before :meth:`SetUpMesh` is called.
    """

    def __init__(self):
        pass

    def SetUpMesh(self, par):
        """Build ghost cells and geometric factors from run parameters.

        Parameters
        ----------
        par : object
            Parameter object with ``nogrid``, ``noghost``, and ``coordsys``.
            Cartesian meshes also require ``area``.

        Raises
        ------
        AttributeError
            If required mesh or parameter attributes are missing.
        ValueError
            If the mesh dimensions or coordinate system are invalid.
        """
        code_units = _code_units(par)
        self.code_units = code_units
        self.coordsys = par.coordsys
        attr = 'boundary'
        if not hasattr(self, attr):
            raise AttributeError("%s does not exist in mesh; quitting."%attr)
        for attr in ('nogrid', 'noghost', 'coordsys'):
            if not hasattr(par, attr):
                raise AttributeError("%s does not exist in params; quitting."%attr)
        if par.nogrid < 1:
            raise ValueError("nogrid has to be at least 1")
        if par.noghost < 1:
            raise ValueError("noghost has to be at least 1")
        if len(self.boundary) != par.nogrid + 1:
            raise ValueError("boundary point and nogrid are inconsistent")
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions

        # add ghost cells:
        noghost = par.noghost
        if code_units is not None:
            self.boundary = as_named_array(to_code_value(self.boundary, code_units.length_unit))
        dx = self.boundary[1] - self.boundary[0] 
        start = self.boundary[0] - dx * noghost
        end = self.boundary[-1] + dx * noghost 
        ghost_front = np.linspace(start, self.boundary[0]-dx, noghost)
        ghost_back  = np.linspace(self.boundary[-1]+dx, end, noghost)
        self.boundary = np.concatenate((ghost_front,self.boundary,ghost_back))

        # mesh size
        self.xdelta = as_named_array(self.boundary[1:] - self.boundary[:-1])
        self.oneoverdx = as_named_array(1.0/self.xdelta)
        if par.coordsys == 'cartesian':
            if not hasattr(par, 'area'):
                raise AttributeError("area does not exist in params; quitting.")
            # coordinate is the midpoint of boundary
            self.coordinate = as_named_array(0.5 * (self.boundary[1:]+self.boundary[:-1]))
            if code_units is not None:
                area_value = to_code_value(par.area, code_units.area_unit)
            else:
                area_value = _to_code_quantity(par.area, getattr(par.area, 'units', 1.0))
            self.area = as_named_array(np.ones(par.nogrid+noghost*2, dtype=float) * np.asarray(area_value, dtype=float))
            self.vol = as_named_array((self.boundary[1:] - self.boundary[:-1]) * self.area)
        elif par.coordsys == 'spherical':
            # check if any value is <0:
            #if len(self.boundary[self.boundary<0.0]) > 0:
            #    raise Exception("Radial coordinate cannot be negative")
            # coordinate is the centroid of the volume (center of gravity?):
            # see Mignone+14
            #area to the left
            self.area = as_named_array((self.boundary[:-1]**2)*4.0*np.pi)
            #cell volume
            self.vol = as_named_array(np.absolute((self.boundary[1:]**3 - self.boundary[:-1]**3))*4.0*np.pi/3.0)
            vol_denom = self.boundary[1:]**3 - self.boundary[:-1]**3
            self.coordinate = as_named_array(0.5 * (self.boundary[1:] + self.boundary[:-1]))
            nonzero_vol_denom = vol_denom != 0.0
            self.coordinate[nonzero_vol_denom] = 0.75 * (
                self.boundary[1:][nonzero_vol_denom]**4
                - self.boundary[:-1][nonzero_vol_denom]**4
            ) / vol_denom[nonzero_vol_denom]
            for ig in range(len(self.vol)):
                # This is the inner sphere
                if ((self.boundary[ig] < 0.0) and (self.boundary[ig+1] > 0.0)):
                    self.vol[ig] = (self.boundary[ig+1]**3)*4.0*np.pi/3.0
                    self.coordinate[ig] = 0.75 * self.boundary[ig+1]
                    self.area[ig] = 0.0
                    

        else:
            raise ValueError("coordsys unknown: %s"%par.coordsys)
            
        if np.any(self.vol == 0.0) or np.any(np.isnan(self.vol)):
            raise ValueError("volume vanished") 

            
