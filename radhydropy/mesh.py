"""Mesh construction utilities for one-dimensional simulations."""

import numpy as np
from radhydropy.units import _code_units, _to_code_quantity, quantity_to_value
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
        if code_units is None:
            raise ValueError("SetUpMesh requires configured code units")
        nogrid = par.mesh.grid_cells
        noghost = par.mesh.ghost_cells
        self.CodeUnits = code_units
        self.coordsys = par.simulation.coordinate_system
        attr = 'boundary'
        if not hasattr(self, attr):
            raise AttributeError("%s does not exist in mesh; quitting."%attr)
        for attr in ('nogrid', 'noghost', 'coordsys'):
            if not hasattr(par, attr):
                raise AttributeError("%s does not exist in params; quitting."%attr)
        if nogrid < 1:
            raise ValueError("nogrid has to be at least 1")
        if noghost < 1:
            raise ValueError("noghost has to be at least 1")
        if len(self.boundary) != nogrid + 1:
            raise ValueError("boundary point and nogrid are inconsistent")
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions

        # add ghost cells:
        self.boundary = as_named_array(quantity_to_value(self.boundary, code_units.length_unit))
        dx = self.boundary[1] - self.boundary[0] 
        start = self.boundary[0] - dx * noghost
        end = self.boundary[-1] + dx * noghost 
        ghost_front = np.linspace(start, self.boundary[0]-dx, noghost)
        ghost_back  = np.linspace(self.boundary[-1]+dx, end, noghost)
        self.boundary = np.concatenate((ghost_front,self.boundary,ghost_back))

        # mesh size
        self.xdelta = as_named_array(self.boundary[1:] - self.boundary[:-1])
        self.oneoverdx = as_named_array(1.0/self.xdelta)
        if self.coordsys == 'cartesian':
            if not hasattr(par, 'area'):
                raise AttributeError("area does not exist in params; quitting.")
            # coordinate is the midpoint of boundary
            self.coordinate = as_named_array(0.5 * (self.boundary[1:]+self.boundary[:-1]))
            area_value = quantity_to_value(
                par.mesh.area,
                code_units.area_unit,
            )
            self.area = as_named_array(
                np.ones(nogrid + noghost * 2, dtype=float) * np.asarray(area_value, dtype=float)
            )
            self.vol = as_named_array((self.boundary[1:] - self.boundary[:-1]) * self.area)
        elif self.coordsys == 'spherical':
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
            raise ValueError("coordinate system unknown: %s" % self.coordsys)
            
        if np.any(self.vol == 0.0) or np.any(np.isnan(self.vol)):
            raise ValueError("volume vanished") 

            
