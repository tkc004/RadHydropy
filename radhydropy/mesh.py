import numpy as np
import unyt

# set up the underlying mesh for fluid
class Mesh:
    def __init__(self):
        pass

    def SetUpMesh(self, par):
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
        dx = self.boundary[1] - self.boundary[0] 
        start = self.boundary[0] - dx * noghost
        end = self.boundary[-1] + dx * noghost 
        ghost_front = np.linspace(start, self.boundary[0]-dx, noghost)
        ghost_back  = np.linspace(self.boundary[-1]+dx, end, noghost)
        self.boundary = np.concatenate((ghost_front,self.boundary,ghost_back))

        # mesh size
        self.xdelta = self.boundary[1:] - self.boundary[:-1]
        self.oneoverdx = 1.0/self.xdelta
        if par.coordsys == 'cartesian':
            if not hasattr(par, 'area'):
                raise AttributeError("area does not exist in params; quitting.")
            # coordinate is the midpoint of boundary
            self.coordinate = 0.5 * (self.boundary[1:]+self.boundary[:-1])
            self.area = np.ones(par.nogrid+noghost*2) * par.area
            self.vol = (self.boundary[1:] - self.boundary[:-1]) * self.area
        elif par.coordsys == 'spherical':
            # check if any value is <0:
            #if len(self.boundary[self.boundary<0.0]) > 0:
            #    raise Exception("Radial coordinate cannot be negative")
            # coordinate is the centroid of the volume (center of gravity?):
            # see Mignone+14
            #area to the left
            self.area = (self.boundary[:-1]**2)*4.0*np.pi
            #cell volume
            self.vol = np.absolute((self.boundary[1:]**3 - self.boundary[:-1]**3))*4.0*np.pi/3.0
            vol_denom = self.boundary[1:]**3 - self.boundary[:-1]**3
            self.coordinate = 0.5 * (self.boundary[1:] + self.boundary[:-1])
            nonzero_vol_denom = vol_denom != 0.0
            self.coordinate[nonzero_vol_denom] = 0.75 * (
                self.boundary[1:][nonzero_vol_denom]**4
                - self.boundary[:-1][nonzero_vol_denom]**4
            ) / vol_denom[nonzero_vol_denom]
            for ig in range(len(self.vol)):
                # This is the inner sphere
                if ((self.boundary[ig].value < 0.0) and (self.boundary[ig+1].value > 0.0)):
                    self.vol[ig] = (self.boundary[ig+1]**3)*4.0*np.pi/3.0
                    self.coordinate[ig] = 0.75 * self.boundary[ig+1]
                    self.area[ig] = 0.0 * self.area.units
                    

        else:
            raise ValueError("coordsys unknown: %s"%par.coordsys)
            
        if np.any(self.vol == 0.0) or np.any(np.isnan(self.vol)):
            raise ValueError("volume vanished") 

            
