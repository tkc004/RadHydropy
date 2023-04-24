import numpy as np
import unyt

# set up the underlying mesh for fluid
class Mesh:
    def __init__(self):
        pass

    def SetUpMesh(self, par):
        attr = 'boundary'
        if not hasattr(self, attr):
            raise Exception("%s does not exist in mesh; quitting."%attr)
        if par.nogrid < 1:
            raise Exception("nogrid has to be bigger than 1")
        if len(self.boundary) != par.nogrid + 1:
            raise Exception("boundary point and nogrid are inconsistent")
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions

        # add ghost cells:
        noghost = par.noghost
        dx = self.boundary[1] - self.boundary[0] 
        start = self.boundary[0] - dx * noghost
        end = self.boundary[-1] + dx * noghost 
        ghost_front = np.linspace(start, self.boundary[0]-dx, noghost)
        ghost_back  = np.linspace(self.boundary[-1]+dx, end, noghost)
        self.boundary = unyt.uconcatenate((ghost_front,self.boundary,ghost_back))

        # mesh size
        self.xdelta = self.boundary[1:] - self.boundary[:-1]
        self.oneoverdx = 1.0/self.xdelta
        if par.coordsys == 'cartesian':
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
            ri = 0.5 * (self.boundary[1:]+self.boundary[:-1]) 
            #dri = 2.0*ri*self.xdelta**2 / ( 12.0 * ri**2 + self.xdelta**2)
            #self.coordinate = ri + dri
            self.coordinate = ri 
            #area to the left
            self.area = (self.boundary[:-1]**2)*4.0*np.pi
            #cell volume
            self.vol = np.absolute((self.boundary[1:]**3 - self.boundary[:-1]**3))*4.0*np.pi/3.0
            for ig in range(par.nogrid):
                # This is the inner sphere
                if ((self.boundary[ig].value < 0.0) and (self.boundary[ig+1].value > 0.0)):
                    self.vol[ig] = (self.boundary[ig+1]**3)*4.0*np.pi/3.0
                    

        else:
            print('coordsys unknown')
            
        if np.any(self.vol == 0.0) or np.any(np.isnan(self.vol)):
            raise Exception("volume vanished") 

            