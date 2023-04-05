import numpy as np
import unyt

# set up the underlying mesh for fluid
class Mesh:
    def __init__(self):
        pass

    def SetUpMesh(self, par, area = 1.0 * unyt.pc**2):

        if par.nogrid < 1:
            print('nogrid has to be bigger than 1')
            exit()
        if len(self.boundary) != par.nogrid + 3:
            print('boundary point and nogrid are inconsistent')
            exit() 
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        # mesh coordsys is the midpoint of boundary
        self.coordinate = 0.5 * (self.boundary[1:]+self.boundary[:-1])
        # mesh size
        self.xdelta = self.boundary[1:] - self.boundary[:-1]
        self.oneoverdx = 1.0/self.xdelta
        if par.coordsys == 'cartesian':
            self.area = np.ones(par.nogrid+2) * area
            self.vol = (self.boundary[1:] - self.boundary[:-1]) * self.area
        elif par.coordsys == 'spherical':
            self.area = (self.boundary[1:]**2 - self.boundary[:-1]**2)*4.0*np.pi
            self.vol = (self.boundary[1:]**3 - self.boundary[:-1]**3)*4.0*np.pi/3.0
        else:
            print('coordsys unknown')
            
        if np.any(self.vol == 0.0) or np.any(np.isnan(self.vol)):
            print('vol vanished')
            exit()

            