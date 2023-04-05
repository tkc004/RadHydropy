import numpy as np
import unyt

# set up the underlying mesh for fluid
class Mesh:
    def __init__(self, boxsize: float, nogrid: int, coordsys: str, area = 1.0 * unyt.pc**2):
        self.nogrid = nogrid
        if nogrid < 1:
            print('nogrid has to be bigger than 1')
            exit()
        self.coordsys = coordsys
        self.boxsize = boxsize
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = boxsize/nogrid
        self.boundary = np.linspace(-dx,boxsize+dx,nogrid+3)
        # mesh coordsys is the midpoint of boundary
        self.coordinate = 0.5 * (self.boundary[1:]+self.boundary[:-1])
        # mesh size
        self.xdelta = self.boundary[1:] - self.boundary[:-1]
        self.oneoverdx = 1.0/self.xdelta
        if coordsys == 'cartesian':
            self.area = np.ones(nogrid+2) * area
            self.vol = (self.boundary[1:] - self.boundary[:-1]) * self.area
        elif coordsys == 'spherical':
            self.area = (self.boundary[1:]**2 - self.boundary[:-1]**2)*4.0*np.pi
            self.vol = (self.boundary[1:]**3 - self.boundary[:-1]**3)*4.0*np.pi/3.0
        else:
            print('coordsys unknown')
            
        if np.any(self.vol == 0.0) or np.any(np.isnan(self.vol)):
            print('vol vanished')
            exit()

            