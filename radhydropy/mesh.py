import numpy as np
import unyt

# set up the underlying mesh for fluid
class Mesh:
    def __init__(self, boxsize: float, nogrid: int, coordinate: str, area = 1.0 * unyt.pc**2):
        self.nogrid = nogrid
        if nogrid < 1:
            print('nogrid has to be bigger than 1')
            exit()
        self.coordinate = coordinate
        self.boxsize = boxsize
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = boxsize/nogrid
        self.xbound = np.linspace(-dx,boxsize+dx,nogrid+3)
        # mesh coordinate is the midpoint of xbound
        self.xmesh = 0.5 * (self.xbound[1:]+self.xbound[:-1])
        # mesh size
        self.xdelta = self.xbound[1:] - self.xbound[:-1]
        self.oneoverdx = 1.0/self.xdelta
        if coordinate == 'cartesian':
            self.area = np.ones(nogrid+2) * area
            self.vol = (self.xbound[1:] - self.xbound[:-1]) * self.area
        elif coordinate == 'spherical':
            self.area = (self.xbound[1:]**2 - self.xbound[:-1]**2)*4.0*np.pi
            self.vol = (self.xbound[1:]**3 - self.xbound[:-1]**3)*4.0*np.pi/3.0
        else:
            print('coordinate unknown')
            
        if np.any(self.vol == 0.0) or np.any(np.isnan(self.vol)):
            print('vol vanished')
            exit()

            