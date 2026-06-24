import unittest
from radhydropy.mesh import Mesh
import unyt
import numpy as np

class Par():
    def __init__(self):
        self.nogrid = 10
        self.noghost = 2
        self.coordsys = 'cartesian'
        self.area = 3.0 * unyt.cm**2

class Testing(unittest.TestCase):
    def setUp(self):
        self.par = Par()
        self.mesh = Mesh()
        self.mesh.boundary = np.linspace(1,10,num=self.par.nogrid+1)*unyt.cm

    def test_SetUpMesh(self):
        self.mesh.SetUpMesh(self.par)
        self.assertEqual(len(self.mesh.vol), self.par.nogrid + 2 * self.par.noghost)
        self.assertEqual(len(self.mesh.boundary), self.par.nogrid + 1 + 2 * self.par.noghost)
        self.assertEqual(self.mesh.vol.units, unyt.cm**3)

    def test_unknown_coordinate_system_raises(self):
        self.par.coordsys = 'cylindrical'
        with self.assertRaises(ValueError):
            self.mesh.SetUpMesh(self.par)
