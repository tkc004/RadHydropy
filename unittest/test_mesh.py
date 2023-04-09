import unittest
from radhydropy.mesh import Mesh
import unyt
import numpy as np

class Par():
    def __init__(self):
        Par.nogrid = 10
        Par.coordsys = 'cartesian'

class Testing(unittest.TestCase):
    def setUp(self):
        self.par = Par()
        self.mesh = Mesh()
        self.mesh.boundary = np.linspace(1,10,num=self.par.nogrid+3)*unyt.cm

    def test_SetUpMesh(self):
        self.mesh.SetUpMesh(self.par)
        self.assertEqual(len(self.mesh.vol),self.par.nogrid+2)

