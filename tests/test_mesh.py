import unittest
from radhydropy.mesh import Mesh
from radhydropy.units import CodeUnits
import numpy as np
import unyt

class Par():
    def __init__(self):
        self.CodeUnits = CodeUnits.from_mapping(
            {
                'name': 'test_units',
                'InternalUnitSystem': {
                    'UnitMass_in_cgs': 1.0,
                    'UnitLength_in_cgs': 1.0,
                    'UnitVelocity_in_cgs': 1.0,
                    'UnitCurrent_in_cgs': 1.0,
                    'UnitTemp_in_cgs': 1.0,
                },
            }
        )
        self.nogrid = 10
        self.noghost = 2
        self.coordsys = 'cartesian'
        self.area = 3.0 * unyt.cm**2

class Testing(unittest.TestCase):
    def setUp(self):
        self.par = Par()
        self.mesh = Mesh()
        self.mesh.boundary = np.linspace(1, 10, num=self.par.nogrid + 1) * unyt.cm

    def test_SetUpMesh(self):
        self.mesh.SetUpMesh(self.par)
        self.assertEqual(len(self.mesh.vol), self.par.nogrid + 2 * self.par.noghost)
        self.assertEqual(len(self.mesh.boundary), self.par.nogrid + 1 + 2 * self.par.noghost)
        np.testing.assert_allclose(self.mesh.vol, np.full(len(self.mesh.vol), 2.7))
        self.assertFalse(hasattr(self.mesh.boundary, 'units'))

    def test_unknown_coordinate_system_raises(self):
        self.par.coordsys = 'cylindrical'
        with self.assertRaises(ValueError):
            self.mesh.SetUpMesh(self.par)

    def test_spherical_origin_face_has_zero_area(self):
        self.par.coordsys = 'spherical'
        self.mesh.boundary = np.linspace(-0.5, 1.5, num=self.par.nogrid + 1)

        self.mesh.SetUpMesh(self.par)

        origin_cell = np.where(
            np.logical_and(self.mesh.boundary[:-1] < 0.0,
                           self.mesh.boundary[1:] > 0.0)
        )[0][0]
        self.assertEqual(self.mesh.area[origin_cell], 0.0)

    def test_spherical_zero_inner_boundary_has_zero_area(self):
        self.par.coordsys = 'spherical'
        self.mesh.boundary = np.linspace(0.0, 1.0, num=self.par.nogrid + 1)

        self.mesh.SetUpMesh(self.par)

        self.assertEqual(self.mesh.boundary[self.par.noghost], 0.0)
        self.assertEqual(self.mesh.area[self.par.noghost], 0.0)
        self.assertAlmostEqual(
            float(self.mesh.coordinate[self.par.noghost]),
            0.75 * float(self.mesh.boundary[self.par.noghost+1] - self.mesh.boundary[self.par.noghost]),
        )
