import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import unyt

import radhydropy.io as rio


class Testing(unittest.TestCase):
    def test_hdf5_roundtrip_preserves_neutral_fraction_when_present(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho=np.ones(3) * unyt.g/unyt.cm**3,
            vel=np.zeros(3) * unyt.cm/unyt.s,
            temp=np.ones(3) * unyt.K,
            mu=np.ones(3),
            xHI=np.array([1.0, 0.5, 0.0]),
        )
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = SimpleNamespace(coordsys='cartesian')
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        np.testing.assert_array_equal(loaded_fluid.xHI, fluid.xHI)


if __name__ == '__main__':
    unittest.main()
