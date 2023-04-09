import unittest
import radhydropy.utils as ru
import unyt

class Testing(unittest.TestCase):
    def test_calpressure(self):
        rho = 1.0 * unyt.mp / unyt.cm**3
        mu = 1.0
        temp = 1.0 * unyt.K
        precal = ru.CalPressure(rho,temp,mu)
        preex = rho / (mu * unyt.mp) * unyt.kb * temp
        self.assertEqual(precal, preex)

    def test_CheckParamDimen(self):
        params = {'vini': 1.0*unyt.s}
        testdim = ru.CheckParamDimen(params)
        self.assertEqual(testdim ,False)
        params = {'vini': 1.0*unyt.cm/unyt.s}
        testdim = ru.CheckParamDimen(params) 
        self.assertEqual(testdim ,True)
    
    


if __name__ == '__main__':
    unittest.main()