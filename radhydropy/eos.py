import numpy as np
import unyt

# set up the equation of state
class EOS:
    def __init__(self,EOStype: str,gamma=5./3.):
        self.EOStype = EOStype
        self.gamma = gamma
        if gamma==1.0:
            raise Exception("gamma cannot be equal to 1")
        if ((self.EOStype != 'polytropic') and (self.EOStype != 'isothermal')):
            raise Exception("EOS not recognized: only polytropic or isothermal")