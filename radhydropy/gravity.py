import numpy as np
import unyt
import radhydropy.utils as ru
from radhydropy.mesh import Mesh


# set up gravity properties

class Gravity():
    def __init__(self, selfgravity=0, externalgravity=0):
        self.selfgravity = selfgravity
        self.externalgravity = externalgravity
    
    



