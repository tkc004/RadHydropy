"""Gravity configuration placeholder."""

import numpy as np
import unyt
import radhydropy.utils as ru
from radhydropy.mesh import Mesh

# set up gravity properties

class Gravity():
    """Store flags for self-gravity and external gravity support."""

    def __init__(self, selfgravity=0, externalgravity=0):
        """Initialize gravity flags."""
        self.selfgravity = selfgravity
        self.externalgravity = externalgravity
    
    

