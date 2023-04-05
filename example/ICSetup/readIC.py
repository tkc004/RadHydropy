import h5py
import os
import unyt
import numpy as np
import radhydropy.io as rio


class Ip():
    def __init__(self) -> None:
        pass

class Mesh():
    def __init__(self) -> None:
        pass

class Fluid():
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    ip = Ip()
    mesh = Mesh()
    fluid = Fluid()
    rio.readhdf5(ip,mesh,fluid,"InitialCondition.hdf5")
    print("temp", fluid.temp)