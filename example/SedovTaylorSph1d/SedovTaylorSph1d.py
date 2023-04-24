from radhydropy.rsim import Rsim
import unyt
from radhydropy.analysis import rplot1d
import matplotlib.pyplot as plt
import numpy as np
import radhydropy.utils as ru 
import radhydropy.io as rio
#importing the os module
import os

#to get the current working directory
rundir = os.getcwd()
print('rundir',rundir)


runparams = {
    'simname':'SedovTaylorSph1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outdir':rundir,
    'outfileprefix':'Output', 
    'outdeltatime':1000.0*unyt.s *0.1,
    'savedir':rundir,
    'coordsys':'spherical', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':5./3., # for polytropic, the polytropic index
    'timesim':1000.0*unyt.s, # final simulation time
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'OpenSph',
    'noghost': 10, #number of ghost cells in front (equal number of ghost cell after)
    'verbose':0, # speak out details?
    'order': 0,
    'dtmin': 2.0e-8*unyt.s,
    'dtmax': 2.0e-1*unyt.s,   
}

ICparams = {
    'nogrid':1000, # number of grid points
    'coordsys':"spherical", # coordinate system
    'boxsize':2.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'rhoini': 1.0 * unyt.g/unyt.cm**3,
    'Eini': 1.0e-4 * unyt.erg, # the total energy in central region
    #'Eini': 0.0 * unyt.erg, # the total energy in central region
    'rini': 0.2 * unyt.cm, # radius of central region
    'muini': 1.0,
    "rinj": 0.1*unyt.cm, 
}


class Par():
    def __init__(self) -> None:
        pass

class Mesh():
    def __init__(self) -> None:
        pass

class Fluid():
    def __init__(self) -> None:
        pass

class Simwrap():
    def __init__(self) -> None:
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        # should be read from parameter files instead
        self.par.nogrid = ICparams["nogrid"]
        self.par.coordsys = ICparams["coordsys"]
        self.par.boxsize = ICparams["boxsize"]*np.ones(1)
        self.par.time = ICparams["time"]*np.ones(1)
        Eini = ICparams["Eini"]
         
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.par.boxsize[0]/self.par.nogrid

        # generate initial condition
        #self.mesh.boundary = np.linspace(0.5*dx,self.par.boxsize[0]+dx,self.par.nogrid+1)
        self.mesh.boundary = np.linspace(ICparams["rinj"],ICparams["rinj"]+self.par.boxsize[0],self.par.nogrid+1)
        self.mesh.coordinate = 0.5 * (self.mesh.boundary[:-1]+self.mesh.boundary[1:])
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm/unyt.s 
        self.fluid.rho =  ICparams["rhoini"] * np.ones(self.par.nogrid)
        self.mesh.vol = (self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3)*4.0*np.pi/3.0 
        # mean molecular weight
        self.fluid.mu = np.ones(self.par.nogrid) * ICparams["muini"] 
        self.fluid.mass = self.fluid.rho*self.mesh.vol 
        #inject Eini to a single particle in the center
        #self.fluid.temp = ICparams["Eini"] * ru.gaussiansph(self.mesh.coordinate, 0.1*unyt.cm) * unyt.mp * self.fluid.mu / (self.fluid.rho * 1.5 * unyt.kb) 
        self.fluid.temp = np.ones(self.par.nogrid) * 0.0 * unyt.K
        icut = np.logical_and(self.mesh.coordinate<ICparams['rini'],self.mesh.coordinate>ICparams['rinj'])
        self.fluid.temp[icut] = ICparams["Eini"] / np.sum(self.mesh.vol[icut] * self.fluid.rho[icut]) * unyt.mp * self.fluid.mu[icut] / (1.5 * unyt.kb) 



# I think the analytic solution only work for constant density
# we cannot do density advection
# we cannot do temperature advection neither
# since temperature changes with density
def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout,yquan='rho', showfig=0,**kwargs)
    # The analytic solution is off by half energy? Why?
    Rst = 1.17 * np.power(0.5*ICparams['Eini']/ICparams['rhoini']*rout.par.time**2, 1./5.)
    plt.axvline(x=Rst,color=kwargs['color']) 



def main():
    ric = Simwrap()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(1,10):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        ReadandPlot(outfilename,ls='none',marker='o', mfc='none', markevery=1,color=next(ax._get_lines.prop_cycler)['color'])
    plt.show()    

if __name__ == "__main__":
    main()