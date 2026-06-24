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
    'simname':'advection1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outdir':rundir,
    'outfileprefix':'Output', 
    'outdeltatime':1.0*unyt.s *0.1,
    'savedir':rundir,
    'coordsys':'spherical', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':1.0*unyt.s, # final simulation time
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'OpenSph',
    'verbose':0, # speak out details?
    'order': 0
}

ICparams = {
    'nogrid':1000, # number of grid points
    'coordsys':"spherical", # coordinate system
    'boxsize':4.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'tempini': 0.0 * unyt.K,
    'muini': 1.0,
    'vini': 1.0 * unyt.cm/unyt.s, #initial velocity
    'rhoini': 1.0 * unyt.g/ unyt.cm**3, # initial density (of the higher density end)
}

# a needs to have unit 1/L
# b unit L
def Qaussian(r, a, b):
    return np.exp(-np.power(a*(r - b), 2.))

# alpha unit 1/T
def Q_analytic(gindex,alpha,t,r, a, b):
    return np.exp(-(gindex+1.)*alpha*t)*Qaussian(r*np.exp(-alpha*t), a, b)

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
         
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.par.boxsize[0]/self.par.nogrid

        # generate initial condition
        self.mesh.boundary = np.linspace(dx,self.par.boxsize[0]+dx,self.par.nogrid+1)
        coordinate = 0.5 * (self.mesh.boundary[1:]+self.mesh.boundary[:-1])
        #print('coordinate',coordinate)
        self.fluid.vel = ICparams["vini"] * np.ones(self.par.nogrid)
        self.fluid.temp = ICparams["tempini"] * np.ones(self.par.nogrid)
        rho = ICparams["rhoini"] * np.ones(self.par.nogrid)
        rho[np.logical_or(coordinate<0.25*self.par.boxsize[0], coordinate>0.75*self.par.boxsize[0])] *= 0.01
        self.fluid.rho = rho
        # mean molecular weight
        self.fluid.mu = np.ones(self.par.nogrid) * ICparams["muini"] # for primordial neutral gas 

# I think the analytic solution only work for constant density
# we cannot do density advection
# we cannot do temperature advection neither
# since temperature changes with density
def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout,yquan='rho', showfig=0,**kwargs)
    rout.mesh.vol = np.absolute((rout.mesh.boundary[1:]**3 - rout.mesh.boundary[:-1]**3))*4.0*np.pi/3.0
    mtot = np.sum(rout.fluid.rho * rout.mesh.vol)
    print('mtot', mtot)
    x = np.linspace(0.0*ICparams["boxsize"],ICparams["boxsize"],ICparams["nogrid"]) 
    x1 = 0.25*ICparams["boxsize"]+rout.par.time*ICparams["vini"]
    x2 = 0.75*ICparams["boxsize"]+rout.par.time*ICparams["vini"]
    #rho = np.ones(ICparams["nogrid"]) * ICparams["rhoini"]
    rho = ICparams["rhoini"] * np.ones(ICparams["nogrid"])
    if x1>ICparams["boxsize"]:
        x1 -= ICparams["boxsize"] 
    if x2>ICparams["boxsize"]:
        x2 -= ICparams["boxsize"]
    if x2>x1:     
        rho[np.logical_or(x<x1, x>x2)] = 0.01 * ICparams["rhoini"]
    if x1>x2:
        rho[x<x1] = 0.01 * ICparams["rhoini"]
    plt.plot(x, rho, color=kwargs['color'],ls='solid')
    #plt.axvline(x = x1,color=kwargs['color'],ls='dashed')
    #plt.axvline(x = x2,color=kwargs['color'],ls='dashed')




def main():
    ric = Simwrap()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(0,9,2):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        ReadandPlot(outfilename,ls='none',marker='o', mfc='none', markevery=10,color=next(ax._get_lines.prop_cycler)['color'])
    plt.show()    

if __name__ == "__main__":
    main()
