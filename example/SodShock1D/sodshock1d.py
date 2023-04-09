from radhydropy.rsim import Rsim
import unyt
from radhydropy.analysis import rplot1d
import matplotlib.pyplot as plt
import numpy as np
import radhydropy.utils as ru 
import radhydropy.io as rio
#importing the os module
import os
from sodshock_analytic import shocktubecal, shocktubeanalyticgraph

#to get the current working directory
rundir = os.getcwd()
print('rundir',rundir)


runparams = {
    'simname':'SodShock1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outdir':rundir,
    'outfileprefix':'Output', 
    'savedir':rundir,
    'coordsys':'cartesian', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':1.0*unyt.s, # final simulation time
    'outdeltatime':1.0*unyt.s *0.1,
    'ngrid':1000, # number of grid to discretize
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'Periodic',
    'verbose':0, # speak out details?
}

ICparams = {
    'nogrid':1000, # number of grid points
    'coordsys':"cartesian", # coordinate system
    'boxsize':4.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'rhoini':1.0 * unyt.g/ unyt.cm**3, # initial density (of the higher density end)
    'vini':0.0 * unyt.km/unyt.s, #initial velocity
    'tempini':1.0 * unyt.g / unyt.cm / unyt.s**2 * (1.28 * unyt.mp) / unyt.kb / (1.0 * unyt.g/unyt.cm**3),
    'muini': 1.0,
    'rhoratio':0.1,
    'tempratio':0.1/0.125
}


def getAnalyticSolution(ICparams, rout):
    p5 = ru.CalPressure(ICparams['rhoini'],ICparams['tempini'],ICparams['muini'])
    p1 = ru.CalPressure(ICparams['rhoini']*ICparams['rhoratio'],ICparams['tempini']*ICparams['tempratio'],ICparams['muini'])
    # scipy cannot handle units!
    p5 = np.array(p5.in_cgs())
    p1 = np.array(p1.in_cgs())
    rho5 = np.array(ICparams['rhoini'].in_cgs())
    rho1 = np.array((ICparams['rhoini']*ICparams['rhoratio']).in_cgs())

    rho2, rho3, p2, v2, vt, vs, Mach = shocktubecal(runparams['gamma'], rho1, rho5, p1, p5)
    rho_ana, p_ana, v_ana = shocktubeanalyticgraph(runparams['gamma'], rho1, rho2, rho3, rho5, p1, p2, p5, v2, vt, vs, np.array(rout.par.time.in_cgs()), np.array(rout.mesh.boundary.in_cgs()), np.array(0.25* ICparams['boxsize']))
    return rho_ana, p_ana, v_ana 


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

        #check the dimension of the initial condition
        if ru.CheckParamDimen(ICparams) == False:
            raise Exception("unit not correctly set in params")
        # should be read from parameter files instead
        self.par.nogrid = ICparams['nogrid']
        self.par.coordsys = ICparams['coordsys']
        self.par.boxsize = ICparams['boxsize'] * np.ones(1)
        self.par.time = np.array([0.0]) * ICparams['time']

        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.par.boxsize[0]/self.par.nogrid

        # generate initial condition
        self.mesh.boundary = np.linspace(-dx,self.par.boxsize[0]+dx,self.par.nogrid+3)
        coordinate = 0.5 * (self.mesh.boundary[1:]+self.mesh.boundary[:-1])
        #print('coordinate',coordinate)

        rho = np.ones(self.par.nogrid+2) * ICparams['rhoini']
        self.fluid.vel = np.ones(self.par.nogrid+2) * ICparams['vini']
        # label the region with low density:
        indexlow = np.logical_and(coordinate>0.25*self.par.boxsize[0],coordinate<0.75*self.par.boxsize[0])
        rho[indexlow] *= ICparams['rhoratio']
        self.fluid.rho = rho
        temp = np.ones(self.par.nogrid+2) * ICparams['tempini']
        temp[indexlow] *= ICparams['tempratio']
        self.fluid.temp = temp
        # mean molecular weight
        self.fluid.mu = np.ones(self.par.nogrid+2) * ICparams['muini'] # for primordial neutral gas 

 

def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout,showfig=0,showhalf=1,**kwargs)
    rho_ana, p_ana, v_ana = getAnalyticSolution(ICparams, rout)
    plt.plot(rout.mesh.boundary,rho_ana)


def main():
    ric = Simwrap()
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll()
    outindex = 2
    outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
    ReadandPlot(outfilename,ls='none',marker='o', mfc='none', markevery=5)
    plt.show() 

if __name__ == "__main__":
    main()