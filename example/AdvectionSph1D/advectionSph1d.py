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
    'boxsize':2.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'tempini': 0.0 * unyt.K,
    'muini': 1.0,
    'alpha_q': 1.0 / unyt.s,
    'a_q': 10.0 / unyt.cm,
    'b_q': 0.5 * unyt.cm,
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
        tempini = ICparams["tempini"]
        muini = ICparams["muini"]
         
        # boundary points of the mesh
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions
        dx = self.par.boxsize[0]/self.par.nogrid

        # generate initial condition
        self.mesh.boundary = np.linspace(dx,self.par.boxsize[0]+dx,self.par.nogrid+1)
        coordinate = 0.5 * (self.mesh.boundary[1:]+self.mesh.boundary[:-1])
        #print('coordinate',coordinate)
        #self.fluid.vel = 0.01*unyt.cm/unyt.s * np.ones(self.par.nogrid+2)
        self.fluid.vel = ICparams["alpha_q"] * coordinate 
        #self.fluid.vel = 1.0*unyt.cm/unyt.s * np.ones(self.par.nogrid+2)
        self.fluid.temp = 0.01 *unyt.K * Qaussian(coordinate,ICparams["a_q"],ICparams["b_q"])
        self.fluid.rho = 1.0 * unyt.g/unyt.cm**3 * np.ones(self.par.nogrid)
        # mean molecular weight
        self.fluid.mu = np.ones(self.par.nogrid) * ICparams["muini"] # for primordial neutral gas 

# I think the analytic solution only work for constant density
# we cannot do density advection
# we cannot do temperature advection neither
# since temperature changes with density
def ReadandPlot(outfilename,**kwargs):
    rout = Simwrap() 
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    rplot1d(rout,yquan='temp', showfig=0,**kwargs)
    #Qanalytic = Q_analytic(2,ICparams["alpha_q"],rout.par.time,rout.mesh.boundary, ICparams["a_q"],ICparams["b_q"])
    #plt.plot(rout.mesh.boundary,Qanalytic, color=kwargs['color'],ls='solid')
    #plt.axvline(x = rout.par.time * 1.0 *unyt.cm/unyt.s + ICparams['b_q'])




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