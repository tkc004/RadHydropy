from radhydropy.rsim import Rsim
import unyt
from radhydropy.analysis import rplot1d
#importing the os module
import os

#to get the current working directory
rundir = os.getcwd()
print('rundir',rundir)


def main():

    params_half = {
        'simname':'advection1d',
        'coordsys':'cartesian', #
        'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
        'savedir':rundir,
        'gamma':1.4, # for polytropic, the polytropic index
        'boxsize':1.0*unyt.pc, # length of the simulation domain
        'timesim':1.0*unyt.Myr, # total simulation time
        'ngrid':1000, # number of grid to discretize
        'CFL':0.1, # CFL condition for time-step
        'ftype':'half', # initial fluid distribution: uniform, half, gaussian
        'tini': 0.*unyt.yr, # initial time
        'rhoini': 1.0 * unyt.mp/ unyt.cm**3, 
        'vini': 1.0 * unyt.km/unyt.s,
        'tempini': 0.1 * unyt.K,
        'boundcond':'Periodic',
        'verbose':0, # speak out details?
    }

    mainrun = Rsim(params_half)
    mainrun.RunAll()
    rplot1d(mainrun,savefig=1,showfig=0)

if __name__ == "__main__":
    main()