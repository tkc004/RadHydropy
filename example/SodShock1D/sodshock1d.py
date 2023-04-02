from radhydropy.rsim import Rsim
import unyt
from radhydropy.analysis import rplot1d
#importing the os module
import os

#to get the current working directory
rundir = os.getcwd()
print('rundir',rundir)


def main():
    params_sodshock = {
        'simname':'SodShock',
        'savedir':rundir, #save at the directory the python is run.
        'coordinate':'cartesian', #
        'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
        'gamma':5./3., # for polytropic, the polytropic index
        'boxsize':10.0*unyt.cm, # length of the simulation domain
        'timesim':1.0*unyt.s, # total simulation time
        'ngrid':1000, # number of grid to discretize
        'CFL':0.1, # CFL condition for time-step
        'ftype':'sodshock', # initial fluid distribution: uniform, half, gaussian, sodshock
        'tini': 0.*unyt.yr, # initial time
        'rhoini':  1.0 * unyt.g/unyt.cm**3, 
        'vini': 0.0 * unyt.cm/unyt.s,
        'tempini': 1.0 * unyt.g / unyt.cm / unyt.s**2 * (1.28 * unyt.mp) / unyt.kb / (1.0 * unyt.g/unyt.cm**3),
        'boundcond':'Periodic',
        'verbose':0, # speak out details?
        'fixtemp':1, #fix the value of temperature
        'fixvel':1   #fix the value of velocity
    }

    mainrun = Rsim(params_sodshock)
    mainrun.RunAll()
    rplot1d(mainrun,savefig=1,showfig=0)

if __name__ == "__main__":
    main()