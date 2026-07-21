from radhydropy.rsim import Rsim
import unyt
import os
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import radhydropy.io as rio
import tools as et

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
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'Periodic',
    'verbose':0, # speak out details?
    'order': 1,
    'dtmin': 2.0e-8*unyt.s,
    'dtmax': 2.0e-1*unyt.s,   
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

def main():
    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll()
    outindex = 2
    outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
    et.ReadandPlot(
        outfilename,
        ICparams,
        runparams,
        ls='none',
        marker='o',
        mfc='none',
        markevery=5,
    )
    figure_filename = rundir + '/SodShock1D.jpg'
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)

if __name__ == "__main__":
    main()
