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
    'simname':'advection1d',
    'ICfilename':rundir+'/InitialCondition.hdf5',
    'outdir':rundir,
    'outfileprefix':'Output', 
    'outdeltatime':2.0*unyt.s *0.1,
    'savedir':rundir,
    'coordsys':'cartesian', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':2.0*unyt.s, # final simulation time
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'Periodic',
    'noghost': 5, #number of ghost cells in front (equal number of ghost cell after)
    'verbose':0, # speak out details?
    'order': 0,
    'dtmin': 2.0e-8*unyt.s,
    'dtmax': 2.0e-1*unyt.s,   
}

ICparams = {
    'nogrid':100, # number of grid points
    'coordsys':"cartesian", # coordinate system
    'boxsize':1.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'rhoini':1.0 * unyt.g/ unyt.cm**3, # initial density (of the higher density end)
    'vini':1.0 * unyt.cm/unyt.s, #initial velocity
    'tempini': 1e-10 * unyt.K,
    'muini': 1.0,
}

def main():
    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(0,10,5):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        et.ReadandPlot(
            outfilename,
            ICparams,
            runparams,
            ls='none',
            marker='o',
            mfc='none',
            markevery=10,
            color=next(ax._get_lines.prop_cycler)['color'],
        )
    figure_filename = rundir + '/Advection1D.jpg'
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)

if __name__ == "__main__":
    main()
