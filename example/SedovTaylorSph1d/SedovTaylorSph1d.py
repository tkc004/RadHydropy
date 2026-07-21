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

et.set_plot_style()


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
    'boxsize':5.0*unyt.cm, # the simulation box size
    'time':0.0*unyt.s, # initial time
    'rhoini': 1.0 * unyt.g/unyt.cm**3,
    'Eini': 1.0e-4 * unyt.erg, # the total energy in central region
    #'Eini': 0.0 * unyt.erg, # the total energy in central region
    'rini': 0.5 * unyt.cm, # radius of central region
    'muini': 1.0,
    "rinj": 0.0*unyt.cm, 
}


def main():
    ric = et.Simwrap(ICparams, runparams)
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(5,10):
        outfilename = runparams['outdir']+'/'+runparams['outfileprefix']+'_%03d'%outindex+'.hdf5'
        et.ReadandPlot(
            outfilename,
            ICparams,
            runparams,
            ls='none',
            marker='o',
            mfc='none',
            markevery=1,
            color=next(ax._get_lines.prop_cycler)['color'],
        )
    figure_filename = rundir + '/SedovTaylorSph1D.jpg'
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)

if __name__ == "__main__":
    main()
