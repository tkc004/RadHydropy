"""Uniform ionized hydrogen cooling box.

This example starts with a static, uniform box of pure ionized hydrogen at
``T = 1e6 K``. Hydrogen cooling and chemistry are enabled, and the run stops
once the mean temperature reaches roughly ``2e4 K``.
"""

import os
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
import unyt

from radhydropy.rsim import Rsim
import radhydropy.io as rio
import tools as et


rundir = os.getcwd()

runparams = {
    'simname': 'HydrogenCooling1D',
    'ICfilename': rundir + '/InitialCondition.hdf5',
    'outdir': rundir,
    'outfileprefix': 'Output',
    'outdeltatime': 1.0e4 * unyt.yr,
    'savedir': rundir,
    'coordsys': 'cartesian',
    'EOStype': 'polytropic',
    'gamma': 5.0 / 3.0,
    'timesim': 1.0e6 * unyt.yr,
    'area': 1.0 * unyt.cm**2,
    'CFL': 0.5,
    'boundcond': 'Periodic',
    'vel_inflow': 0.0 * unyt.cm / unyt.s,
    'rho_inflow': 1.0 * unyt.mp / unyt.cm**3,
    'temp_inflow': 0.0 * unyt.K,
    'mu_inflow': 1.0,
    'vel_outflow': 0.0 * unyt.cm / unyt.s,
    'rho_outflow': 1.0 * unyt.mp / unyt.cm**3,
    'temp_outflow': 0.0 * unyt.K,
    'mu_outflow': 1.0,
    'noghost': 2,
    'verbose': 0,
    'order': 0,
    'dtmin': 1.0e-6 * unyt.yr,
    'dtmax': 1.0e3 * unyt.yr,
    'hydrogen_chemistry': True,
    'hydrogen_mass_fraction': 1.0,
    'hydrogen_xHI_initial': 0.0,
    'hydrogen_xHI_inflow': 0.0,
    'hydrogen_xHI_outflow': 0.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_update_mu': True,
}

ICparams = {
    'nogrid': 16,
    'coordsys': 'cartesian',
    'boxsize': 1.0 * unyt.kpc,
    'time': 0.0 * unyt.yr,
    'nHini': 100.0 / unyt.cm**3,
    'tempini': 1.0e6 * unyt.K,
    'xHIini': 0.0,
}

target_temperature = 2.0e4 * unyt.K


def main():
    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    history = {'time_yr': [], 'temperature_K': [], 'xHI': []}
    outindex = 0
    output_interval = sim.par.outdeltatime.copy()
    next_output_time = output_interval.copy()
    last_output_time = et.write_output(sim, outindex)
    et.append_history(sim, history)
    outindex += 1

    while (
        et.mean_temperature(sim) > target_temperature
        and et.time_value(sim, unyt.s) < float(sim.par.timesim.to_value(unyt.s))
    ):
        sim.Step(mode='hydro_sources')
        et.append_history(sim, history)
        if sim.fluid.time >= next_output_time:
            last_output_time = et.write_output(sim, outindex)
            outindex += 1
            next_output_time += output_interval

    if et.time_value(sim, unyt.s) != last_output_time:
        et.write_output(sim, outindex)

    figure_filename = rundir + '/HydrogenCooling1D.jpg'
    et.save_history_plot(history, figure_filename)

    print('Hydrogen cooling example finished')
    print('time = %.3e yr' % et.time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % et.mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % et.mean_neutral_fraction(sim))
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
