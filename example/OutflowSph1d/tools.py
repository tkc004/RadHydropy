"""Helper utilities for the spherical outflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs
import outflow_sph_analytic as oa


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)

        self.mesh.boundary = np.linspace(
            icparams['rinj'],
            icparams['rinj'] + self.par.boxsize[0],
            self.par.nogrid + 1,
        )
        self.fluid.vel = icparams['vini'] * np.ones(self.par.nogrid)
        self.fluid.temp = icparams['tempini'] * np.ones(self.par.nogrid)
        self.fluid.rho = icparams['rhoini'] * np.ones(self.par.nogrid)
        self.fluid.mu = icparams['muini'] * np.ones(self.par.nogrid)


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = Simwrap(icparams)
    code_units_obj = CodeUnits.from_mapping(runparams.get('CodeUnits'))
    rout.par.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    boundary = np.asarray(rout.mesh.boundary, dtype=float)
    x_center = 0.5 * (boundary[1:] + boundary[:-1]) * code_units_obj.length_unit
    rho_num = code_quantity_to_cgs(rout.fluid.rho, code_units_obj, 'density_g_cm3')
    rho_num = rho_num * (1.0 * runparams['rho_outflow'].units)
    rho_ana = oa.density_profile(
        x_center,
        runparams['rho_outflow'],
        icparams['rinj'],
    )
    front = oa.front_position(
        rout.par.time * code_units_obj.time_unit,
        runparams['vel_outflow'],
    )
    x_values = x_center.to_value(icparams['boxsize'].units)
    rho_values = np.asarray(rho_num.to_value(runparams['rho_outflow'].units), dtype=float)
    rho_ana_values = np.asarray(rho_ana.to_value(runparams['rho_outflow'].units), dtype=float)
    plt.plot(x_values, rho_values, **kwargs)
    plt.plot(
        x_values,
        rho_ana_values,
        ls='dashed',
        color='k',
    )
    plt.axvline(
        x=front.to_value(icparams['boxsize'].units),
        color=kwargs['color'],
        ls='dashed',
    )
    plt.xlim(
        xmin=float(np.min(x_values)),
        xmax=float(np.max(x_values)),
    )
    plt.yscale('log')
    plt.xlabel(r'Radius [cm]')
    plt.ylabel(r'$\rho$ [g/cm$^3$]')
