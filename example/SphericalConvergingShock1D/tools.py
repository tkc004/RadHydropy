"""Initial-condition and output helpers for the spherical shock example."""

from types import SimpleNamespace

import numpy as np

import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.units import CodeUnits, quantity_to_value


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    """Build the spherical converging-flow IC for ``writehdf5``."""

    def __init__(self, icparams, runparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        if code_units is None:
            code_units = CodeUnits.from_mapping(
                runparams['units']['CodeUnits']
            )
        self.par.CodeUnits = code_units
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(icparams['grid_cells'])
        self.par.coordsys = icparams['coordinate_system']
        rmin = quantity_to_value(
            icparams['rmin'], code_units.length_unit
        )
        rmax = quantity_to_value(
            icparams['rmax'], code_units.length_unit
        )
        if not 0.0 < rmin < rmax:
            raise ValueError('spherical IC requires 0 < rmin < rmax')
        self.par.inner_radius = rmin
        self.par.boxsize = np.asarray([rmax])
        self.par.time = np.asarray([
            quantity_to_value(icparams['current_time'], code_units.time_unit)
        ])
        self.par.simulation = SimpleNamespace(
            current_time=self.par.time,
            box_size=self.par.boxsize,
            coordinate_system=self.par.coordsys,
        )
        self.par.mesh = SimpleNamespace(
            grid_cells=self.par.nogrid,
            ghost_cells=0,
        )
        hydro = runparams['hydrodynamics']
        self.par.hydrodynamics = SimpleNamespace(
            gamma=float(hydro['gamma']),
        )
        self.par.dual_energy = bool(hydro.get('dual_energy', False))

        faces = np.linspace(
            self.par.inner_radius,
            self.par.boxsize[0],
            self.par.nogrid + 1,
        )
        self.mesh.boundary = faces
        self.mesh.coordinate = 0.5 * (faces[1:] + faces[:-1])
        self.mesh.area = 4.0 * np.pi * faces[:-1] ** 2
        self.mesh.vol = 4.0 * np.pi / 3.0 * np.diff(faces ** 3)
        self.fluid.rho_code = np.full(
            self.par.nogrid,
            quantity_to_value(
                icparams['initial_density'], code_units.density_unit
            ),
        )
        self.fluid.temp_code = np.full(
            self.par.nogrid,
            quantity_to_value(
                icparams['temperature'], code_units.temperature_unit
            ),
        )
        self.fluid.mu = np.full(
            self.par.nogrid, float(icparams['mean_molecular_weight'])
        )
        self.fluid.vel_code = np.full(
            self.par.nogrid,
            quantity_to_value(
                icparams['velocity'], code_units.velocity_unit
            ),
        )


def read_output(filename, runparams):
    """Read one output with the metadata needed by the HDF5 reader."""
    code_units = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    result = Simwrap.__new__(Simwrap)
    result.par = Par()
    result.mesh = Mesh()
    result.fluid = Fluid()
    result.par.CodeUnits = code_units
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.simulation = SimpleNamespace(coordinate_system='spherical')
    result.par.mesh = SimpleNamespace(
        grid_cells=int(runparams['mesh']['grid_cells']),
        ghost_cells=int(runparams['mesh']['ghost_cells']),
    )
    result.par.hydrodynamics = SimpleNamespace(
        gamma=float(runparams['hydrodynamics']['gamma']),
    )
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    result.fluid.eos = EOS(
        runparams['hydrodynamics']['eos_type'],
        result.par.hydrodynamics.gamma,
        code_units,
    )
    return result
