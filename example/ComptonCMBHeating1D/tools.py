"""Small IC helper for the fixed-density CMB Compton example."""

import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.units import CodeUnits


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    """Build the physical IC from the nested example configuration.

    ``icparams`` is deliberately kept separate from ``runtime`` so IC-only
    values cannot leak into the parameter namespace passed to ``Rsim``.
    """

    def __init__(self, icparams, runtime):
        simulation = runtime['simulation']
        mesh = runtime['mesh']
        code_units = CodeUnits.from_mapping(runtime['units']['CodeUnits'])

        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(
            current_time=icparams['current_time'],
            box_size=icparams['box_size'],
            coordinate_system=simulation['coordinate_system'],
        )
        self.par.mesh = SimpleNamespace(
            grid_cells=int(mesh['grid_cells']),
            ghost_cells=0,
        )
        self.par.hydrodynamics = SimpleNamespace(
            gamma=float(runtime.get('hydrodynamics', {}).get('gamma', 5.0 / 3.0)),
        )
        self.par.coordinate_frame = 'physical'
        self.par.velocity_representation = 'physical'
        self.par.density_representation = 'physical'
        self.par.temperature_representation = 'physical'

        self.mesh.boundary = np.linspace(
            0.0,
            1.0,
            self.par.mesh.grid_cells + 1,
        ) * icparams['box_size']
        self.fluid.rho_code = (
            np.ones(self.par.mesh.grid_cells)
            * icparams['hydrogen_density']
            * unyt.mp
        ).to(unyt.g / unyt.cm**3)
        self.fluid.vel_code = np.zeros(self.par.mesh.grid_cells) * unyt.cm / unyt.s
        self.fluid.temp_code = np.ones(self.par.mesh.grid_cells) * icparams['initial_temperature']
        self.fluid.xHI = np.ones(self.par.mesh.grid_cells) * icparams['xHI']
        self.fluid.mu = np.ones(self.par.mesh.grid_cells) * icparams['mean_molecular_weight']
