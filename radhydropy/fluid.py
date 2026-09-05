"""Fluid state container and primitive thermodynamic updates."""

import numpy as np
import unyt
import radhydropy.chemistry_species.hydrogen as rh
from radhydropy.units import _code_units, _to_code_quantity, photon_number_density, quantity_to_value
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.mesh import Mesh
from radhydropy.arrays import as_named_array
from radhydropy.state_boundaries import CodeFluidState, UnitBoundaryError
from radhydropy.cosmological_variables import (
    supercomoving_scale,
    to_supercomoving_density,
    to_supercomoving_temperature,
    to_supercomoving_velocity,
)


# set up fluid properties

class Fluid():
    """Store primitive and conserved fluid quantities.

    Fluid quantities are expected to be ``unyt`` arrays so that units propagate
    through pressure, energy, sound-speed, and finite-volume calculations.
    """

    # import mesh and EOS information into Fluid
    def __init__(self):
        self.time_code = 0.0

    @property
    def code_state(self):
        """Return the current runtime arrays as a validated typed state.

        The property is deliberately constructed on access so it cannot become
        stale while solver operators update the mutable fluid arrays.
        """
        specific_energy_code = None
        if hasattr(self, "eth_code"):
            if hasattr(self.eth_code, "units"):
                raise UnitBoundaryError(
                    "eth_code must be a unitless numeric code-unit array"
                )
            specific_energy_code = np.divide(
                np.asarray(self.eth_code, dtype=float),
                np.maximum(np.asarray(self.rho_code, dtype=float), np.finfo(float).tiny),
            )
        return CodeFluidState(
            rho_code=self.rho_code,
            vel_code=self.vel_code,
            temp_code=self.temp_code,
            pre_code=getattr(self, "pre_code", None),
            specific_energy_code=specific_energy_code,
            Mass_code=getattr(self, "Mass_code", None),
            Mom_code=getattr(self, "Mom_code", None),
            Energy_code=getattr(self, "Energy_code", None),
            ngamma_code=getattr(self, "ngamma_code", None),
            mu_dimensionless=getattr(self, "mu", None),
            xHI_dimensionless=getattr(self, "xHI", None),
            time_code=self.time_code,
        )

    def SetPressure(self):
        """Set gas pressure from density, temperature, and mean molecular weight."""
        self.pre_code = self.eos.pressure(self.rho_code, self.temp_code, self.mu)
        
    def SetEnergyDensity(self):
        """Set thermal energy density from pressure and the fluid EOS."""
        self.eth_code = self.eos.thermal_energy_density(self.pre_code)
        
    def SetSoundSpeed(self):
        """Set adiabatic sound speed from pressure, density, and the fluid EOS."""
        self.cs_code = self.eos.sound_speed(
            self.rho_code,
            self.pre_code,
            temp=self.temp_code,
            mu=self.mu,
        )

    def SetHydrogenMu(self, hydrogen_mass_fraction=1.0):
        """Set mean molecular weight from hydrogen neutral fraction."""
        self.mu = rh.mean_molecular_weight_mu(
            self.xHI,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
        )

    def SetHydrogenHeliumMu(self, hydrogen_mass_fraction=0.75, helium_mass_fraction=0.25):
        xHI = np.asarray(self.xHI, dtype=float)
        xHeI = np.asarray(self.xHeI, dtype=float)
        xHeII = np.asarray(self.xHeII, dtype=float)
        xHeIII = np.asarray(self.xHeIII, dtype=float)
        nH = hydrogen_mass_fraction * np.asarray(self.rho_code, dtype=float) / unyt.mp.to_value(unyt.g)
        nHe = helium_mass_fraction * np.asarray(self.rho_code, dtype=float) / (4.0 * unyt.mp.to_value(unyt.g))
        ne = nH * (1.0 - xHI) + nHe * (xHeII + 2.0 * xHeIII)
        nt = nH + nHe + ne
        self.mu = as_named_array(np.asarray(self.rho_code, dtype=float) / (unyt.mp.to_value(unyt.g) * np.maximum(nt, 1.0e-99)))
        
    def SetUpFluid(self, par, mesh=None):
        """Normalize primitive quantities into code units, append ghost cells, and
        initialize pressure.

        Parameters
        ----------
        par : object
            Parameter object with the ``noghost`` attribute.

        Raises
        ------
        Exception
            If any required primitive quantity is missing.
        """
        code_units = _code_units(par)
        if code_units is None:
            raise ValueError("SetUpFluid requires configured code units")
        self.CodeUnits = code_units
        self.supercomoving = bool(getattr(par, 'supercomoving_coordinates', False))
        self.time_code = 0.0

        # check if the required attributes exist
        attrlist = ['rho_code','temp_code','mu','vel_code']
        valuelist = [1.0, 0.0, 1.0, 0.0]
        for attr in attrlist:
            if not hasattr(self, attr):
                raise Exception("%s does not exist in fluid; quitting."%attr)

        self.rho_code = as_named_array(quantity_to_value(self.rho_code, code_units.density_unit))
        self.temp_code = as_named_array(quantity_to_value(self.temp_code, code_units.temperature_unit))
        self.vel_code = as_named_array(quantity_to_value(self.vel_code, code_units.velocity_unit))
        if getattr(par, 'gas_angular_momentum', False) or hasattr(
            self, 'specific_angular_momentum_code'
        ):
            if not hasattr(self, 'specific_angular_momentum_code'):
                self.specific_angular_momentum_code = as_named_array(
                    np.full(
                        np.shape(self.rho_code),
                        float(getattr(par, 'gas_specific_angular_momentum', 0.0)),
                    )
                )
            else:
                self.specific_angular_momentum_code = as_named_array(
                    quantity_to_value(
                        self.specific_angular_momentum_code,
                        code_units.length_unit * code_units.velocity_unit,
                    )
                )

        if getattr(par, 'supercomoving_coordinates', False):
            if not hasattr(par, 'cosmology'):
                raise ValueError("supercomoving coordinates require par.cosmology")
            a, hubble = supercomoving_scale(par, time=self.time_code)
            gamma = float(self.eos.gamma)
            if getattr(par, 'density_representation', 'physical') == 'physical':
                self.rho_code = as_named_array(to_supercomoving_density(self.rho_code, a))
            if getattr(par, 'temperature_representation', 'physical') == 'physical':
                self.temp_code = as_named_array(to_supercomoving_temperature(self.temp_code, a, gamma))
            if getattr(par, 'velocity_representation', 'physical') == 'physical':
                if mesh is None:
                    raise ValueError(
                        "physical velocity ICs require mesh for supercomoving conversion"
                    )
                first = int(par.mesh.ghost_cells)
                last = first + int(par.mesh.grid_cells)
                radius = np.asarray(mesh.coordinate[first:last], dtype=float)
                self.vel_code = as_named_array(
                    to_supercomoving_velocity(self.vel_code, radius, a, hubble)
                )

        if getattr(par, 'hydrogen_chemistry', False) and not hasattr(self, 'xHI'):
            self.xHI = (
                as_named_array(np.ones(np.shape(self.rho_code), dtype=float))
                * getattr(par, 'hydrogen_xHI_initial', 1.0)
            )
        if hasattr(self, 'xHI'):
            attrlist.append('xHI')
            valuelist.append(getattr(par, 'hydrogen_xHI_initial', 1.0))
        if getattr(par, 'thermochemistry_network', 'hydrogen') == 'hydrogen_helium':
            for attr, default in (
                ('xHeI', getattr(par, 'hydrogen_helium_xHeI_initial', 1.0)),
                ('xHeII', getattr(par, 'hydrogen_helium_xHeII_initial', 0.0)),
                ('xHeIII', getattr(par, 'hydrogen_helium_xHeIII_initial', 0.0)),
            ):
                if not hasattr(self, attr):
                    setattr(self, attr, as_named_array(np.ones(np.shape(self.rho_code)) * default))
                attrlist.append(attr)
                valuelist.append(default)

        if (
            (
                getattr(par, 'hydrogen_radiation_field', False)
                or getattr(par, 'radiative_transfer', False)
            )
            and not hasattr(self, 'ngamma_code')
        ):
            ngamma_unit = code_units.number_density_unit
            self.ngamma_code = (
                as_named_array(np.ones(np.shape(self.rho_code), dtype=float))
                * quantity_to_value(
                    photon_number_density(getattr(par, 'hydrogen_ngamma_initial', 0.0)),
                    ngamma_unit,
                )
            )
        if hasattr(self, 'ngamma_code'):
            attrlist.append('ngamma_code')
            ngamma_initial = quantity_to_value(
                photon_number_density(getattr(par, 'hydrogen_ngamma_initial', 0.0)),
                code_units.number_density_unit,
            )
            valuelist.append(float(np.asarray(ngamma_initial, dtype=float)))

        if hasattr(self, 'specific_angular_momentum_code'):
            attrlist.append('specific_angular_momentum_code')
            valuelist.append(0.0)
        if getattr(par, 'gravity_potential_energy', False) and not hasattr(
            self, 'GravitationalPotentialEnergy_code'
        ):
            self.GravitationalPotentialEnergy_code = as_named_array(
                np.zeros(np.shape(self.rho_code), dtype=float)
            )
        if hasattr(self, 'GravitationalPotentialEnergy_code'):
            attrlist.append('GravitationalPotentialEnergy_code')
            valuelist.append(0.0)
            

        #add ghost cells:
        noghost = int(par.mesh.ghost_cells)
        for iattr, attr in enumerate(attrlist): 
            quan = getattr(self, attr)
            #print('attr,qaun',attr,quan)
            units = getattr(quan, 'units', None)
            if attr == 'ngamma_code' and np.ndim(quan) == 2:
                values = np.asarray(quan, dtype=float)
                ghost = np.full(
                    (values.shape[0], noghost),
                    valuelist[iattr],
                    dtype=float,
                )
                quan = as_named_array(
                    np.concatenate((ghost, values, ghost), axis=1)
                )
                setattr(self, attr, quan)
                continue
            if units is None:
                ghost = np.ones(noghost, dtype=float) * valuelist[iattr]
                quan = as_named_array(np.concatenate((ghost, np.asarray(quan, dtype=float), ghost)))
            else:
                values = np.concatenate(
                    (
                        np.ones(noghost, dtype=float) * valuelist[iattr],
                        np.asarray(quan.to_value(units), dtype=float),
                        np.ones(noghost, dtype=float) * valuelist[iattr],
                    )
                )
                quan = as_named_array(values * units)
            setattr(self, attr, quan)
        if hasattr(self, 'xHI'):
            self.xHI = rh.clip_neutral_fraction(self.xHI)
        if (
            getattr(par, 'hydrogen_chemistry', False)
            and getattr(par, 'hydrogen_update_mu', False)
        ):
            if getattr(par, 'thermochemistry_network', 'hydrogen') == 'hydrogen_helium':
                self.SetHydrogenHeliumMu(
                    hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 0.75),
                    helium_mass_fraction=getattr(par, 'helium_mass_fraction', 0.25),
                )
            else:
                self.SetHydrogenMu(
                    hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
                )
        self.SetPressure() 

    def SetTemperature(self):
        """Set gas temperature from density, pressure, and mean molecular weight."""
        self.temp_code = self.eos.temperature(self.rho_code, self.pre_code, self.mu)

    def SetFluidTime(self, time): 
        """Set the current fluid time."""
        self.time_code = time
