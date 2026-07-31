"""Fluid state container and primitive thermodynamic updates."""

import numpy as np
import unyt
import radhydropy.chemistry_species.hydrogen as rh
from radhydropy.units import _code_units, _to_code_quantity, photon_number_density, quantity_to_value
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.mesh import Mesh
from radhydropy.arrays import as_named_array


# set up fluid properties

class Fluid():
    """Store primitive and conserved fluid quantities.

    Fluid quantities are expected to be ``unyt`` arrays so that units propagate
    through pressure, energy, sound-speed, and finite-volume calculations.
    """

    # import mesh and EOS information into Fluid
    def __init__(self):
        self.time = 0.0

    def SetPressure(self):
        """Set gas pressure from density, temperature, and mean molecular weight."""
        self.pre = self.eos.pressure(self.rho, self.temp, self.mu)
        
    def SetEnergyDensity(self):
        """Set thermal energy density from pressure and the fluid EOS."""
        self.eth = self.eos.thermal_energy_density(self.pre)
        
    def SetSoundSpeed(self):
        """Set adiabatic sound speed from pressure, density, and the fluid EOS."""
        self.cs = self.eos.sound_speed(
            self.rho,
            self.pre,
            temp=self.temp,
            mu=self.mu,
        )

    def SetHydrogenMu(self, hydrogen_mass_fraction=1.0):
        """Set mean molecular weight from hydrogen neutral fraction."""
        self.mu = rh.mean_molecular_weight_mu(
            self.xHI,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
        )
        
    def SetUpFluid(self, par):
        """Append ghost cells to primitive quantities and initialize pressure.

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
            raise ValueError("SetUpFluid requires par.CodeUnits")
        self.code_units = code_units
        self.time = 0.0

        # check if the required attributes exist
        attrlist = ['rho','temp','mu','vel'] 
        valuelist = [1.0, 0.0, 1.0, 0.0]
        for attr in attrlist:
            if not hasattr(self, attr):
                raise Exception("%s does not exist in fluid; quitting."%attr)

        self.rho = as_named_array(quantity_to_value(self.rho, code_units.density_unit))
        self.temp = as_named_array(quantity_to_value(self.temp, code_units.temperature_unit))
        self.vel = as_named_array(quantity_to_value(self.vel, code_units.velocity_unit))

        if getattr(par, 'hydrogen_chemistry', False) and not hasattr(self, 'xHI'):
            self.xHI = (
                as_named_array(np.ones(np.shape(self.rho), dtype=float))
                * getattr(par, 'hydrogen_xHI_initial', 1.0)
            )
        if hasattr(self, 'xHI'):
            attrlist.append('xHI')
            valuelist.append(getattr(par, 'hydrogen_xHI_initial', 1.0))

        if (
            (
                getattr(par, 'hydrogen_radiation_field', False)
                or getattr(par, 'radiative_transfer', False)
            )
            and not hasattr(self, 'ngamma')
        ):
            ngamma_unit = code_units.number_density_unit
            self.ngamma = (
                as_named_array(np.ones(np.shape(self.rho), dtype=float))
                * quantity_to_value(
                    photon_number_density(getattr(par, 'hydrogen_ngamma_initial', 0.0)),
                    ngamma_unit,
                )
            )
        if hasattr(self, 'ngamma'):
            attrlist.append('ngamma')
            ngamma_initial = quantity_to_value(
                photon_number_density(getattr(par, 'hydrogen_ngamma_initial', 0.0)),
                code_units.number_density_unit,
            )
            valuelist.append(float(np.asarray(ngamma_initial, dtype=float)))
            

        #add ghost cells:
        noghost = par.noghost
        for iattr, attr in enumerate(attrlist): 
            quan = getattr(self, attr)
            #print('attr,qaun',attr,quan)
            units = getattr(quan, 'units', None)
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
            self.SetHydrogenMu(
                hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
            )
        self.SetPressure() 

    def SetTemperature(self):
        """Set gas temperature from density, pressure, and mean molecular weight."""
        self.temp = self.eos.temperature(self.rho, self.pre, self.mu)

    def SetFluidTime(self, time): 
        """Set the current fluid time."""
        self.time = time       
