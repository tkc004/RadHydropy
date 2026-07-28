"""Fluid state container and primitive thermodynamic updates."""

import numpy as np
import unyt
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.mesh import Mesh

# set up fluid properties

class Fluid():
    """Store primitive and conserved fluid quantities.

    Fluid quantities are expected to be ``unyt`` arrays so that units propagate
    through pressure, energy, sound-speed, and finite-volume calculations.
    """

    # import mesh and EOS information into Fluid
    def __init__(self):
        self.time = 0.0 * unyt.s

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
        self.mu = rh.pure_hydrogen_mu(
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
        # check if the required attributes exist
        attrlist = ['rho','temp','mu','vel'] 
        valuelist = [1.0, 0.0, 1.0, 0.0]
        for attr in attrlist:
            if not hasattr(self, attr):
                raise Exception("%s does not exist in fluid; quitting."%attr)

        if getattr(par, 'hydrogen_chemistry', False) and not hasattr(self, 'xHI'):
            self.xHI = (
                np.ones(np.shape(self.rho), dtype=float)
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
            self.ngamma = (
                np.ones(np.shape(self.rho), dtype=float)
                * rh.photon_number_density(getattr(par, 'hydrogen_ngamma_initial', 0.0))
            )
        if hasattr(self, 'ngamma'):
            attrlist.append('ngamma')
            ngamma_initial = rh.photon_number_density(
                getattr(par, 'hydrogen_ngamma_initial', 0.0)
            )
            valuelist.append(ngamma_initial.to_value(self.ngamma.units))
            

        #add ghost cells:
        noghost = par.noghost
        for iattr, attr in enumerate(attrlist): 
            quan = getattr(self, attr)
            #print('attr,qaun',attr,quan)
            try:
                units = quan.units
            except AttributeError:
                units = 1.0
            ghost = np.ones(noghost) * units * valuelist[iattr] 
            quan = unyt.uconcatenate((ghost,quan,ghost))
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
