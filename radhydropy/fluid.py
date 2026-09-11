"""Fluid state container and primitive thermodynamic updates."""

import numpy as np
import unyt
import radhydropy.chemistry_species.hydrogen as rh
from radhydropy.units import _code_units, _to_code_quantity, photon_number_density, quantity_to_value
import radhydropy.utils as ru
from radhydropy.eos import EOS
from radhydropy.mesh import Mesh
from radhydropy.arrays import as_named_array
from radhydropy.state_boundaries import (
    ProperCodeState,
    SupercomovingCodeState,
    UnitBoundaryError,
)
from radhydropy.cosmological_variables import (
    supercomoving_scale,
    to_supercomoving_density,
    to_supercomoving_temperature,
    to_supercomoving_velocity,
)
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
    require_runtime_fields,
    runtime_fields,
)


# set up fluid properties

class Fluid():
    """Store primitive and conserved fluid quantities.

    Runtime fluid quantities are unitless numeric code-unit arrays. Physical
    quantities are converted at setup and source-state boundaries.
    """

    # import mesh and EOS information into Fluid
    def __init__(self):
        # The concrete representation-specific clock is installed by the
        # runtime setup.  Proper-code is the non-cosmological default for a
        # bare container used before parameterized setup.
        self.time_proper_code = 0.0
        self.runtime_fields = None
        self.runtime_state = None

    def configure_runtime_fields(self, par):
        """Select and validate the canonical field contract for this run."""
        fields = runtime_fields(par)
        self.runtime_fields = fields
        require_runtime_fields(self, fields, "fluid")
        return fields

    def _refresh_runtime_state(self):
        """Refresh the typed representation state after a primitive update."""
        if self.runtime_fields is None:
            return
        if self.runtime_fields is PROPER_RUNTIME_FIELDS:
            density = self.rho_proper_code
            velocity = self.vel_proper_code
            if not hasattr(self, "pre_proper_code"):
                self.pre_proper_code = as_named_array(np.zeros_like(density))
            pressure = self.pre_proper_code
            temperature = self.temp_proper_code
            time = self.time_proper_code
        elif self.runtime_fields is SUPERCOMOVING_RUNTIME_FIELDS:
            density = self.rho_comoving_code
            velocity = self.vel_supercomoving_code
            if not hasattr(self, "pre_supercomoving_code"):
                self.pre_supercomoving_code = as_named_array(np.zeros_like(density))
            pressure = self.pre_supercomoving_code
            temperature = self.temp_supercomoving_code
            time = self.tau_supercomoving_code
        else:
            raise ValueError("unknown runtime field representation")
        self.runtime_state = FluidRuntimeState.from_arrays(
            self.runtime_fields,
            **{self.runtime_fields.density: density, self.runtime_fields.velocity: velocity, self.runtime_fields.pressure: pressure, self.runtime_fields.temperature: temperature, self.runtime_fields.time: time},
            mu_dimensionless=getattr(self, "mu", None),
            xHI_dimensionless=getattr(self, "xHI", None),
        )

    @property
    def code_state(self):
        """Return the current runtime arrays as a validated typed state.

        The property is deliberately constructed on access so it cannot become
        stale while solver operators update the mutable fluid arrays.
        """
        if self.runtime_state is not None:
            if self.runtime_fields is None:
                raise UnitBoundaryError(
                    "typed runtime state requires configured representation fields"
                )
            if self.runtime_fields.time == "time_proper_code":
                density_code = self.runtime_state.rho_proper_code
                velocity_code = self.runtime_state.vel_proper_code
                temperature_code = self.runtime_state.temp_proper_code
                pressure_code = self.runtime_state.pre_proper_code
                time_code = self.runtime_state.time_proper_code
            else:
                density_code = self.runtime_state.rho_comoving_code
                velocity_code = self.runtime_state.vel_supercomoving_code
                temperature_code = self.runtime_state.temp_supercomoving_code
                pressure_code = self.runtime_state.pre_supercomoving_code
                time_code = self.runtime_state.tau_supercomoving_code
            specific_energy_code = None
            if hasattr(self, "eth_code"):
                specific_energy_code = np.divide(
                    np.asarray(self.eth_code, dtype=float),
                    np.maximum(
                        np.asarray(density_code, dtype=float),
                        np.finfo(float).tiny,
                    ),
                )
            state_type = (
                ProperCodeState
                if self.runtime_fields is PROPER_RUNTIME_FIELDS
                else SupercomovingCodeState
            )
            state_kwargs = {
                "specific_energy_proper_code" if self.runtime_fields is PROPER_RUNTIME_FIELDS
                else "specific_energy_supercomoving_code": specific_energy_code,
                "Mass_code": getattr(self, "Mass_code", None),
                "Mom_code": getattr(self, "Mom_code", None),
                "Energy_code": getattr(self, "Energy_code", None),
                "ngamma_code": getattr(self, "ngamma_code", None),
                "mu_dimensionless": getattr(self, "mu", None),
                "xHI_dimensionless": getattr(self, "xHI", None),
            }
            if self.runtime_fields is PROPER_RUNTIME_FIELDS:
                state_kwargs.update(
                    rho_proper_code=density_code,
                    vel_proper_code=velocity_code,
                    temp_proper_code=temperature_code,
                    pre_proper_code=pressure_code,
                    time_proper_code=time_code,
                )
            else:
                state_kwargs.update(
                    rho_comoving_code=density_code,
                    vel_supercomoving_code=velocity_code,
                    temp_supercomoving_code=temperature_code,
                    pre_supercomoving_code=pressure_code,
                    tau_supercomoving_code=time_code,
                )
            return state_type(**state_kwargs)
        raise UnitBoundaryError(
            "fluid must have a representation-specific typed runtime state"
        )

    def SetPressure(self):
        """Set gas pressure from density, temperature, and mean molecular weight."""
        if self.runtime_fields is PROPER_RUNTIME_FIELDS:
            self.pre_proper_code = as_named_array(
                self.eos.pressure(
                    self.rho_proper_code, self.temp_proper_code, self.mu
                )
            )
        elif self.runtime_fields is SUPERCOMOVING_RUNTIME_FIELDS:
            self.pre_supercomoving_code = as_named_array(
                self.eos.pressure(
                    self.rho_comoving_code,
                    self.temp_supercomoving_code,
                    self.mu,
                )
            )
        else:
            self.pre_code = self.eos.pressure(self.rho_code, self.temp_code, self.mu)
        self._refresh_runtime_state()
        
    def SetEnergyDensity(self):
        """Set thermal energy density from pressure and the fluid EOS."""
        if self.runtime_fields is PROPER_RUNTIME_FIELDS:
            pressure = self.pre_proper_code
        elif self.runtime_fields is SUPERCOMOVING_RUNTIME_FIELDS:
            pressure = self.pre_supercomoving_code
        else:
            pressure = self.pre_code
        self.eth_code = self.eos.thermal_energy_density(pressure)
        
    def SetSoundSpeed(self):
        """Set adiabatic sound speed from pressure, density, and the fluid EOS."""
        if self.runtime_fields is PROPER_RUNTIME_FIELDS:
            density = self.rho_proper_code
            pressure = self.pre_proper_code
            temperature = self.temp_proper_code
        elif self.runtime_fields is SUPERCOMOVING_RUNTIME_FIELDS:
            density = self.rho_comoving_code
            pressure = self.pre_supercomoving_code
            temperature = self.temp_supercomoving_code
        else:
            density = self.rho_code
            pressure = self.pre_code
            temperature = self.temp_code
        self.cs_code = self.eos.sound_speed(
            density,
            pressure,
            temp=temperature,
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
        if self.runtime_fields is PROPER_RUNTIME_FIELDS:
            density = self.rho_proper_code
        elif self.runtime_fields is SUPERCOMOVING_RUNTIME_FIELDS:
            density = self.rho_comoving_code
        else:
            density = self.rho_code
        nH = hydrogen_mass_fraction * np.asarray(density, dtype=float) / unyt.mp.to_value(unyt.g)
        nHe = helium_mass_fraction * np.asarray(density, dtype=float) / (4.0 * unyt.mp.to_value(unyt.g))
        ne = nH * (1.0 - xHI) + nHe * (xHeII + 2.0 * xHeIII)
        nt = nH + nHe + ne
        self.mu = as_named_array(np.asarray(density, dtype=float) / (unyt.mp.to_value(unyt.g) * np.maximum(nt, 1.0e-99)))
        
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
        if not getattr(par, "supercomoving_coordinates", False):
            return self._set_up_proper_fluid(par, code_units)
        return self._set_up_supercomoving_fluid(par, code_units)

    def _set_up_supercomoving_fluid(self, par, code_units):
        """Initialize a cosmological fluid using canonical fields."""
        required = (
            "rho_comoving_code",
            "vel_supercomoving_code",
            "temp_supercomoving_code",
            "mu",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise AttributeError(
                "supercomoving fluid setup requires: " + ", ".join(missing)
            )
        self.CodeUnits = code_units
        self.supercomoving = True
        self.runtime_fields = SUPERCOMOVING_RUNTIME_FIELDS
        self.tau_supercomoving_code = float(
            np.asarray(getattr(self, "tau_supercomoving_code", 0.0), dtype=float)
        )
        self.rho_comoving_code = as_named_array(
            quantity_to_value(self.rho_comoving_code, code_units.density_unit)
        )
        self.temp_supercomoving_code = as_named_array(
            quantity_to_value(self.temp_supercomoving_code, code_units.temperature_unit)
        )
        self.vel_supercomoving_code = as_named_array(
            quantity_to_value(self.vel_supercomoving_code, code_units.velocity_unit)
        )
        self.mu = as_named_array(np.asarray(self.mu, dtype=float))
        noghost = int(par.mesh.ghost_cells)
        for attr, default in (
            ("rho_comoving_code", 1.0),
            ("vel_supercomoving_code", 0.0),
            ("temp_supercomoving_code", 0.0),
            ("mu", 1.0),
        ):
            values = np.asarray(getattr(self, attr), dtype=float)
            setattr(
                self,
                attr,
                as_named_array(
                    np.concatenate((np.full(noghost, default), values, np.full(noghost, default)))
                ),
            )
        if hasattr(self, "specific_angular_momentum_code"):
            values = np.asarray(self.specific_angular_momentum_code, dtype=float)
            self.specific_angular_momentum_code = as_named_array(
                np.concatenate((np.zeros(noghost), values, np.zeros(noghost)))
            )
        if getattr(par, "hydrogen_chemistry", False) and not hasattr(self, "xHI"):
            self.xHI = as_named_array(
                np.full(np.shape(self.rho_comoving_code), getattr(par, "hydrogen_xHI_initial", 1.0))
            )
        if hasattr(self, "xHI"):
            values = rh.clip_neutral_fraction(np.asarray(self.xHI, dtype=float))
            self.xHI = as_named_array(
                np.concatenate((np.full(noghost, getattr(par, "hydrogen_xHI_initial", 1.0)), values, np.full(noghost, getattr(par, "hydrogen_xHI_initial", 1.0))))
            )
        if hasattr(self, "ngamma_code"):
            values = np.asarray(self.ngamma_code, dtype=float)
            self.ngamma_code = as_named_array(
                np.concatenate((np.zeros(noghost), values, np.zeros(noghost)))
            )
        self.SetPressure()
        return None

        self.CodeUnits = code_units
        self.supercomoving = bool(getattr(par, 'supercomoving_coordinates', False))
        self.time_code = 0.0

        # check if the required attributes exist
        attrlist = ['rho_code','temp_code','mu','vel_code']
        valuelist = [1.0, 0.0, 1.0, 0.0]
        for attr in attrlist:
            if not hasattr(self, attr):
                raise Exception("%s does not exist in fluid; quitting."%attr)

        # The current solver arrays are migrated in the next runtime slice;
        # fail early when a caller claims to provide the canonical container
        # without actually populating it.
        self.runtime_fields = runtime_fields(par)
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
            ghost = np.ones(noghost, dtype=float) * valuelist[iattr]
            quan = as_named_array(
                np.concatenate((ghost, np.asarray(quan, dtype=float), ghost))
            )
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

    def _set_up_proper_fluid(self, par, code_units):
        """Initialize a non-cosmological fluid using canonical proper fields.

        Input quantities may carry physical units at this boundary.  All
        stored runtime arrays are plain numeric proper-code arrays.
        """
        required = (
            "rho_proper_code",
            "vel_proper_code",
            "temp_proper_code",
            "mu",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise AttributeError(
                "proper-code fluid setup requires: " + ", ".join(missing)
            )

        self.CodeUnits = code_units
        self.supercomoving = False
        self.runtime_fields = PROPER_RUNTIME_FIELDS
        self.time_proper_code = float(
            np.asarray(getattr(self, "time_proper_code", 0.0), dtype=float)
        )
        self.rho_proper_code = as_named_array(
            quantity_to_value(self.rho_proper_code, code_units.density_unit)
        )
        self.temp_proper_code = as_named_array(
            quantity_to_value(self.temp_proper_code, code_units.temperature_unit)
        )
        self.vel_proper_code = as_named_array(
            quantity_to_value(self.vel_proper_code, code_units.velocity_unit)
        )
        self.mu = as_named_array(np.asarray(self.mu, dtype=float))

        if getattr(par, "gas_angular_momentum", False) or hasattr(
            self, "specific_angular_momentum_code"
        ):
            if not hasattr(self, "specific_angular_momentum_code"):
                self.specific_angular_momentum_code = as_named_array(
                    np.full(
                        np.shape(self.rho_proper_code),
                        float(getattr(par, "gas_specific_angular_momentum", 0.0)),
                    )
                )
            else:
                self.specific_angular_momentum_code = as_named_array(
                    quantity_to_value(
                        self.specific_angular_momentum_code,
                        code_units.length_unit * code_units.velocity_unit,
                    )
                )

        if getattr(par, "hydrogen_chemistry", False) and not hasattr(self, "xHI"):
            self.xHI = as_named_array(
                np.full(
                    np.shape(self.rho_proper_code),
                    float(getattr(par, "hydrogen_xHI_initial", 1.0)),
                )
            )
        if hasattr(self, "xHI"):
            self.xHI = rh.clip_neutral_fraction(
                as_named_array(np.asarray(self.xHI, dtype=float))
            )

        if getattr(par, "thermochemistry_network", "hydrogen") == "hydrogen_helium":
            for attr, default in (
                ("xHeI", getattr(par, "hydrogen_helium_xHeI_initial", 1.0)),
                ("xHeII", getattr(par, "hydrogen_helium_xHeII_initial", 0.0)),
                ("xHeIII", getattr(par, "hydrogen_helium_xHeIII_initial", 0.0)),
            ):
                if not hasattr(self, attr):
                    setattr(
                        self,
                        attr,
                        as_named_array(np.full(np.shape(self.rho_proper_code), default)),
                    )
                else:
                    setattr(self, attr, as_named_array(np.asarray(getattr(self, attr), dtype=float)))

        noghost = int(par.mesh.ghost_cells)
        for attr, default in (
            ("rho_proper_code", 1.0),
            ("vel_proper_code", 0.0),
            ("temp_proper_code", 0.0),
            ("mu", 1.0),
            ("xHI", getattr(par, "hydrogen_xHI_initial", 1.0)),
            ("xHeI", getattr(par, "hydrogen_helium_xHeI_initial", 1.0)),
            ("xHeII", getattr(par, "hydrogen_helium_xHeII_initial", 0.0)),
            ("xHeIII", getattr(par, "hydrogen_helium_xHeIII_initial", 0.0)),
        ):
            if hasattr(self, attr):
                values = np.asarray(getattr(self, attr), dtype=float)
                setattr(
                    self,
                    attr,
                    as_named_array(
                        np.concatenate(
                            (
                                np.full(noghost, default, dtype=float),
                                values,
                                np.full(noghost, default, dtype=float),
                            )
                        )
                    ),
                )
        if hasattr(self, "specific_angular_momentum_code"):
            values = np.asarray(self.specific_angular_momentum_code, dtype=float)
            self.specific_angular_momentum_code = as_named_array(
                np.concatenate((np.zeros(noghost), values, np.zeros(noghost)))
            )

        if (
            getattr(par, "hydrogen_radiation_field", False)
            or getattr(par, "radiative_transfer", False)
        ) and not hasattr(self, "ngamma_code"):
            self.ngamma_code = as_named_array(
                np.full(
                    np.shape(self.rho_proper_code),
                    quantity_to_value(
                        photon_number_density(
                            getattr(par, "hydrogen_ngamma_initial", 0.0)
                        ),
                        code_units.number_density_unit,
                    ),
                )
            )
        if hasattr(self, "ngamma_code"):
            values = np.asarray(self.ngamma_code, dtype=float)
            initial = float(
                np.asarray(
                    quantity_to_value(
                        photon_number_density(
                            getattr(par, "hydrogen_ngamma_initial", 0.0)
                        ),
                        code_units.number_density_unit,
                    )
                )
            )
            if values.ndim == 2:
                ghost = np.full(
                    (values.shape[0], noghost),
                    initial,
                    dtype=float,
                )
                self.ngamma_code = as_named_array(
                    np.concatenate((ghost, values, ghost), axis=1)
                )
            else:
                self.ngamma_code = as_named_array(
                    np.concatenate(
                        (
                            np.full(noghost, initial),
                            values,
                            np.full(noghost, initial),
                        )
                    )
                )

        if getattr(par, "gravity_potential_energy", False) and not hasattr(
            self, "GravitationalPotentialEnergy_code"
        ):
            self.GravitationalPotentialEnergy_code = as_named_array(
                np.zeros(np.shape(self.rho_proper_code), dtype=float)
            )
        if hasattr(self, "GravitationalPotentialEnergy_code"):
            values = np.asarray(self.GravitationalPotentialEnergy_code, dtype=float)
            self.GravitationalPotentialEnergy_code = as_named_array(
                np.concatenate((np.zeros(noghost), values, np.zeros(noghost)))
            )

        if (
            getattr(par, "hydrogen_chemistry", False)
            and getattr(par, "hydrogen_update_mu", False)
        ):
            if getattr(par, "thermochemistry_network", "hydrogen") == "hydrogen_helium":
                self.SetHydrogenHeliumMu(
                    hydrogen_mass_fraction=getattr(par, "hydrogen_mass_fraction", 0.75),
                    helium_mass_fraction=getattr(par, "helium_mass_fraction", 0.25),
                )
            else:
                self.SetHydrogenMu(
                    hydrogen_mass_fraction=getattr(par, "hydrogen_mass_fraction", 1.0)
                )
        self.SetPressure()

    def SetTemperature(self):
        """Set gas temperature from density, pressure, and mean molecular weight."""
        if self.runtime_fields is PROPER_RUNTIME_FIELDS:
            self.temp_proper_code = as_named_array(
                self.eos.temperature(
                    self.rho_proper_code, self.pre_proper_code, self.mu
                )
            )
        elif self.runtime_fields is SUPERCOMOVING_RUNTIME_FIELDS:
            self.temp_supercomoving_code = as_named_array(
                self.eos.temperature(
                    self.rho_comoving_code,
                    self.pre_supercomoving_code,
                    self.mu,
                )
            )
        else:
            self.temp_code = self.eos.temperature(self.rho_code, self.pre_code, self.mu)
        self._refresh_runtime_state()

    def SetFluidTime(self, time_proper_code):
        """Set the current numeric proper-code fluid time.

        Physical time quantities are accepted only at this input boundary and
        immediately converted to the configured code-time scalar.
        """
        if hasattr(time_proper_code, "to_value"):
            code_units = getattr(getattr(self, "eos", None), "code_units", None)
            if code_units is None:
                code_units = getattr(getattr(self, "eos", None), "CodeUnits", None)
            if code_units is None:
                code_units = getattr(self, "code_units", None)
            if code_units is None:
                raise ValueError(
                    "unit-bearing fluid time requires EOS code units"
                )
            time_proper_code = float(
                np.asarray(time_proper_code.to_value(code_units.time_unit))
            )
        else:
            time_proper_code = float(np.asarray(time_proper_code, dtype=float))
        fields = self.runtime_fields
        if fields is None:
            self.time_proper_code = time_proper_code
        else:
            setattr(self, fields.time, time_proper_code)
        self._refresh_runtime_state()
