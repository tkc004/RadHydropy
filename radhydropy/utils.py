"""Numerical and thermodynamic helper functions."""

import numpy as np
import unyt
from radhydropy.arrays import as_named_array


def periodic_roll(values, shift):
    """Return a 1D periodic shift without calling ``np.roll``."""
    out = np.empty_like(values)
    if out.size == 0:
        return out

    shift = int(shift) % out.shape[0]
    if shift == 0:
        out[...] = values
    else:
        out[:shift] = values[-shift:]
        out[shift:] = values[:-shift]
    return out

def SafeDivide(numerator, denominator):
    """Divide two ``unyt`` quantities and return zero where the denominator is zero."""
    if hasattr(numerator, "units") or hasattr(denominator, "units"):
        numerator_value, denominator_value = np.broadcast_arrays(
            np.asarray(getattr(numerator, "value", numerator), dtype=float),
            np.asarray(getattr(denominator, "value", denominator), dtype=float),
        )
        quotient = np.zeros_like(denominator_value, dtype=float)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            np.divide(
                numerator_value,
                denominator_value,
                out=quotient,
                where=denominator_value != 0.0,
            )
        numerator_units = getattr(numerator, "units", 1.0)
        denominator_units = getattr(denominator, "units", 1.0)
        return quotient * (numerator_units / denominator_units)
    numerator_value, denominator_value = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    quotient = np.zeros_like(denominator_value, dtype=float)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        np.divide(
            numerator_value,
            denominator_value,
            out=quotient,
            where=denominator_value != 0.0,
        )
    return as_named_array(quotient)

def CalPressure(rho,temp,mu):
    """Calculate ideal-gas pressure from density, temperature, and molecular weight."""
    if hasattr(rho, "units") or hasattr(temp, "units"):
        return rho / (mu * unyt.mp) * unyt.kb * temp
    return np.asarray(rho, dtype=float) * np.asarray(temp, dtype=float) / np.asarray(mu, dtype=float)

def CalTemperature(rho,pressure,mu):
    """Calculate ideal-gas temperature from density, pressure, and molecular weight."""
    if not (hasattr(rho, "units") or hasattr(pressure, "units")):
        return np.asarray(pressure, dtype=float) / np.asarray(rho, dtype=float) * np.asarray(mu, dtype=float)
    pressure_over_rho = SafeDivide(pressure, rho)
    return (pressure_over_rho * (mu * unyt.mp) / unyt.kb).to(unyt.K)

def CalEnergyDensity(pressure, gamma):
    """Calculate thermal energy density for a polytropic gas."""
    return pressure / (gamma-1.0)

def CalSoundSpeed(pressure,rho, gamma):
    """Calculate adiabatic sound speed and zero invalid values."""
    if not (hasattr(pressure, "units") or hasattr(rho, "units")):
        pressure_over_rho = SafeDivide(pressure, rho)
        soundspeed = np.sqrt(gamma * pressure_over_rho)
        soundspeed[np.isnan(soundspeed)] = 0.0
        return soundspeed
    pressure_over_rho = SafeDivide(pressure, rho)
    soundspeed = np.sqrt(gamma * pressure_over_rho).to(unyt.cm / unyt.s)
    soundspeed[np.isnan(soundspeed)] = 0.0 * unyt.cm/unyt.s
    return soundspeed


def CheckParamDimen(params):
    """Validate known dimensional parameters.

    Returns ``True`` when all recognized parameters have compatible dimensions;
    otherwise returns the first key with incompatible units.
    """
    unitdir = {'boxsize':1.0*unyt.pc, 'tini':1.0*unyt.yr, 'vini':1.0*unyt.pc/unyt.yr,
                'rhoini':1.0*unyt.g/unyt.cm**3, 'tempini':1.0*unyt.K, 'gamma':1.0}
    for key in unitdir: 
        if key in params.keys():
            try:
                CheckDimension(params[key],unitdir[key])
            except unyt.exceptions.UnitOperationError:
                return key
    return True


def CheckDimension(a,dimcheck):
    """Raise a ``unyt`` error if ``a`` is not dimensionally compatible."""
    if not hasattr(a, "units"):
        return
    dummy = a+dimcheck
    pass


def gaussian(x, mu, sig):
    """Evaluate a normalized one-dimensional Gaussian profile."""
    return np.exp(-0.5 * np.power(x - mu, 2.) / np.power(sig, 2.)) / (np.sqrt(2.0*np.pi) * sig)

def gaussiansph(r, sig):
    """Evaluate a normalized spherical Gaussian profile."""
    return np.exp(-0.5 * np.power(r, 2.) / np.power(sig, 2.)) / (np.sqrt(2.0*np.pi) * sig)**3


def CalGradient(quan,xdelta):
    """Calculate a centered periodic gradient."""
    # only work for periodic boundary condition!
    dqdx = (periodic_roll(quan, -1) - periodic_roll(quan, 1)) / (2. * xdelta)
    return dqdx

def CalInterFaceFluxGLF(flux_L: float, flux_R: float, q_L: float, q_R: float, cmax: float) -> float:
    """Calculate a Lax-Friedrichs interface flux."""
    # Global Lax Friedrich function
    # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)    
    InterFaceFlux = 0.5 * (flux_L + flux_R)
    # apply artifical diffusion +0.5*cmax*(q_L-q_R)
    InterFaceFlux += 0.5*cmax*(q_L-q_R)
    return InterFaceFlux
    
def CalFluxLimiter(rlim, limiter='minmod'):
    """Calculate a slope limiter from the ratio of neighboring gradients."""
    if limiter=='minmod':
        firststep = np.minimum(np.ones(len(rlim)), rlim)
        philim = np.maximum(np.zeros(len(rlim)), firststep)
    elif limiter=='vanLeer':
        # is it correct when rlim -> inf, philim -> 2?
        philim = (rlim + np.absolute(rlim)) / (1.0 + np.absolute(rlim))
    else:
        raise ValueError("flux limiter unknown: %s"%limiter)
    return philim

def extrapolateToFace(fluxarray: float, xb:float, fgrad:float, order=1):
    """Extrapolate cell-centered values to left and right faces."""
    #numpy roll Rroll, put the right value to this cell
    if order == 0:
        flux_R = fluxarray
        flux_L = periodic_roll(fluxarray, 1)
    elif order == 1:    
        xdhalf = 0.5*(xb[1:]-xb[:-1])
        flux_R = fluxarray - fgrad * xdhalf
        # the following is correct in the first order case
        flux_L = periodic_roll(fluxarray + fgrad * xdhalf, 1)
    else:
        raise ValueError("order unknown: %s"%order)
    return flux_L, flux_R

def GetFQ(rho,vel,pre,gamma):
    """Return Euler fluxes and conserved densities for mass, momentum, and energy."""
    Fmass = rho * vel
    qmass = rho
    Fmom  = rho * vel* vel
    #Fmom  = rho * vel**2
    Fmom[np.logical_or(vel==0.0,np.isnan(vel))] = 0.0 * rho[0] * vel[0]**2
    Fmom += pre
    qmom  = rho * vel
    FEn   = vel * (gamma * pre / (gamma - 1.0) + 0.5 * rho * vel**2)
    qEn   = pre/(gamma-1.0) + rho * vel**2*0.5
    return Fmass, qmass, Fmom, qmom, FEn, qEn 


def CalFluxFromLR(rho_L,rho_R,u_L,u_R,p_L,p_R,gamma,cmax):
    """Calculate Rusanov/GLF fluxes from left and right primitive states."""
    Fmass_L, qmass_L, Fmom_L, qmom_L, FEn_L, qEn_L  = GetFQ(rho_L, u_L, p_L, gamma)
    Fmass_R, qmass_R, Fmom_R, qmom_R, FEn_R, qEn_R  = GetFQ(rho_R, u_R, p_R, gamma)

    Mass_flux = CalInterFaceFluxGLF(Fmass_L, Fmass_R, qmass_L, qmass_R, cmax)
    Mom_flux = CalInterFaceFluxGLF(Fmom_L, Fmom_R, qmom_L, qmom_R, cmax)
    Energy_flux = CalInterFaceFluxGLF(FEn_L, FEn_R, qEn_L, qEn_R, cmax)   
    return Mass_flux, Mom_flux, Energy_flux



def ApplyFluxLimiter(q,flux_1,flux_0):
    """Blend first-order and second-order fluxes using a minmod limiter."""
    #numpy roll Rroll, put the right value to this cell
    q_l1 = periodic_roll(q, 1)
    q_l2 = periodic_roll(q, 2)
    bottom = q - q_l1
    top = q_l1 - q_l2
    rlim = np.ones(len(q)) * 1000.0
    nonzero = bottom != 0.0
    rlim[nonzero] = np.asarray(top[nonzero] / bottom[nonzero])
    # if bottom is zero, we just assign a very large number
    rlim[np.isnan(rlim)] = 0.0
    #rlim[np.logical_or(bottom==0,np.isnan(bottom))] = 0.0
    philim = CalFluxLimiter(rlim)
    #print('philim',philim)
    return flux_0 - philim * (flux_0-flux_1), philim
