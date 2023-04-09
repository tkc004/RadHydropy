import numpy as np
import unyt
from unyt import dimensions as dim

def CalPressure(rho,temp,mu):
    pressure = rho / (mu * unyt.mp) * unyt.kb * temp
    return pressure

def CalEnergyDensity(pressure, gamma):
    energydensity = pressure / (gamma-1.0)
    return energydensity

def CalSoundSpeed(pressure,rho, gamma):
    soundspeed = np.sqrt(gamma * pressure / rho)
    return soundspeed


def CheckParamDimen(params):
    unitdir = {'boxsize':1.0*unyt.pc, 'tini':1.0*unyt.yr, 'vini':1.0*unyt.pc/unyt.yr,
                'rhoini':1.0*unyt.g/unyt.cm**3, 'tempini':1.0*unyt.K, 'gamma':1.0}
    for key in unitdir: 
        if key in params.keys():
            try:
                CheckDimension(params[key],unitdir[key])
            except unyt.exceptions.UnitOperationError:
                return False
    return True


def CheckDimension(a,dimcheck):
    dummy = a+dimcheck
    pass


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def CalGradient(quan,xdelta):
    # only work for periodic boundary condition!
    dqdx = ( np.roll(quan,-1) - np.roll(quan,1) ) / (2. * xdelta)
    return dqdx

def CalInterFaceFluxGLF(flux_L: float, flux_R: float, q_L: float, q_R: float, cmax: float) -> float:
    # Global Lax Friedrich function
    # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)    
    InterFaceFlux = 0.5 * (flux_L + flux_R)
    # apply artifical diffusion +0.5*cmax*(q_L-q_R)
    InterFaceFlux += 0.5*cmax*(q_L-q_R)
    return InterFaceFlux
    
def CalFluxLimiter(rlim, limiter='minmod'):
    if limiter=='minmod':
        firststep = np.minimum(np.ones(len(rlim)), rlim)
        philim = np.maximum(np.zeros(len(rlim)), firststep)
    elif limiter=='vanLeer':
        # is it correct when rlim -> inf, philim -> 2?
        philim = (rlim + np.absolute(rlim)) / (1.0 + np.absolute(rlim))
    return philim

def extrapolateToFace(fluxarray: float, xb:float, fgrad:float, order=1):
    #numpy roll Rroll, put the right value to this cell
    Lroll = 1
    Rroll = -1
    if order == 0:
        flux_R = fluxarray
        flux_L = np.roll(fluxarray,Lroll)
    elif order == 1:    
        xdhalf = 0.5*(xb[1:]-xb[:-1])
        flux_R = fluxarray - fgrad * xdhalf
        # the following is correct in the first order case
        flux_L = np.roll(fluxarray+fgrad*xdhalf , Lroll) 
    return flux_L, flux_R

def GetFQ(rho,vel,pre,gamma):
    Fmass = rho*vel
    qmass = rho
    #Fmom  = rho * vel**2 * vel/np.absolute(vel) 
    Fmom  = rho * vel**2
    Fmom[np.logical_or(vel==0.0,np.isnan(vel))] = 0.0 * rho[0] * vel[0]**2
    Fmom += pre
    qmom  = rho * vel
    FEn   = vel * (gamma * pre / (gamma - 1.0) + 0.5 * rho * vel**2)
    qEn   = pre/(gamma-1.0) + rho * vel**2*0.5
    return Fmass, qmass, Fmom, qmom, FEn, qEn 


def CalFluxFromLR(rho_L,rho_R,u_L,u_R,p_L,p_R,gamma,cmax):
    Fmass_L, qmass_L, Fmom_L, qmom_L, FEn_L, qEn_L  = GetFQ(rho_L, u_L, p_L, gamma)
    Fmass_R, qmass_R, Fmom_R, qmom_R, FEn_R, qEn_R  = GetFQ(rho_R, u_R, p_R, gamma)

    Mass_flux = CalInterFaceFluxGLF(Fmass_L, Fmass_R, qmass_L, qmass_R, cmax)
    Mom_flux = CalInterFaceFluxGLF(Fmom_L, Fmom_R, qmom_L, qmom_R, cmax)
    Energy_flux = CalInterFaceFluxGLF(FEn_L, FEn_R, qEn_L, qEn_R, cmax)   
    return Mass_flux, Mom_flux, Energy_flux



def ApplyFluxLimiter(q,flux_1,flux_0):
    #numpy roll Rroll, put the right value to this cell
    L2roll = 2
    Lroll = 1
    Rroll = -1
    R2roll = -2
    bottom = q - np.roll(q,Lroll)
    rlim = (np.roll(q,Lroll)-np.roll(q,L2roll))/bottom
    # if bottom is zero, we just assign a very large number
    rlim[bottom==0] = 1000.0
    #rlim[np.logical_or(bottom==0,np.isnan(bottom))] = 0.0
    philim = CalFluxLimiter(rlim)
    #print('philim',philim)
    return flux_0 - philim * (flux_0-flux_1), philim