import numpy as np
from scipy import special

# analytic Sedov blast wave solution
# see Sedov 1993 ``Similarity and dimensional methods in mechanics'' 
# Korobeinikov et al. 1961 for power law density profile
# We use the version of Book 1994 Shock Waves 4:1-10
# rho0 = A0 r^-w 
# Rs = (E t^2 / (alpha A0))^(1/(nu+2-w))
# g is adiabatic index
# nu is dimension of the problem


def get_rho0(r,A0,w):
    # we assume a power law density profile:
    # rho0 = A0 r^-w
    # note that the dimenion of rho0 is [M/L^nu], is the dimension of the problem 
    # Thus, A0 dimension is [M L^(w-nu)]
    rho0 = A0 * np.power(r, -w)    
    return rho0

def get_wa(nu,g):
    # Get wa of solution
    # nu is dimension of the problem
    # g is adiabatic index 
    wa = np.zeros(6)
    wa[1] = (3.0 * nu - 2.0 + g * (2.0 - nu)) / (g + 1)
    wa[2] = (2.0 * (g - 1) + nu) / g
    wa[3] = nu * (2.0 - g)
    wa[4] = nu/g
    wa[5] = 2.0 * nu / (g + 1.0)
    return wa

def get_beta_index(nu,w,g,wa):
    # get the power law index for the analytic blastwave solution
    # nu is dimension of the problem
    # w is the exponent of initial density profile from rho0 = A r^-w  
    # g is the adiabatic index
    # wa are some parameters of solution 
    b = np.zeros(9)
    b[0] = 1.0/ (nu * g - nu + 2.0)
    b[2] = (g-1)/(g* (wa[2] - w))
    b[3] = (nu - w) / (g * (wa[2] - w))
    b[5] = (2.0 * nu - w * (g+1)) / (wa[3] - w)
    b[6] = 2.0 / (nu + 2.0 - w)
    b[1] = b[2] + (g+1) * b[0] - b[6]
    b[4] = b[1] * (nu - w) * (nu + 2.0 - w) / (wa[3] - w)
    b[7] = w * b[6]
    b[8] = nu * b[6]
    return b


def get_Cc(nu,w,g,wa,b):
    # nu is dimension of the problem
    # w is the exponent of initial density profile from rho0 = A r^-w  
    # g is the adiabatic index
    # wa are some parameters of solution 
    # b is power law index for the analytic blastwave solution 
    Cc = np.zeros(7)
    #Cc[0] = 2.0 * (nu - 1.0) * np.pi + (nu - 2.0) * (nu - 3.0)
    Cc[0] = 2.0**nu * np.pi**(0.5*(nu-1.0)) * special.gamma(0.5*(nu+1.0)) / special.gamma(nu)
    Cc[5] = 2.0 / (g-1.0)
    Cc[6] = (g+1.0) / 2.0
    Cc[1] = g * Cc[5]
    Cc[2] = Cc[6] / g
    Cc[3] = (nu * g - nu + 2.0) / (wa[1] - w) / Cc[6]
    Cc[4] = (nu + 2.0 - w) * b[0] * Cc[6]
    return Cc


def getShockquan(g, nu, w, A0, Rs, t):
    # density, velocity and pressure of the shock
    # g is the adiabatic index
    # nu is dimension of the problem
    # w is the exponent of initial density profile from rho0 = A0 r^-w  
    Rsdot = 2.0 * Rs / (nu + 2.0 - w) / t #dRs/dt
    rho0 = get_rho0(Rs,A0,w) # TK: rho0 should be the pre-shock density at shock radius?
    rhos = (g + 1.0) / (g - 1.0) * rho0 
    vs = 2.0 * Rsdot / (g + 1.0)
    ps = 2.0 * rho0 * Rsdot **2 / (g + 1.0)
    return rhos, vs, ps


def getRs(E0,A0,nu,w,alpha,t):
    Rs = np.power(E0 * t**2 / alpha / A0, 1.0/(nu + 2.0 - w))
    return Rs

def eta_func(F,b,Cc):
    eta = np.power(F,-b[6]) * np.power(Cc[1] * (F - Cc[2]), b[2]) * np.power(Cc[3]*(Cc[4] - F),-b[1])
    return eta

def D_func(F,b,Cc,w):
    Df = np.power(F, b[7]) * np.power(Cc[1]*(F-Cc[2]), b[3]-w*b[2]) * np.power(Cc[3]*(Cc[4]-F), b[4]+w*b[1]) * np.power(Cc[5]*(Cc[6] - F), -b[5])
    return Df

def V_func(F,b,Cc):
    eta = eta_func(F,b,Cc)
    Vf = eta * F
    return Vf

def P_func(F,b,Cc,w):
    Pf = np.power(F, b[8]) * np.power(Cc[3]*(Cc[4] - F), b[4]+(w-2.0)*b[1]) * np.power(Cc[5]*(Cc[6]-F), 1.0 - b[5])
    return Pf

def integral_solution(nu,g,w):
    wa = get_wa(nu,g)
    b = get_beta_index(nu,w,g,wa)
    Cc = get_Cc(nu,w,g,wa,b)
    if w < wa[1]:
        Fmin = Cc[2]  
    else:
        Fmin = Cc[6]
    F = np.linspace(Fmin,1.0,10000)
    eta = eta_func(F,b,Cc)
    Df = D_func(F,b,Cc,w)
    Vf = V_func(F,b,Cc)
    Pf = P_func(F,b,Cc,w)
    deta_dF = np.gradient(eta,F)
    Integrant = np.power(eta, nu-1.0) * (Df * Vf * Vf + Pf) * deta_dF
    Integrated_value = np.trapz(Integrant,F)
    alpha = 8.0 * Cc[0] / (g**2 - 1.0) / (nu + 2.0 + w)**2 * Integrated_value 
    return alpha, eta, Df, Vf, Pf

def get_blastwave_solution(E0,A0,nu,g,w,t):
    alpha, eta, Df, Vf, Pf = integral_solution(nu,g,w) 
    Rs = getRs(E0,A0,nu,w,alpha,t)
    rhos, vs, ps = getShockquan(g, nu, w, A0, Rs, t)
    r = eta * Rs
    rho = rhos * Df
    v = vs * Vf
    p = ps * Pf
    return r, rho, v, p, Rs







