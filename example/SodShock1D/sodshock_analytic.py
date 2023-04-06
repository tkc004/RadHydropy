import numpy as np
from scipy.optimize import fsolve
import unyt


def shocktubecal(gamma, rho1, rho5, p1, p5):
    # p5 and rho5 are higher than p1 and rho1
    # see Pfrommer et al. 2006 for reference
    mu2 = (gamma-1.)/(gamma+1.)
    c1 = np.sqrt(gamma*p1/rho1)
    c5 = np.sqrt(gamma*p5/rho5)
    func = lambda p2 : (p2/p1-1.)*np.sqrt((1.-mu2)/gamma/(p2/p1+mu2))-2./(gamma-1.)*c5/c1*(1.-np.power(p2/p5, (gamma-1.)/2./gamma))
    p2_initial_guess = 0.4
    p2_solution = fsolve(func, p2_initial_guess)
    p2 = p2_solution[0]
    rho3 = rho5*np.power(p2/p5, 1./gamma)
    rho2 = rho1*(p2+mu2*p1)/(p1+mu2*p2)
    v2 = 2.*c5/(gamma-1.)*(1.-np.power(p2/p5, (gamma-1.)/2./gamma))
    vt = c5 - v2/(1.-mu2)
    vs = v2/(1.-rho1/rho2)
    Mach = vs/c1
    return rho2, rho3, p2, v2, vt, vs, Mach



def shocktubeanalyticgraph(gamma, rho1, rho2, rho3, rho5, p1, p2, p5, v2, vt, vs, time, xcor, xint):
    # p5 and rho5 are higher than p1 and rho1
    # assume the initial interface located at xint
    mu2 = (gamma-1.)/(gamma+1.)
    c1 = np.sqrt(gamma*p1/rho1)
    c5 = np.sqrt(gamma*p5/rho5)
    xnor = np.array(xcor)-xint
    rho_ana=np.zeros(len(xnor))
    p_ana=np.zeros(len(xnor))
    v_ana=np.zeros(len(xnor))
    logical5 = xnor<-c5*time
    logical4 = np.logical_and(xnor>-c5*time, xnor<-vt*time)
    logical3 = np.logical_and(xnor>-vt*time, xnor<v2*time)
    logical2 = np.logical_and(xnor>v2*time, xnor<vs*time)
    logical1 = xnor>vs*time
    xnor4=xnor[logical4]
    rho_ana[logical5]=rho5
    rho_ana[logical4]=rho5*np.power(-mu2*xnor4/c5/time+(1.-mu2), 2./(gamma-1.))
    rho_ana[logical3]=rho3
    rho_ana[logical2]=rho2
    rho_ana[logical1]=rho1
    p_ana[logical5]=p5
    p_ana[logical4]=p5*np.power(-mu2*xnor4/c5/time+(1.-mu2), 2.*gamma/(gamma-1.))
    p_ana[logical3]=p2
    p_ana[logical2]=p2
    p_ana[logical1]=p1
    v_ana[logical5]=0.0
    v_ana[logical4]=(1.0-mu2)*(xnor4/time+c5)
    v_ana[logical3]=v2
    v_ana[logical2]=v2
    v_ana[logical1]=0.0
    return rho_ana, p_ana, v_ana