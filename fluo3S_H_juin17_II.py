# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:27:38 2017
@author: Simon
"""

from __future__ import division
import numpy as np
from bases_pivot_Gauss import *
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf
import time

N = len(LJmJmI())

##### VALEURS NUMERIQUES #####
mub = 1.3996245042      # magnéton de Bohr en MHz/G        (CODATA14)
me = 9.10938356e-31     # masse de l'électron en kg        (CODATA14)
mN = 1.672621898e-27    # masse du proton en kg            (CODATA14)
qe = 1.6021766208e-19   # charge de l'électron en C        (CODATA14)
a0 = 0.52917721067e-10  # rayon de Bohr en m               (CODATA14)
h = 6.626070040e-34     # constante de Planck en J.s       (CODATA14)
gN = 5.585694702        # facteur de Landé du noyau        (CODATA14)
ge = 2.00231930436182   # facteur de Landé de l'électron   (CODATA14)
c = 299792458           # vitesse de la lumière dans le vide en m/s
alpha = 7.2973525664e-3 # constante de structure fine      (CODATA14)
# Lamb shift en MHz (n,L) (Galtier thèse) :
LS = {(1,0):8172.840, (3,0):311.404, (3,1):0}
# Constante de couplage en MHz (n) :
A_SHF = {1:1420.405751768, 3:52.6094446}
# Facteur de Landé de l'électron (n) (Indelicato)
gS = {1:2.00228377, 3:2.0023152}
# Largeur des niveaux en MHz (n,L) :
gamma = {(1,0):0, (3,0):1.004945452, (3,1):30.192}
        
##### HAMILTONIENS #####
def H_SFHF(E0=0): # en MHz (Hagel thèse, Brodsky67, Glass thèse) 
    # base = 'LJFmF' 
    H = np.zeros((N,N))
    for n, niv1 in enumerate(LJFmF()):
        for m, niv2 in enumerate(LJFmF()):
            if n == m: # (I·J)
                H[n,m] = E(niv1.n, niv1.L, niv1.J) - E0 \
                  + (3/16)*A_SHF[niv1.n] \
                  *(niv1.F*(niv1.F+1)-niv1.J*(niv1.J+1)-niv1.I*(niv1.I+1)) \
                  /(niv1.J*(niv1.J+1)*(niv1.L+1/2))
            if niv1.L != 0 and niv1.J != niv2.J \
            and niv1.L==niv2.L and niv1.F==niv2.F and niv1.mF==niv2.mF: # (I·L)
                H[n,m] = (3/16)*A_SHF[3] \
                  *(-1)**(2*niv1.J+niv1.L+niv1.F+niv1.I+3/2) \
                  *np.sqrt((2*niv1.J+1)*(2*niv2.J+1) \
                          *(2*niv1.I+1)*(niv1.I+1)*niv1.I \
                          *(2*niv1.L+1)*(niv1.L+1)*niv1.L) \
                  *wigner6j(niv1.F,niv1.I,niv2.J,1,niv1.J,niv1.I) \
                  *wigner6j(niv1.L,niv2.J,1/2,niv1.J,niv1.L,1) \
                  /(niv1.L*(niv1.L+1)*(niv1.L+1/2))
    return H 
    
def E(n,L,J): # Dirac + recul + Lamb Shift, pour Z=1, en MHz
    mu = me*mN/(me+mN)
    epsilon = J + 1/2 - np.sqrt((J+1/2)**2-alpha**2)
    E = (mu*c**2)*(1/np.sqrt(1+(alpha/(n-epsilon))**2) - 1)
    E -= (mu**2*c**2)*alpha**4 / ((me+mN)*8*n**4)
    E *= 1e-6/h # conversion en MHz
    E += LS[(n,L)]
    return E
    
def H_Zeeman(B): # en MHz (Hagel thèse, Glass thèse)
    # base = 'LmSmLmI'
    H = np.zeros((N,N))
    for n, niv in enumerate(LmSmLmI()):
        H[n,n] = (gS[niv.n]*niv.mS + (1-me/mN)*niv.mL - gN*me/mN*niv.mI)*mub*B
        H[n,n] -= diamagnetique(niv.n,niv.L,niv.mL)*B**2
    return H
    
def diamagnetique(n,L,mL): # en MHz/G² (Delande thèse)
    r_perp_2 = n**2*(5*n**2+1-3*L*(L+1))*(L**2+L-1+mL**2)/((2*L-1)*(2*L+3))
    return r_perp_2 * qe**2*a0**2/(8*me*h) * 1e-14  # Hz/T² -> MHz/G²

def H_Stark(B): # en MHz/(km/s) (Hagel thèse, Glass thèse)
    #base = 'LJmJmI'
    H = np.zeros((N,N))
    for n, niv1 in enumerate(LJmJmI()):
        for m, niv2 in enumerate(LJmJmI()):
            if niv1.mI != niv2.mI:
                H[n,m] = 0
            else:
                H[n,m] = R(niv1.n, niv1.L, niv2.n, niv2.L) \
                        *A(niv1.L,niv1.I,niv1.J,niv1.mJ,
                           niv2.L,niv2.I,niv2.J,niv2.mJ) \
                        *a0*qe*B/h * 1e-7  # Hz/(T*m/s) -> MHz/(G*km/s)
    return H
    
def A(L1,I1,J1,mJ1,L2,I2,J2,mJ2):
    # Polarisation du champ motionnel normale à l'axe de quantification
    k, S = 1, 1/2 # ordre, spin
    return np.sum([-q*np.sin(np.pi/2)/np.sqrt(2) \
           * (-1)**(S+mJ1) \
           * np.sqrt((2*J1+1)*(2*J2+1)*(2*L1+1)*(2*L2+1)) \
           * wigner6j(J1,k,J2,L2,S,L1) \
           * wigner3j(J1,k,J2,-mJ1,q,mJ2) \
           * wigner3j(L1,k,L2,0,0,0) for q in [-1,1]]) # q = delta_mI = +-1
           
def R(n1,L1,n2,L2):
    if n1==n2 and np.abs(L1-L2)==1:
        return 3/2*n1*np.sqrt(n1**2-max(L1,L2)**2)
    else:
        return 0

def H_2photons(rabi):
    H = np.zeros((N,N),dtype=complex)       
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            if a.n!=d.n and a.L==d.L and a.mJ==d.mJ and a.mI==d.mI:
                H[i,j] = rabi
    return H
    
def convert(H,P):
    return np.dot(P,np.dot(H,P.transpose()))

##### POPULATIONS ET FLUORESCENCE #####
def matrice_densite(f=0,B=180,v=3,rabi=0.01):
    H = np.zeros((N,N),dtype=complex)
    H += convert(H_SFHF(),LJF_vers_LJI()) \
      + convert(H_Zeeman(B),LSI_vers_LJI()) \
      + H_Stark(B)*v \
      + H_2photons(rabi)
      
    for i,u in enumerate(LJFmF()):
        if getattr(u,'n')==1 and getattr(u,'mF')==1:
            E1S = H_SFHF()[i,i]
        if getattr(u,'n')==3 and getattr(u,'L')==0 and getattr(u,'mF')==1:
            E3S = H_SFHF()[i,i]
    f += (E3S - E1S)*(1 + (v*1E3)**2/(2*c**2)) # avec v en km/s

    C = np.zeros((N,N),dtype=complex)
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            C[i,j] = -1j/(4*np.pi)*(gamma[(a.n,a.L)] + gamma[(d.n,d.L)])
            if a.n==1 and d.n==3:
                C[i,j] += f
            if a.n==3 and d.n==1:
                C[i,j] -= f
                 
    A = np.zeros((N**2,N**2),dtype=complex)
    B = np.zeros(N**2,dtype=complex)  
    k = 0
    for i in range(N):
        for j in range(N):
            A_ij = np.zeros((N,N),dtype=complex)
            A_ij[:,j] = H[i,:].transpose()
            A_ij[i,:] -= H[:,j].transpose()
            A_ij[i,j] += C[i,j]
            A[k,:] = A_ij.reshape((1,N**2))
            k += 1
    for i in range(4): # si les niveaux 1S sont les 4 premiers de la base
        B[i*(N+1)] += -1j

    X = np.linalg.solve(A,B)
    return X.reshape((N,N))

def coefv(v,sigma,vo):      #(Olander70, Arnoult thèse, Galtier thèse)
    xd = 6.5e-6             # taille de la zone de détection/2 en km
    zr = 35e-6              # longueur de Rayleigh en km
    taue = 1e-6/(2*np.pi)   # durée de vie en s    
    z = v/(np.sqrt(2)*sigma)
    psi = (z*np.exp(-z**2)+np.sqrt(np.pi)/2.*(1+2*z**2)*erf(z)) \
          /(np.sqrt(2*np.pi)*z**2)
    K = 0.01
    maxwell = 4./np.sqrt(np.pi)*z**2*np.exp(-z**2)
    olander = np.sqrt(np.pi)/2.*np.sqrt(erf(psi/(2*K)))/np.sqrt(psi/(2*K))
    olivier = np.arctan((xd-v*taue)/zr)+np.arctan((xd+v*taue)/zr)
    return maxwell*olander*olivier*np.exp(-vo/v)

#def forme_de_raie(B,sigma,v0):
#    debut = time.time()
#    frequences = np.linspace(-5,5,1001)       # en MHz
#    vitesses = np.linspace(0.1,10.1,101)      # en km/s (v non nul pour coefv)
#    normalisation = quad(lambda x:coefv(x,sigma,vo),0.1,10.1)[0] 
#    fluo = np.zeros(len(frequences))
#    fluo_v = np.zeros(len(vitesses))
#    for i,delta in enumerate(frequences):
#        for j,v in enumerate(vitesses):
#            w = 'delta E 1S-3S avec LS' + v**2*nu0/(2*c**2)
#            pop = np.diag(matrice_densite(w,B,v))[4:,4:]
#            fluo_v[j] = gamma[(3,0)]*np.sum(pop[:4]) \
#                        + branch_3P*gamma[(3,1)]*np.sum(pop[4:]) \
#                        * coefv(v,sigma,v0)
#        fluo[i] = quad(interp1d(vitesses,fluo_v[:,k],kind='cubic'),0.1,10.1)[0]
#        fluo[i] *= 1/normalisation
#    print 'Calcul fini pour B =',B,', sigma =',sigma,', v0 =',vo, \
#          ', en ',int(time.time()-debut),' s'
#    return frequences,fluo*1000