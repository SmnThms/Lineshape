# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:27:38 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from bases_pivot_Gauss import *
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize._lsq import lsq_linear
N = 20

##### CONSTANTES #####
mub = 1.3996245042      # magnéton de Bohr en MHz/G        (CODATA14)
me = 9.10938356e-31     # masse de l'électron en kg        (CODATA14)
mN = 1.672621898e-27    # masse du proton en kg            (CODATA14)
qe = 1.6021766208e-19   # charge de l'électron en C        (CODATA14)
a0 = 0.52917721067e-10  # rayon de Bohr en m               (CODATA14)
h = 6.626070040e-34     # constante de Planck en J.s       (CODATA14)
gN = 5.585694702        # facteur de Landé du noyau        (CODATA14)
ge = 2.00231930436182   # facteur de Landé de l'électron   (CODATA14)
gS_1S = 2.00228377      # facteur de Landé de l'électron   (Indelicato)
gS_3S = 2.0023152       # facteur de Landé de l'électron   (Indelicato)
c = 299792458           # vitesse de la lumière dans le vide en m/s
alpha = 7.2973525664e-3 # constante de structure fine      (CODATA14)


##### NIVEAUX D'ÉNERGIE #####
# Structure fine en MHz (n,L,J):
SF = {(1,0,1/2):1, (3,0,1/2):0, (3,1,1/2):-314.784, (3,1,3/2):2934.968}
# Lamb shift en MHz (n,L) (Galtier thèse) :
LS = {(1,0):8172.840, (3,0):311.404, (3,1):0}
# Constante de couplage (n) :
A_SHF = {1:1420.405751768, 3:52.6094446}
# Facteur de Landé de l'électron
gS = {1:2.00228377, 3:2.0023152}
# Largeur des niveaux (MHz) :
gamma = {(1,0):0, (3,0):1.004945452, (3,1):30.192}
branch_3P = 0.11834

nu0 = 2922742937 # en Mhz (pourrait tout aussi bien être 2922742900)    

#A1S = 1420.405751768    # Écart hyperfin en MHz en champ nul (Hellwig70)
# A3S = A1S/3**3        # Sans correction de Breit : *1/n^3
#A3S = 52.6094446        # Avec correction de Breit (Galtier thèse) 
#gamma3S = 1.004945452                    # en MHz
#gamma3P = 30.192                         # en MHz
#gamma3P_12 = 30.19175875                 # en MHz
#gamma3P_32 = 30.19165419                 # en MHz

##### CLASSE #####
#class Hamiltonien:
#    def __init__(self,base,H):
#        self.base = base
#        self.H = H
#    
#    def convert(self,P):
##        if self.base is not P.base_depart:
##            return False
#        H1S_conv = np.dot(P.M1S,np.dot(self.H1S,P.M1S.transpose()))
#        H3S3P_conv = np.dot(P.M3S3P,np.dot(self.H3S3P,P.M3S3P.transpose()))
#        return Hamiltonien(P.base_arrivee,H1S_conv,H3S3P_conv)
#        
#    def diagonaliser(self):
#        self.base = 'base H0'
#        self.E1S, self.M1S = np.linalg.eigh(self.H1S)
#        self.H1S = np.diag(self.E1S)
#        self.E3S, self.M3S = np.linalg.eigh(self.H3S3P[-4:,-4:])
#        self.E3P, self.M3P = np.linalg.eigh(self.H3S3P[:-4,:-4])
#        self.E3S3P = np.concatenate((self.E3S,self.E3P))
#        self.H3S3P[-4:,-4:] = np.diag(self.E3S)
#        self.H3S3P[:-4,:-4] = np.diag(self.E3P)
#        self.M3S3P = np.zeros((16,16))
#        self.M3S3P[-4:,-4:] = self.M3S
#        self.M3S3P[:-4,:-4] = self.M3P
#        self.LJF_vers_baseH0 = Passage('LJFmF',self.base,self.M1S.transpose(),
#                                       self.M3S3P.transpose())     
#        
#    def additionner(self,H_ajoute):
##        if self.base is not H_ajoute.base:
##            return False
#        return Hamiltonien(self.base,self.H1S + H_ajoute.H1S,
#                           self.H3S3P + H_ajoute.H3S3P)
#                           
#    def energies(self):
#        self.E1S = np.diag(self.H1S)
#        self.E3S = np.diag(self.H3S3P[-4:,-4:])
#        self.E3P = np.diag(self.H3S3P[:-4,:-4])
        
##### CALCUL DES 3 HAMILTONIENS #####
def H_SHF(): # en MHz (Hagel thèse, Brodsky67, Glass thèse) 
    # base = 'LJFmF' 
    H = np.zeros((N,N))
    for n, niv1 in enumerate(LJFmF()):
        for m, niv2 in enumerate(LJFmF()):
            if n == m: # (I·J)
                H[n,m] = E(niv1.n, niv1.L, niv1.J) \
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
    H = np.zeros((N,N))       
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            if a.n!=d.n and a.L==d.L and a.mJ==d.mJ and a.mI==d.mI:
                H[i,j] = rabi
    return H
    
def convert(H,P):
    return np.dot(P,np.dot(H,P.transpose()))

##### CALCUL #####
def populations(w,B,v,rabi=1): # Résolution de [H,rho] + ((C_ij*rho_ij)) = 0

    # Hamiltonien
    H = convert(H_SHF(),LJF_vers_LJI()) \
      + convert(H_Zeeman(B),LSI_vers_LJI()) \
      + H_Stark(B)*v \
      + H_2photons(rabi)  
    
    # Coefs de l'équation    
    C = np.zeros((N,N),dtype=complex)
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            C[i,j] = (1j/(2*np.pi))*(gamma[(a.n,a.L)] + gamma[(d.n,d.L)])/2
            if a.n != d.n:
                C[i,j] += w*np.sign(j-i)
    
    # Mise sous la forme AX=B
    A = np.zeros((N**2,N**2),dtype=complex)
    k = 0
    for i in range(N):
        for j in range(N):
            A_ij = np.zeros((N,N),dtype=complex)
            A_ij[:,j] = H[i,:].transpose()
            A_ij[i,:] = -H[:,j].transpose()
            A_ij[i,j] = C[i,j]
            A[k,:] = A_ij.reshape((1,N**2))
            k += 1
    B = np.zeros((N**2,1))
       
    # Résolution
    X, residuals, rank, s = np.linalg.lstsq(A,B)
#    rho = X.reshape((N,N))
    print rank
    return X.reshape((N,N))
          
def X0():
    x0 = np.zeros((N**2,1))
    for i in range(4):
        x0[i*N + i,0] = 1/2
    return x0         
        
def matrice_densite(w=2922742937,B=170,v=5,rabi=1):
    # Résolution de [H,rho] + ((C_ij*rho_ij)) = 0
    # Hamiltonien
    H = convert(H_SHF(),LJF_vers_LJI()) \
      + convert(H_Zeeman(B),LSI_vers_LJI()) \
      + H_Stark(B)*v \
      + H_2photons(rabi)  
    # Coefs de l'équation    
    C = np.zeros((N,N),dtype=complex)
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            C[i,j] = (1j/(2*np.pi))*(gamma[(a.n,a.L)] + gamma[(d.n,d.L)])/2
            if a.n != d.n:
                C[i,j] += w*np.sign(j-i)
    # Mise sous la forme AX=B
    A = np.zeros((N**2+1,N**2),dtype=complex)
    B = np.zeros((N**2+1,1))
    # Première équation : somme des populations = 1
    for i in range(N):
        A[0,i*(N+1)] = 1
    B[0,0] = 1
    # N**2 équations de Bloch optiques    
    k = 1
    for i in range(N):
        for j in range(N):
            A_ij = np.zeros((N,N),dtype=complex)
            A_ij[:,j] = H[i,:].transpose()
            A_ij[i,:] = -H[:,j].transpose()
            A_ij[i,j] = C[i,j]
            A[k,:] = A_ij.reshape((1,N**2))
            k += 1
    # Résolution numérique de AX-B -> 0
    bounds1 = np.array([-np.inf]*N**2)
    for i in range(N):
        bounds1[i*(N+1)] = 0
    bounds2 = np.array([np.inf]*N**2)
    x = lsq_linear(A,B,bounds=(bounds1,bounds2))
    return x
#    X, residuals, rank, s = np.linalg.lstsq(A,B)
#    return X.reshape((N,N))
    
rho = matrice_densite()
popul = np.diag(rho)
    
#xx = leastsq(equation_function,X0())
        
def forme_de_raie(B,sigma,vo=0):
    debut = time.time()
    H0 = H_SHF().additionner(H_Zeeman(B).convert(LSI_vers_LJI()) \
                .convert(LJI_vers_LJF()))
    H0.diagonaliser() # On se place désormais dans la base propre de H0
    H0_B0 = H_SHF().additionner(H_Zeeman(0).convert(LSI_vers_LJI()) \
                .convert(LJI_vers_LJF())) 
    H0_B0.diagonaliser()
    hfs = [1,2,3]     # [mF=1, mF=0, mF=-1]
    if B > 20:
        hfs = [1,3]   # [mF=1, mF=-1]
    matrice_Stark = H_Stark(B).convert(LJI_vers_LJF()) \
                        .convert(H0.LJF_vers_baseH0).H3S3P[:-4,12:]
    frequences = np.linspace(-5,5,1001)       # en MHz
    vitesses = np.linspace(0.1,10.1,101)      # en km/s (v non nul pour coefv)
    normalisation = quad(lambda x:coefv(x,sigma,vo),0.1,10.1)[0] 
    fluo = np.zeros(len(frequences))
    fluo_v = np.zeros((len(vitesses),len(H0.E3S)))
    for i,delta in enumerate(frequences):
        for j,v in enumerate(vitesses):
            coef_Stark = v*matrice_Stark      # en MHz
            coef_Doppler = v**2*nu0/(2*c**2)  # en MHz ATTENTION c en m/s
            for k in hfs: 
                ecart = H0.E1S[k] - H0_B0.E1S[k] + delta + coef_Doppler \
                        - H0.E3S[k] + H0_B0.E3S[k]
                coupl = np.sum(coef_Stark[:,k]**2 \
                        /(-gamma3P/2 + 1j*(ecart + H0.E3S[k] - H0.E3P)))
                BB = -coef_Stark[:,k]**2*(gamma3S+gamma3P) \
                     /((H0.E3S[k]-H0.E3P)**2+((gamma3S+gamma3P)/2)**2)
                A = coupl - gamma3S/2 + 1j*ecart
                CC = np.real(coef_Stark[:,k]**2 \
                     /(A*(ecart+H0.E3S[k]-H0.E3P+1j*gamma3P/2) \
                     *(H0.E3S[k]-H0.E3P+1j*(gamma3S+gamma3P)/2)))
                num = np.real(-1/A) - np.sum(CC*(1+BB/(gamma3P-BB)))
                den = gamma3S - np.sum(BB*(1+BB/(gamma3P-BB)))
                pop3S = num/den
                pop3P = (CC-pop3S*BB)/(gamma3P-BB)
                fluo_v[j,k] = gamma3S*pop3S + branch_3P*np.dot(gamma3P,pop3P)
                fluo_v[j,k] *= coefv(v,sigma,vo)
        fluo[i] = np.sum([quad(interp1d(vitesses,fluo_v[:,k],kind='cubic'), \
                               0.1,10.1)[0]/normalisation for k in hfs])
    print 'Calcul fini pour B =',B,', sigma =',sigma,', v0 =',vo, \
          ', en ',int(time.time()-debut),' s'
    return frequences,fluo*1000    
    
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
    
def test(M,fig=0):
    if np.max(M)<1E-12:
        print 'Matrice nulle'
    else:
        if fig is 0:
            plt.close('all')
        else:
            plt.figure(fig)
        plt.imshow(M, interpolation='nearest',cmap=plt.jet())
        plt.colorbar()
        
#populations(2922742937,170,5,100)