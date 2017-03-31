# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from fluo3S_H_mars2017_I import *

##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage

##### 2. Définition des hamiltoniens

mub = 1.3996245042      # magnéton de Bohr en MHz/G        (CODATA14)
me = 9.10938356e-31     # masse de l'électron en kg        (CODATA14)
mN = 1.672621898e-27    # masse du proton en kg            (CODATA14)
gN = 5.585694702        # facteur de Landé du noyau        (CODATA14)
ge = 2.00231930436182   # facteur de Landé de l'électron   (CODATA14)
qe = 1.6021766208e-19   # charge de l'électron en C        (CODATA14)
a0 = 0.52917721067e-10  # rayon de Bohr en m               (CODATA14)
h = 6.626070040e-34     # constante de Planck en J.s       (CODATA14)


class Hamiltonien:
    def __init__(self,base,H1S,H3S3P):
        self.base = base
        self.H1S, self.H3S3P = H1S, H3S3P
    
    def convert(self,P):
        if self.base is not P.base_depart:
            return False
        H1S_conv = np.dot(P.M1S,np.dot(self.H1S,P.M1S.transpose()))
        H3S3P_conv = np.dot(P.M3S3P,np.dot(self.H3S3P,P.M3S3P.transpose()))
        return Hamiltonien(P.base_arrivee,H1S_conv,H3S3P_conv)
        
    def diagonalise(self):
        self.base = 'base H0'
        self.E1S, self.M1S = np.linalg.eigh(self.H1S)
        self.H1S = np.diag(self.E1S)
        self.E3S, self.M3S = np.linalg.eigh(self.H3S3P[-4:,-4:])
        self.E3P, self.M3P = np.linalg.eigh(self.H3S3P[:-4,:-4])
#        self.E3S3P, self.M3S3P = np.linalg.eigh(self.H3S3P)
#        self.H3S3P = np.diag(self.E3S3P)
#        self.E3S = ?
#        self.E3P = ?
        self.H3S3P[-4:,-4:] = np.diag(self.E3S)
        self.H3S3P[:-4,:-4] = np.diag(self.E3P)
        self.M3S3P = np.zeros((16,16))
        self.M3S3P[-4:,-4:] = self.M3S
        self.M3S3P[:-4,:-4] = self.M3P
        self.LJF_vers_baseH0 = Passage('LJFmF',self.base,self.M1S.transpose(),
                                       self.M3S3P.transpose())     
        
    def additionner(self,H_ajoute):
        if self.base is not H_ajoute.base:
            return False
        return Hamiltonien(self.base,self.H1S + H_ajoute.H1S,
                           self.H3S3P + H_ajoute.H3S3P)
        
    def multiplier(self,v):
        return Hamiltonien(self.base,self.H1S*v,self.H3S3P*v)
        
        
def H_SHF(): # en MHz
    # (Hagel thèse, Brodsky67, Glass-Maujean thèse)
    base = 'LJFmF'
    A1S = 1420.405751768     # Écart hyperfin en MHz en champ nul (Hellwig70)
    H1S = A1S/4*np.diag([1,1,-3,1])
    freq_SF = {1/2:0, 10+1/2:-314.784, 10+3/2:2934.968} # Structure fine
    # A3S = A1S/3**3  # Sans correction de Breit : *1/n^3
    A3S = 52.6094446  # Avec correction de Breit (Galtier thèse)   
    H3S3P = np.zeros((16,16))
    for n, niv1 in enumerate(LJFmF()):
        for m, niv2 in enumerate(LJFmF()):
            # I·J :
            if n == m:
                H3S3P[n,m] = freq_SF[niv1.L*10+niv1.J] \
                  + (3/16)*A3S \
                  *(niv1.F*(niv1.F+1)-niv1.J*(niv1.J+1)-niv1.I*(niv1.I+1)) \
                  /(niv1.J*(niv1.J+1)*(niv1.L+1/2))
            # I·L :
            if niv1.L != 0 and niv1.J != niv2.J \
            and niv1.L==niv2.L and niv1.F==niv2.F and niv1.mF==niv2.mF:
                H3S3P[n,m] = (3/16)*A3S \
                  *(-1)**(2*niv1.J+niv1.L+niv1.F+niv1.I+3/2) \
                  *np.sqrt((2*niv1.J+1)*(2*niv2.J+1) \
                          *(2*niv1.I+1)*(niv1.I+1)*niv1.I \
                          *(2*niv1.L+1)*(niv1.L+1)*niv1.L) \
                  *wigner6j(niv1.F,niv1.I,niv2.J,1,niv1.J,niv1.I) \
                  *wigner6j(niv1.L,niv2.J,1/2,niv1.J,niv1.L,1) \
                  /(niv1.L*(niv1.L+1)*(niv1.L+1/2))
    return Hamiltonien(base,H1S,H3S3P)     
    
def H_Zeeman(B): # en MHz
    base = 'LmSmLmI'
    # (Glass-Maujean thèse)
    mub = 1.399601126       # magnéton de Bohr en MHz/G
    epmr = 5.446170232E-4   # electron-to-proton mass ratio
    gN = 5.585694675        # facteur de Landé du proton
    coef_diamagnetique = 1.488634644e-10  # en MHz/G²
#    coef_diamagnetique = qe**2/(8*me) * 1e2 # en MHz/G²

    H1S = np.zeros((4,4))
    gS_1S = 2.00228377
    H1S = mub*B*(1/2*gS_1S*np.diag([1,1,-1,-1]) \
             -1/2*gN*me/mN*np.diag([1,-1,1,-1]))
    H1S += -2*B**2*coef_diamagnetique*np.diag([1,1,1,1])

    H3S3P = np.zeros((16,16))
    for n, niv in enumerate(LmSmLmI()):
        gS = 2.0023152
        # Breit :
        H3S3P[n,n] = (gS*niv.mS + (1-me/mN)*niv.mL - gN*me/mN*niv.mI)*mub*B
        if niv.L is 0:
            H3S3P[n,n] += -138*coef_diamagnetique*B**2
        if niv.L is 1:
            H3S3P[n,n] += -360*coef_diamagnetique*B**2
    return Hamiltonien(base,H1S,H3S3P)
    
def H_Stark(B): # en MHz/(km/s)
    base = 'LJmJmI'
    H3S3P = np.zeros((16,16))
    for n, niv1 in enumerate(LJmJmI()):
        for m, niv2 in enumerate(LJmJmI()):
            if niv1.mI != niv2.mI:
                H3S3P[n,m] = 0
            else:
                H3S3P[n,m] = R(niv1.n, niv1.L, niv2.n, niv2.L) \
                             *A(niv1.L,niv1.I,niv1.J,niv1.mJ,
                                niv2.L,niv2.I,niv2.J,niv2.mJ) \
                             *qe*B/h*1e-5 # Hz/(T*m/s) -> MHz/(G*km/s)
    return Hamiltonien(base,np.zeros((4,4)),H3S3P)
    
def A(L1,I1,J1,mJ1,L2,I2,J2,mJ2):
    # Polarisation du champ motionnel normale à l'axe de quantification
    # q = delta_mI = +-1
    # (Hagel thèse, de Beauvoir thèse, Glass-Maujean thèse)
    k = 1    
    S = 1/2
    return np.sum([-q*np.sin(np.pi/2)/np.sqrt(2) \
           * (-1)**(S+mJ1) \
           * np.sqrt((2*J1+1)*(2*J2+1)*(2*L1+1)*(2*L2+1)) \
           * wigner6j(J1,k,J2,L2,S,L1) \
           * wigner3j(J1,k,J2,-mJ1,q,mJ2) \
           * wigner3j(L1,k,L2,0,0,0) for q in [-1,1]])
           
def R(n1,L1,n2,L2):
#    coef = 1.279544928
    if n1==n2 and np.abs(L1-L2)==1:
        return a0*3/2*n1*np.sqrt(n1**2-max(L1,L2)**2)
    else:
        return 0