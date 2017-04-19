# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from fluo3S_H_avr17_I import *

##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage

##### 2. Définition des hamiltoniens

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

##### NIVEAUX D'ÉNERGIE #####
A1S = 1420.405751768    # Écart hyperfin en MHz en champ nul (Hellwig70)
SF = {1/2:0, 10+1/2:-314.784, 10+3/2:2934.968} # Structure fine
# A3S = A1S/3**3        # Sans correction de Breit : *1/n^3
A3S = 52.6094446        # Avec correction de Breit (Galtier thèse) 

##### CLASSE #####
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
        
    def diagonaliser(self):
        self.base = 'base H0'
        self.E1S, self.M1S = np.linalg.eigh(self.H1S)
        self.H1S = np.diag(self.E1S)
        self.E3S, self.M3S = np.linalg.eigh(self.H3S3P[-4:,-4:])
        self.E3P, self.M3P = np.linalg.eigh(self.H3S3P[:-4,:-4])
        self.E3S3P = np.concatenate((self.E3S,self.E3P))
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
        
##### CALCUL DES 3 HAMILTONIENS #####
def H_SHF(): # en MHz (Hagel thèse, Brodsky67, Glass thèse) 
    base = 'LJFmF' 
    H1S = A1S/4*np.diag([1,1,-3,1])
    H3S3P = np.zeros((16,16))
    for n, niv1 in enumerate(LJFmF()):
        for m, niv2 in enumerate(LJFmF()):
            if n == m: # (I·J)
                H3S3P[n,m] = SF[niv1.L*10+niv1.J] \
                  + (3/16)*A3S \
                  *(niv1.F*(niv1.F+1)-niv1.J*(niv1.J+1)-niv1.I*(niv1.I+1)) \
                  /(niv1.J*(niv1.J+1)*(niv1.L+1/2))
            if niv1.L != 0 and niv1.J != niv2.J \
            and niv1.L==niv2.L and niv1.F==niv2.F and niv1.mF==niv2.mF: # (I·L)
                H3S3P[n,m] = (3/16)*A3S \
                  *(-1)**(2*niv1.J+niv1.L+niv1.F+niv1.I+3/2) \
                  *np.sqrt((2*niv1.J+1)*(2*niv2.J+1) \
                          *(2*niv1.I+1)*(niv1.I+1)*niv1.I \
                          *(2*niv1.L+1)*(niv1.L+1)*niv1.L) \
                  *wigner6j(niv1.F,niv1.I,niv2.J,1,niv1.J,niv1.I) \
                  *wigner6j(niv1.L,niv2.J,1/2,niv1.J,niv1.L,1) \
                  /(niv1.L*(niv1.L+1)*(niv1.L+1/2))
    return Hamiltonien(base,H1S,H3S3P)     
    
def H_Zeeman(B): # en MHz (Hagel thèse, Glass thèse)
    base = 'LmSmLmI'
    H1S = np.zeros((4,4))
    H1S = mub*B*(1/2*gS_1S*np.diag([1,1,-1,-1]) \
             -1/2*gN*me/mN*np.diag([1,-1,1,-1])) \
             -diamagnetique(1,0,0)*B**2*np.diag([1,1,1,1])
    H3S3P = np.zeros((16,16))
    for n, niv in enumerate(LmSmLmI()):
        H3S3P[n,n] = (gS_3S*niv.mS + (1-me/mN)*niv.mL - gN*me/mN*niv.mI)*mub*B
        H3S3P[n,n] -= diamagnetique(niv.n,niv.L,niv.mL)*B**2
    return Hamiltonien(base,H1S,H3S3P)
    
def diamagnetique(n,L,mL): # en MHz/G² (Delande thèse)
    r_perp_2 = n**2*(5*n**2+1-3*L*(L+1))*(L**2+L-1+mL**2)/((2*L-1)*(2*L+3))
    return r_perp_2 * qe**2*a0**2/(8*me*h) * 1e-14  # Hz/T² -> MHz/G²

def H_Stark(B): # en MHz/(km/s) (Hagel thèse, Glass thèse)
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
                             *a0*qe*B/h * 1e-7  # Hz/(T*m/s) -> MHz/(G*km/s)
    return Hamiltonien(base,np.zeros((4,4)),H3S3P)
    
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