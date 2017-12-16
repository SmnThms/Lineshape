# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017 @author: Simon
"""

from __future__ import division
import numpy as np
from scipy.misc import factorial

##### STRUCTURE DU PROGRAMME #####
# I.   Définition des bases et matrices de passage
# II.  Définition des hamiltoniens, calcul matrice densité et forme de raie
# III. Exploitation (ajustement, enregistrement, affichage)

##### STRUCTURE DU FICHIER #######
# class Niveau
# bases LmSmLmI, LJmJmI, LJFmF (n=1&3) et LJmJ (n=1&2&3)
# matrices de changement de base
# Clebsh, 3j, 6j

class Niveau:
    def __init__(self,n=False, S=False, L=False, I=False, J=False, F=False, 
                 mS=False, mL=False, mI=False, mJ=False, mF=False):
        self.n, self.S, self.L, self.I, self.J, self.F = n, S, L, I, J, F
        self.mS, self.mL, self.mI, self.mJ, self.mF = mS, mL, mI, mJ, mF

def LmSmLmI():
    LmSmLmI = []
    LmSmLmI.append(Niveau( n=1, L=0, mS=1/2,  mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=1, L=0, mS=1/2,  mL=0,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=1, L=0, mS=-1/2, mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=1, L=0, mS=-1/2, mL=0,  mI=-1/2, S=1/2 ))

    LmSmLmI.append(Niveau( n=3, L=0, mS=1/2,  mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=0, mS=1/2,  mL=0,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=0, mS=-1/2, mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=0, mS=-1/2, mL=0,  mI=-1/2, S=1/2 ))
    
    LmSmLmI.append(Niveau( n=3, L=1, mS=1/2,  mL=1,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=1/2,  mL=1,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=1/2,  mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=-1/2, mL=1,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=1/2,  mL=0,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=1/2,  mL=-1, mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=-1/2, mL=1,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=-1/2, mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=1/2,  mL=-1, mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=-1/2, mL=0,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=-1/2, mL=-1, mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=1, mS=-1/2, mL=-1, mI=-1/2, S=1/2 ))
    return LmSmLmI

def LJmJmI():
    LJmJmI = []
    LJmJmI.append(Niveau( n=1, L=0, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=1, L=0, J=1/2, mJ=1/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=1, L=0, J=1/2, mJ=-1/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=1, L=0, J=1/2, mJ=-1/2, mI=-1/2, I=1/2 ))

    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=1/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=-1/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=-1/2, mI=-1/2, I=1/2 ))
        
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=3/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=3/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=1/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=1/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=-1/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=-1/2, mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=-3/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=-3/2, mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=1/2, mJ=1/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=1/2, mJ=-1/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=1/2, mJ=-1/2, mI=-1/2, I=1/2 ))
    return LJmJmI
    
def LJFmF():    
    LJFmF = []
    LJFmF.append(Niveau( n=1, L=0, J=1/2, F=1, mF=1,  I=1/2 ))
    LJFmF.append(Niveau( n=1, L=0, J=1/2, F=1, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=1, L=0, J=1/2, F=0, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=1, L=0, J=1/2, F=1, mF=-1, I=1/2 ))
    
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=1, mF=1,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=1, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=0, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=1, mF=-1, I=1/2 ))

    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=2,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=1,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=1, mF=1,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=1, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=-1, I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=1, mF=-1, I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=-2, I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=1, mF=1,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=1, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=0, mF=0,  I=1/2 ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=1, mF=-1, I=1/2 ))
    return LJFmF

def LJmJmI2():
    LJmJmI = []
    LJmJmI.append(Niveau( n=1, L=0, J=1/2, mJ=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=1, L=0, J=1/2, mJ=-1/2, I=1/2 ))
    
    LJmJmI.append(Niveau( n=2, L=0, J=1/2, mJ=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=2, L=0, J=1/2, mJ=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=2, L=1, J=3/2, mJ=3/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=2, L=1, J=3/2, mJ=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=2, L=1, J=3/2, mJ=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=2, L=1, J=3/2, mJ=-3/2, I=1/2 ))
    LJmJmI.append(Niveau( n=2, L=1, J=1/2, mJ=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=2, L=1, J=1/2, mJ=-1/2, I=1/2 ))

    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=3/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=3/2, mJ=-3/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=1/2, mJ=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=1, J=1/2, mJ=-1/2, I=1/2 ))
    return LJmJmI
    
def LSI_vers_LJI():
    P = np.zeros((20,20))
    for m,d in enumerate(LmSmLmI()): # départ
        for n,a in enumerate(LJmJmI()): # arrivée
            if d.L != a.L or d.mI != a.mI or d.n != a.n:
                P[n,m] = 0 # Mêmes mI et L pour les deux bases
            else:
                P[n,m] = clebsch(j1=d.L,m1=d.mL,j2=d.S,m2=d.mS,J=a.J,M=a.mJ)
    return P
    
def LJI_vers_LSI():
    return LSI_vers_LJI().transpose()
            
def LJI_vers_LJF():
    P = np.zeros((20,20))
    for m,d in enumerate(LJmJmI()): # départ
        for n,a in enumerate(LJFmF()): # arrivée
            if d.L != a.L or d.J != a.J or d.n != a.n:
                P[n,m] = 0 # Mêmes J et L pour les deux bases
            else:
                P[n,m] = clebsch(j1=d.J,m1=d.mJ,j2=d.I,m2=d.mI,J=a.F,M=a.mF)
    return P
    
def LJF_vers_LJI():
    return LJI_vers_LJF().transpose()
    
def clebsch(j1,m1,j2,m2,J,M):
    return (-1)**(j1-j2+M)*np.sqrt(2*J+1)*wigner3j(j1,j2,J,m1,m2,-M)
    
def wigner3j(j1,j2,j3,m1,m2,m3):
    if m1+m2+m3!=0:
        return 0
    if j1-m1!=np.floor(j1-m1) or j2-m2!=np.floor(j2-m2) \
                              or j3-m3!=np.floor(j3-m3):
        return 0
    if j3>j1+j2 or j3<abs(j1-j2):
        return 0
    t1 = j2 - m1 - j3
    t2 = j1 + m2 - j3
    t3 = j1 + j2 - j3
    t4 = j1 - m1
    t5 = j2 + m2
    tmin = max(0, max(t1,t2))
    tmax = min(t3, min(t4,t5))
    tvec = np.arange(tmin,tmax+1,1)
    wigner = 0
    for t in tvec:
        wigner += (-1)**t/( factorial(t) * factorial(t-t1) * factorial(t-t2) \
                  * factorial(t3-t) * factorial(t4-t) * factorial(t5-t) )
    return wigner * (-1)**(j1-j2-m3) * np.sqrt( factorial(j1+j2-j3) \
           *factorial(j1-j2+j3)*factorial(-j1+j2+j3) / factorial(j1+j2+j3+1) \
           *factorial(j1+m1)*factorial(j1-m1)*factorial(j2+m2) \
           *factorial(j2-m2)*factorial(j3+m3)*factorial(j3-m3) )
           
def wigner6j(j1,j2,j3,J1,J2,J3):
    if abs(j1-j2)>j3 or j1+j2<j3 or abs(j1-J2)>J3 or j1+J2<J3 \
    or abs(J1-j2)>J3 or J1+j2<J3 or abs(J1-J2)>j3 or J1+J2<j3:
        return 0
    if 2*(j1+j2+j3)!=round(2*(j1+j2+j3)) or 2*(j1+J2+J3)!=round(2*(j1+J2+J3)) \
    or 2*(J1+j2+J3)!=round(2*(J1+j2+J3)) or 2*(J1+J2+j3)!=round(2*(J1+J2+j3)):
        return 0
    t1 = j1+j2+j3
    t2 = j1+J2+J3
    t3 = J1+j2+J3
    t4 = J1+J2+j3
    t5 = j1+j2+J1+J2
    t6 = j2+j3+J2+J3
    t7 = j1+j3+J1+J3
    tmin = max(0, max(t1, max(t2, max(t3,t4))))
    tmax = min(t5, min(t6,t7))
    tvec = np.arange(tmin,tmax+1,1)
    wigner = 0
    for t in tvec:
        wigner += (-1)**t*factorial(t+1)/(factorial(t-t1)*factorial(t-t2) \
                  *factorial(t-t3)*factorial(t-t4)*factorial(t5-t) \
                  *factorial(t6-t)*factorial(t7-t))
    return wigner*np.sqrt(TriaCoeff(j1,j2,j3)*TriaCoeff(j1,J2,J3) \
                 *TriaCoeff(J1,j2,J3)*TriaCoeff(J1,J2,j3))

def TriaCoeff(a,b,c):
    return factorial(a+b-c)*factorial(a-b+c)*factorial(-a+b+c) \
           /(factorial(a+b+c+1))