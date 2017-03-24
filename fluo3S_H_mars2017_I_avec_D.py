# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from scipy.misc import factorial
from fluo3S_H_mars2017_IV import *

##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage

##### 1. Définition des bases et matrices de passage

class Niveau:
    def __init__(self,n=False, S=False, L=False, I=False, J=False, F=False, 
                 mS=False, mL=False, mI=False, mJ=False, mF=False):
        self.n, self.S, self.L, self.I, self.J, self.F = n, S, L, I, J, F
        self.mS, self.mL, self.mI, self.mJ, self.mF = mS, mL, mI, mJ, mF

def LmSmLmI():
    LmSmLmI = []
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

    LmSmLmI.append(Niveau( n=3, L=0, mS=1/2,  mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=0, mS=1/2,  mL=0,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=0, mS=-1/2, mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=0, mS=-1/2, mL=0,  mI=-1/2, S=1/2 ))

    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=-2, mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=-2, mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=-2, mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=-2, mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=-1, mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=-1, mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=-1, mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=-1, mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=0,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=0,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=0,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=1,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=1,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=1,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=1,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=2,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=-1/2, mL=2,  mI=1/2,  S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=2,  mI=-1/2, S=1/2 ))
    LmSmLmI.append(Niveau( n=3, L=2, mS=1/2,  mL=2,  mI=1/2,  S=1/2 ))
    return LmSmLmI

def LJmJmI():
    LJmJmI = []
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
    
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=1/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=-1/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=0, J=1/2, mJ=-1/2, mI=-1/2, I=1/2 ))
    
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=-3/2, mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=-3/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=-1/2, mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=-1/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=1/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=1/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=3/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=3/2, mJ=3/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=-5/2, mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=-5/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=-3/2, mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=-3/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=-1/2, mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=-1/2, mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=1/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=1/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=3/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=3/2,  mI=1/2,  I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=5/2,  mI=-1/2, I=1/2 ))
    LJmJmI.append(Niveau( n=3, L=2, J=5/2, mJ=5/2,  mI=1/2,  I=1/2 ))
    return LJmJmI
    
def LJFmF():    
    LJFmF = []
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=2  ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=1  ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=1, mF=1  ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=0  ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=1, mF=0  ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=-1 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=1, mF=-1 ))
    LJFmF.append(Niveau( n=3, L=1, J=3/2, F=2, mF=-2 ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=1, mF=1  ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=1, mF=0  ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=0, mF=0  ))
    LJFmF.append(Niveau( n=3, L=1, J=1/2, F=1, mF=-1 ))
   
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=1, mF=1  ))
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=1, mF=0  ))
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=0, mF=0  ))
    LJFmF.append(Niveau( n=3, L=0, J=1/2, F=1, mF=-1 ))

    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=2, mF=-2 ))
    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=2, mF=-1 ))
    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=2, mF=0  ))
    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=2, mF=1  ))
    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=2, mF=2  ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=3, mF=-3 ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=3, mF=-2 ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=3, mF=-1 ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=3, mF=0  ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=3, mF=1  ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=3, mF=2  ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=3, mF=3  ))
    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=1, mF=-1 ))
    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=1, mF=0  ))
    LJFmF.append(Niveau( n=3, L=2, J=3/2, F=1, mF=1  ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=2, mF=-2 ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=2, mF=-1 ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=2, mF=0  ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=2, mF=1  ))
    LJFmF.append(Niveau( n=3, L=2, J=5/2, F=2, mF=2  ))
    return LJFmF

class Passage:
    def __init__(self,base_depart,base_arrivee,M1S,M3S3P,M3):
        self.base_depart, self.base_arrivee = base_depart, base_arrivee
        self.M1S, self.M3S3P, self.M3 = M1S, M3S3P, M3
        
def LSI_vers_LJI():
    P = np.zeros((36,36))
    for m,d in enumerate(LmSmLmI()): # départ
        for n,a in enumerate(LJmJmI()): # arrivée
            if d.L != a.L or d.mI != a.mI:
                P[n,m] = 0 # Mêmes mI et L pour les deux bases
            else:
                P[n,m] = clebsch(j1=d.L,m1=d.mL,j2=d.S,m2=d.mS,J=a.J,M=a.mJ)
    return Passage('LmSmLmI','LJmJmI',P[12:16,12:16],P[:16,:16],P)
            
def LJI_vers_LJF():
    P = np.zeros((36,36))
    for m,d in enumerate(LJmJmI()): # départ
        for n,a in enumerate(LJFmF()): # arrivée
            if d.L != a.L or d.J != a.J:
                P[n,m] = 0 # Mêmes J et L pour les deux bases
            else:
                P[n,m] = clebsch(j1=d.J,m1=d.mJ,j2=d.I,m2=d.mI,J=a.F,M=a.mF)
    return Passage('LJmJmI','LJFmF',P[12:16,12:16],P[:16,:16],P)
    
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
           
###############################################################################

test(LSI_vers_LJI().M3,1)
test(LJI_vers_LJF().M3,2)