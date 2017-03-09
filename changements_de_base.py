# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 11:42:50 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from wigner import Wigner3j
from matplotlib import pyplot as plt

    
def LSI_vers_LJI():
    P = np.zeros((16,16))
    for m,d in enumerate(LmSmLmI): # départ
        for n,a in enumerate(LJmJmI): # arrivée
            if d.L != a.L or d.mI != a.mI:
                P[n,m] = 0 # Parce qu'on doit avoir les mêmes mI et L pour les deux bases
            else:
                P[n,m] = clebsch(j1=d.L,m1=d.mL,j2=d.S,m2=d.mS,J=a.J,M=a.mJ)
    return P
            
def LJI_vers_LJF():
    P = np.zeros((16,16))
    for m,d in enumerate(LJmJmI): # départ
        for n,a in enumerate(LJFmF): # arrivée
            if d.L != a.L or d.J != a.J:
                P[n,m] = 0 # Parce qu'on doit avoir les mêmes J et L pour les deux bases
            else:
                P[n,m] = clebsch(j1=d.J,m1=d.mJ,j2=d.I,m2=d.mI,J=a.F,M=a.mF)
#            ntest,mtest = n,m
    return P
    
def clebsch(j1,m1,j2,m2,J,M):
    return (-1)**(j1-j2+M)*np.sqrt(2*J+1)*Wigner3j(j1,j2,J,m1,m2,-M)
    
class Niveau:
    def __init__(self,S=False, L=False, I=False, J=False, F=False, mS=False, mL=False, mI=False, mJ=False, mF=False):
        self.S, self.L, self.I, self.J, self.F = S, L, I, J, F
        self.mS, self.mL, self.mI, self.mJ, self.mF = mS, mL, mI, mJ, mF
        
def frigeo(matrice):
    plt.imshow(matrice, interpolation='none', cmap=plt.gray() )


LmSmLmI = []
LmSmLmI.append(Niveau( L=1, mS=1/2,  mL=1,  mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=1/2,  mL=1,  mI=-1/2, S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=1/2,  mL=0,  mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=-1/2, mL=1,  mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=1/2,  mL=0,  mI=-1/2, S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=1/2,  mL=-1, mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=-1/2, mL=1,  mI=-1/2, S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=-1/2, mL=0,  mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=1/2,  mL=-1, mI=-1/2, S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=-1/2, mL=0,  mI=-1/2, S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=-1/2, mL=-1, mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=1, mS=-1/2, mL=-1, mI=-1/2, S=1/2 ))
LmSmLmI.append(Niveau( L=0, mS=1/2,  mL=0,  mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=0, mS=1/2,  mL=0,  mI=-1/2, S=1/2 ))
LmSmLmI.append(Niveau( L=0, mS=-1/2, mL=0,  mI=1/2,  S=1/2 ))
LmSmLmI.append(Niveau( L=0, mS=-1/2, mL=0,  mI=-1/2, S=1/2 ))

LJmJmI = []
LJmJmI.append(Niveau( L=1, J=3/2, mJ=3/2,  mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=1, J=3/2, mJ=3/2,  mI=-1/2, I=1/2 ))
LJmJmI.append(Niveau( L=1, J=3/2, mJ=1/2,  mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=1, J=3/2, mJ=1/2,  mI=-1/2, I=1/2 ))
LJmJmI.append(Niveau( L=1, J=3/2, mJ=-1/2, mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=1, J=3/2, mJ=-1/2, mI=-1/2, I=1/2 ))
LJmJmI.append(Niveau( L=1, J=3/2, mJ=-3/2, mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=1, J=3/2, mJ=-3/2, mI=-1/2, I=1/2 ))
LJmJmI.append(Niveau( L=1, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=1, J=1/2, mJ=1/2,  mI=-1/2, I=1/2 ))
LJmJmI.append(Niveau( L=1, J=1/2, mJ=-1/2, mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=1, J=1/2, mJ=-1/2, mI=-1/2, I=1/2 ))
LJmJmI.append(Niveau( L=0, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=0, J=1/2, mJ=1/2,  mI=-1/2, I=1/2 ))
LJmJmI.append(Niveau( L=0, J=1/2, mJ=-1/2, mI=1/2,  I=1/2 ))
LJmJmI.append(Niveau( L=0, J=1/2, mJ=-1/2, mI=-1/2, I=1/2 ))

LJFmF = []
LJFmF.append(Niveau( L=1, J=3/2, F=2, mF=2  ))
LJFmF.append(Niveau( L=1, J=3/2, F=2, mF=1  ))
LJFmF.append(Niveau( L=1, J=3/2, F=1, mF=1  ))
LJFmF.append(Niveau( L=1, J=3/2, F=2, mF=0  ))
LJFmF.append(Niveau( L=1, J=3/2, F=1, mF=0  ))
LJFmF.append(Niveau( L=1, J=3/2, F=2, mF=-1 ))
LJFmF.append(Niveau( L=1, J=3/2, F=1, mF=-1 ))
LJFmF.append(Niveau( L=1, J=3/2, F=2, mF=-2 ))
LJFmF.append(Niveau( L=1, J=1/2, F=1, mF=1  ))
LJFmF.append(Niveau( L=1, J=1/2, F=1, mF=0  ))
LJFmF.append(Niveau( L=1, J=1/2, F=0, mF=0  ))
LJFmF.append(Niveau( L=1, J=1/2, F=1, mF=-1 ))
LJFmF.append(Niveau( L=0, J=1/2, F=1, mF=1  ))
LJFmF.append(Niveau( L=0, J=1/2, F=1, mF=0  ))
LJFmF.append(Niveau( L=0, J=1/2, F=0, mF=0  ))
LJFmF.append(Niveau( L=0, J=1/2, F=1, mF=-1 ))
        