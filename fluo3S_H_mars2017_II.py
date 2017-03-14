# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np


##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage


##### 2. Définition des hamiltoniens

def H_HFS_S(n): # dans la base LJFmF
    if n is 1:
        coef = 1420.40575176     # A1s en MHz pour le 1S en champ nul
    elif n is 3:
        coef = (8/27)*177.55686  # A3s = A2s*8/27 en MHz en champ nul
    return coef/4*np.diag([1,1,-3,1])

def H_HFS_P(n=3): # dans la base LJFmF
    # valeurs sans structure hyperfine ; le zéro d'énergie est le 3S1/2
    ep3 = 2934.968          # énergie du 3P3/2 en MHz
    L3 = -314.784           # énergie du 3P1/2 en MHz
    A3s = (8/27)*177.55686  # A3s = A2s*8/27 en MHz en champ nul
    H = np.diag([1,1,0,1,0,1,0,1,0,0,0,0])*(ep3 + A3s/20) # P3/2_2
    H[2,2] = H[4,4] = H[6,6] = ep3 - A3s/12               # P3/2_1
    H[8,8], H[9,9] = (L3 + A3s/12)*(1,1)                  # P1/2_1
    H[10,10] = L3 - A3s/4                                 # P1/2_0
    H[11,11] = L3 + A3s/12                                # P1/2_1
    H[2,8] = H[4,9] = H[6,11] = -np.sqrt(2)/48*A3s        # couplage
    H[8,2] = H[9,4] = H[11,6] = -np.sqrt(2)/48*A3s        # couplage
    return H
    
def H_HFS():
    H = np.zeros((20,20))
    H[:4,:4] = H_HFS_S(n=1)
    H[4:8,4:8] = H_HFS_S(n=3)
    H[8:20,8:20] = H_HFS_P()
    return H

# Constantes :
mub = 1.399601126           # magnéton de Bohr en MHz/G
epmr = 5.446170232e-4       # electron-to-proton mass ratio 
coef_diamagnetique = 1.488634644e-10
gn = 5.585694675
    
def H_Zeeman_S(B,n): # dans la base LmSmLmI
    if n is 1:
        g1s = 2.00228377    # facteur de Landé (à vérifier)
        MJ = 0.5*mub*B*g1s  # c1ss
        diam = -2*coef_diamagnetique*B*B
    elif n is 3:
        g3s = 2.0023152     # facteur de Landé (à vérifier)
        MJ = 0.5*mub*B*g3s  # c3ss
        diam = -138*coef_diamagnetique*B*B 
    MI = -0.5*mub*B*gn*epmr # coeffiz
    return MJ*np.diag([1,1,-1,-1]) + MI*np.diag([1,1,1,1]) + diam*np.diag([1,-1,1,-1])

def H_Zeeman_P(B,n=3): # dans la base LmSmLmI, pour n = 3
    g3p = 2.0023152         # facteur de Landé (à vérifier)
    MJ = 0.5*mub*B*g3p      # c3ps
    ML = mub*B*(1-epmr)     # coefflz
    MI = -0.5*mub*B*gn*epmr # coeffiz
    diam = -360*coef_diamagnetique*B**2
    H = MJ*np.diag([1,1,1,-1,1,1,-1,-1,1,-1,-1,-1])  # Szp
    H += ML*np.diag([1,1,0,1,0,-1,1,0,-1,0,-1,-1])   # Lzp
    H += diam*np.diag([2,2,1,2,1,2,2,1,2,1,2,2])     # Lzp
    H += MI*np.diag([1,-1,1,1,-1,1,-1,1,-1,-1,1,-1]) # Izp
    return H
   
def H_Zeeman(B):
    H = np.zeros((20,20))
    H[:4,:4] = H_Zeeman_S(B,n=1)
    H[4:8,4:8] = H_Zeeman_S(B,n=3)
    H[8:20,8:20] = H_Zeeman_P(B)
    return H
   
def H_Stark_reduit():
    return 0