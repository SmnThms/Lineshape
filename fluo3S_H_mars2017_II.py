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
        self.E1S, self.M1S = np.linalg.eig(self.H1S)
        self.H1S = np.diag(self.E1S)
        self.E3S, self.M3S = np.linalg.eig(self.H3S3P[-4:,-4:])
        self.E3P, self.M3P = np.linalg.eig(self.H3S3P[:-4,:-4])
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
        
def H_HFS(): # dans la base LJFmF
    base = 'LJFmF'
    coef1S = 1420.40575176     # A1s en MHz pour le 1S en champ nul
    H1S = coef1S/4*np.diag([1,1,-3,1])
    coef3S = (8/27)*177.55686  # A3s = A2s*8/27 en MHz en champ nul
    H3S = coef3S/4*np.diag([1,1,-3,1])
    # valeurs sans structure hyperfine ; le zéro d'énergie est le 3S1/2
    ep3 = 2934.968             # énergie du 3P3/2 en MHz
    L3 = -314.784              # énergie du 3P1/2 en MHz
    A3s = (8/27)*177.55686     # A3s = A2s*8/27 en MHz en champ nul
    H3P = np.diag([1,1,0,1,0,1,0,1,0,0,0,0])*(ep3 + A3s/20) # P3/2_2
    H3P[2,2] = H3P[4,4] = H3P[6,6] = ep3 - A3s/12           # P3/2_1
    H3P[8,8] = H3P[9,9] = L3 + A3s/12                       # P1/2_1
    H3P[10,10] = L3 - A3s/4                                 # P1/2_0
    H3P[11,11] = L3 + A3s/12                                # P1/2_1
    H3P[2,8] = H3P[4,9] = H3P[6,11] = -np.sqrt(2)/48*A3s    # couplage
    H3P[8,2] = H3P[9,4] = H3P[11,6] = -np.sqrt(2)/48*A3s    # couplage
    H3S3P = np.zeros((16,16))
    H3S3P[-4:,-4:] = H3S
    H3S3P[:-4,:-4] = H3P
    return Hamiltonien(base,H1S,H3S3P)    
    
def H_Zeeman(B): # dans la base LmSmLmI
    base = 'LmSmLmI'
    mub = 1.399601126       # magnéton de Bohr en MHz/G
    epmr = 5.446170232e-4   # electron-to-proton mass ratio 
    coef_diamagnetique = 1.488634644e-10
    gn = 5.585694675
    ML = mub*B*(1-epmr)     # coefflz
    MI = -0.5*mub*B*gn*epmr # coeffiz
    # 1S
    g_1S = 2.00228377       # facteur de Landé (à vérifier)
    MJ_1S = 0.5*mub*B*g_1S  # c1ss
    diam_1S = -2*coef_diamagnetique*B*B
    H1S = MJ_1S*np.diag([1,1,-1,-1]) + diam_1S*np.diag([1,1,1,1]) \
          + MI*np.diag([1,-1,1,-1])
    # 3S
    g_3S = 2.0023152    
    MJ_3S = 0.5*mub*B*g_3S  # c3ss
    diam_3S = -138*coef_diamagnetique*B*B 
    H3S = MJ_3S*np.diag([1,1,-1,-1]) + diam_3S*np.diag([1,1,1,1]) \
          + MI*np.diag([1,-1,1,-1])
    # 3P
    g_3P = 2.0023152        
    MJ_3P = 0.5*mub*B*g_3P  # c3ps
    diam_3P = -360*coef_diamagnetique*B**2
    H3P = MJ_3P*np.diag([1,1,1,-1,1,1,-1,-1,1,-1,-1,-1])  # Szp
    H3P += ML*np.diag([1,1,0,1,0,-1,1,0,-1,0,-1,-1])      # Lzp
    H3P += diam_3P*np.diag([2,2,1,2,1,2,2,1,2,1,2,2])     # Lzp
    H3P += MI*np.diag([1,-1,1,1,-1,1,-1,1,-1,-1,1,-1])    # Izp
    H3S3P = np.zeros((16,16))
    H3S3P[-4:,-4:] = H3S
    H3S3P[:-4,:-4] = H3P
    return Hamiltonien(base,H1S,H3S3P) 
   
def H_FS(): # dans la base LJmJmI
    base = 'LJmJmI'
    H1S = np.zeros((4,4)) # on n'en a pas besoin
    # Partie angulaire
    # le champ électrique est perpendiculaire à l'axe de quantification, 
    # et on a deltami=0 et |deltamj|=1
    H3S3P = np.zeros((16,16))
    H3S3P[12,0] = H3S3P[13,1] = -1/np.sqrt(6)
    H3S3P[14,2] = H3S3P[15,3] = -1/(3*np.sqrt(2))
    H3S3P[12,4] = H3S3P[13,5] = 1/(3*np.sqrt(2))
    H3S3P[14,6] = H3S3P[15,7] = 1/np.sqrt(6)
    H3S3P[14,8] = H3S3P[15,9] = H3S3P[12,10] = H3S3P[13,11] = 1/3
    H3S3P[:12,12] = H3S3P[12,:12]    
    H3S3P[:12,13] = H3S3P[13,:12]
    H3S3P[:12,14] = H3S3P[14,:12]
    H3S3P[:12,15] = H3S3P[15,:12]
    # Partie radiale R/a0 
    H3S3P *= 1.279544928*9*np.sqrt(2)
    return Hamiltonien(base,H1S,H3S3P)