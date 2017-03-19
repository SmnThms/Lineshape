# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from fluo3S_H_mars2017_I import *
from fluo3S_H_mars2017_II import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf
import time
import math



##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage

##### 3. Calcul de la forme de raie

c = 299792.458  # en km/s
nu0 = 2922742937 # en Mhz

def forme_de_raie(B,sigma,vo=0):
    # On se place dans la base des vecteurs propres de H0
    H0 = H_HFS().additionner(H_Zeeman(B).convert(LSI_vers_LJI()).convert(LJI_vers_LJF()))
    H0.diagonalise()
    H0_B0 = H_HFS() # Pour B = 0 G
    H0_B0.diagonalise()
    H_Stark_reduit = H_FS().convert(LJI_vers_LJF()).convert(H0.LJF_vers_baseH0) # H_Stark/(v*B)
        
    frequences = np.linspace(-5,5,1001)   # en MHz
    vitesses = np.linspace(0.01,10.,101)  # en km/s (v doit être non nul)
    Norm = quad(lambda x:coefv(x,sigma,vo),0.,10.)[0] 
    hfs = [1,2,3] # [mF=0, mF=1, mF=-1]
    fluo, fluo_v = np.zeros(len(frequences)), [[0]*len(vitesses)]*len(H0.E1S)
    for i,delta in enumerate(frequences):
        for j,v in enumerate(vitesses):
            coef_Stark = H_Stark_reduit.multiplier(v*B/1000).H3S3P[:-4,12:]
            # 1/1000 = 1/10000 (passage des T en Gauss) * 1000 (pour les km/s) /100 (pour les V/cm)
            coef_Doppler = v**2*nu0/(2*c**2)
            gamma3S = 1.004945452       # en MHz
            gamma3P = 30.192            # en MHz
            for k in hfs: 
                ecart = H0.E3S[k] - H0_B0.E3S[k] + delta + coef_Doppler - H0.E1S[k] + H0_B0.E1S[k]
                coupl = np.sum(coef_Stark[:,k]**2/(-gamma3P/2 + 1j*(ecart + H0.E3S[k] - H0.E3P)))
                BB = -coef_Stark[:,k]**2*(gamma3S+gamma3P)/((H0.E3S[k]-H0.E3P)**2+((gamma3S+gamma3S)/2)**2)
                A = coupl - gamma3S/2 + 1j*ecart
                CC = np.real(coef_Stark[:,k]**2/(A*(ecart+H0.E3S[k]-H0.E3P+1j*gamma3P/2)*(H0.E3S[k]-H0.E3P+1j*(gamma3S+gamma3P)/2)))
                num = np.real(-1/A) - np.sum(CC*(1+BB/(gamma3P-BB)))
                den = gamma3S - np.sum(BB*(1+BB/(gamma3P-BB)))
                pop3S = num/den
                pop3P = np.sum((CC-pop3S*BB)/(gamma3P-BB))
                fluo_v[k][j] = coefv(v,sigma,vo)*(gamma3S*pop3S + 0.11834*gamma3P*pop3P)
        fluo[i] = np.sum([quad(interp1d(vitesses,fluo_v[k],kind='cubic'),0.01,10.)[0]/Norm for k in hfs])
    return frequences,fluo*1000
    
def coefv(v,sigma,vo):
    xd = 6.5 # taille de la zone de détection/2 en mm
    zr = 35 # longueur de Rayleigh en mm
    taue = 1/(2*np.pi)
    z = v/(np.sqrt(2.)*sigma)
    psi = (z*np.exp(-z**2)+np.sqrt(np.pi)/2.*(1+2*z**2)*erf(z))/(np.sqrt(2.*np.pi)*z**2)
    K = 0.01
    maxwell = 4./np.sqrt(np.pi)*z**2*np.exp(-z**2)
    olander = np.sqrt(np.pi)/2.*np.sqrt(erf(psi/(2*K)))/np.sqrt(psi/(2.*K))
    olivier = np.arctan((xd-v*taue)/zr)+np.arctan((xd+v*taue)/zr)
    return maxwell*olander*olivier*np.exp(-vo/v)