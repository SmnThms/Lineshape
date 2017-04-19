# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from fluo3S_H_avr17_II import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf
import time

##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage

##### 3. Calcul de la forme de raie

##### DONNÉES NUMÉRIQUES #####
c = 299792.458   # en km/s
nu0 = 2922742937 # en Mhz (pourrait tout aussi bien être 2922742900)    
gamma3S = 1.004945452                     # en MHz
gamma3P = 30.192*np.ones(12)              # en MHz
#gamma3P_12 = 30.19175875                 # en MHz
#gamma3P_32 = 30.19165419                 # en MHz
branch_3P = 0.11834

##### CALCUL #####
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
            coef_Doppler = v**2*nu0/(2*c**2)  # en MHz
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
#                print fluo_v[j,k]
                fluo_v[j,k] *= coefv(v,sigma,vo)
        fluo[i] = np.sum([quad(interp1d(vitesses,fluo_v[:,k],kind='cubic'), \
                               0.1,10.1)[0]/normalisation for k in hfs])
    print 'Calcul fini pour B =',B,', sigma =',sigma,', v0 =',vo, \
          ', en ',int(time.time()-debut),' s'
    return frequences,fluo*1000
    
def coefv(v,sigma,vo): #(Olander70, Arnoult thèse, Galtier thèse)
    xd = 6.5e-6        # taille de la zone de détection/2 en km
    zr = 35e-6         # longueur de Rayleigh en km
    taue = 1e-6        # durée de vie en s                      #1/(2*np.pi) ?
    z = v/(np.sqrt(2)*sigma)
    psi = (z*np.exp(-z**2)+np.sqrt(np.pi)/2.*(1+2*z**2)*erf(z)) \
          /(np.sqrt(2*np.pi)*z**2)
    K = 0.01
    maxwell = 4./np.sqrt(np.pi)*z**2*np.exp(-z**2)
    olander = np.sqrt(np.pi)/2.*np.sqrt(erf(psi/(2*K)))/np.sqrt(psi/(2*K))
    olivier = np.arctan((xd-v*taue)/zr)+np.arctan((xd+v*taue)/zr)
    return maxwell*olander*olivier*np.exp(-vo/v)