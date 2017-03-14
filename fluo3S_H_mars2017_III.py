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



##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage

##### 3. Calcul de la forme de raie

c = 299792.458  # en km/s
nu0 = 2922742937 # en Mhz

# Description du système par sa matrice densité (20x20)
# Hamiltonien total
#     = H0 (avec HFS, Zeeman et diamagnétisme) 
#     + H1 (interaction dipolaire avec le laser)
#     + H_Stark (effet Stark motionnel en présence de champ B)

def forme_de_raie(B,sigma,vo=0):

    # On se place dans la base des vecteurs propres de H0
    H0 = H_Zeeman(B) + H_HFS()
    E,niveau = np.linalg.eig(H0)
    
    H_Stark_reduit = H_Stark_reduit()

    
    
    frequences = np.linspace(-5,5,1001) # en MHz
    vitesses = np.linspace(0,10,101)    # en km/s
    fluo_v, fluo = np.zeros(len(vitesse)), np.zeros(len(frequences))
    Norm = quad(coef,0.,10.)[0] 
    for i,delta in enumerate(frequences):
        for j,v in enumerate(vitesses):
            H_Stark = v*H_Stark_reduit
            coef_Doppler = v**2*nu0/(2*c**2)
#            for k,mF in enumerate([-1,0,1]):
            hfs = range(3) # [mF = 0, mF = 1, mF = -1]
            ecart = E.3S - E_Bnul.3S + delta + coef_Doppler - E.1S + E_Bnul.1S
            coupl = [np.sum(H_Stark.3P**2/(-gamma3P/2 + 1j*(ecart[k]+E.3S[k]-E.3P))) for k in hfs]
            BB = [-H_Stark.3P**2*(gamma3S+gamma3P)/((E.3S[k]-E.3P)**2+((gamma3S+gamma3S)/2)**2) for k in hfs]
            A = coupl - gamma3S/2 + 1j*ecart
            K = np.real(-1/(A))
            CC = [np.real(H_Stark.3P**2/(A[k]*(ecart[k]+E.3S[k]-E.3P+1j*gamma3P/2)*(E.3S[k]-E.3P+1j*(gamma3S+gamma3P)/2))) for k in hfs]
            num = [K - np.sum(CC[k]*(1+BB/(gamma3P-BB))) for k in hfs]
            den = [gamma3S - np.sum(BB*(1+BB/(gamma3P-BB))) for k in hfs]
            pop3S = [num[k]/den[k] for k in hfs]
            pop3P = [np.sum(CC-pop3S[k]*BB)/(gamma3P-BB) for k in hfs]
            fluo_v[j] = np.array([gamma3S*pop3S[k] + 0.11834*gamma3P*pop3P[k] for k in hfs])
            fluo_v[j] *= coefv(v,sigma,vo)
        fluo[i] = np.sum([quad(interp1d(np.linspace(0,10.,101),fluo_v[:][k],kind='cubic'),0,10.)[0]/Norm for k in hfs])
    return fluo
    
    
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