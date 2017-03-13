# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from fluo3S_H_mars2017_I import *
from fluo3S_H_mars2017_II import *
from fluo3S_H_mars2017_III import *


##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Résolution des équations de Bloch optiques
# 4. Calcul de la forme de raie finale et enregistrement des valeurs
# 5. Ajustement par une lorentzienne et enregistrement des valeurs


##### 4. Calcul de la forme de raie finale et enregistrement des valeurs

c = 299792.458  # en km/s
nu0 = 2922742937 # en Mhz

# Description du système par sa matrice densité (20x20)
# Hamiltonien total
#     = H0 (avec HFS, Zeeman et diamagnétisme) 
#     + H1 (interaction dipolaire avec le laser)
#     + H_Stark (effet Stark motionnel en présence de champ B)

def forme_de_raie(B,sigma):

# On se place dans la base des vecteurs propres de H0
H0 = H_Zeeman(B) + H_HFS()
E,niveau = np.linalg.eig(H0)

H_Stark_reduit = H_Stark_reduit()

    
    
    frequences = np.linspace(-5,5,1001) # en MHz
    vitesses = np.linspace(0,10,101)    # en km/s
    for i,delta in enumerate(frequences):
        for j,v in enumerate(vitesses):
            H_Stark = v*H_Stark_reduit
            coef_Doppler = v**2*nu0/(2*c**2)
            
            hfs = range(3) # [mF = 0, mF = 1, mF = -1]
            ecart = E.3S - E_Bnul.3S + delta + coef_Doppler - E.1S + E_Bnul.1S
            coupl = [np.sum(H_Stark.3P**2/(-gamma3P/2 + 1j*(ecart[k]+E.3S[k]-E.3P))) for k in hfs]
            BB = [-H_Stark.3P**2*(gamma3S+gamma3P)/((E.3S[k]-E.3P)**2+((gamma3S+gamma3S)/2)**2) for k in hfs]
            A = coupl - gamma3S/2 + 1j*ecart
            K = np.real(-1/(A))
            CC = [np.real(H_Stark.3P**2/(A[k]*(ecart[k]+E.3S[k]-E.3P+1j*gamma3P/2)*(E.3S[k]-E.3P+1j*(gamma3S+gamma3P)/2))) for k in hfs]
            num = [K - np.sum(CC[k]*(1+BB/(gamma3P-BB))) for k in hfs]
            den = [gamma3S - np.sum(BB*(1+BB/(gamma3P-BB))) for k in hfs]
            pop3S = num/den
            pop3P = [np.sum(CC-pop3S[k]*BB)/(gamma3P-BB) for k in hfs]
            fluo = gamma3S*pop3S + 0.11834*gamma3P*pop3P


def fluo_sous_niveau(delta,F=1,mF=1):
    ecart = E.3s[]
    niveau_Bnul['3s'] - niveau['3s'] + delta + coef_Doppler - niveau_Bnul['1s']
    
    ############## Niveau 3S1/2 (F=1,mF=1)
    ecart1 = delta+dop-(niv3s[2]-niv3s0[2]-niv1s[2]+niv1s0[2])  
                        # écart à résonance en MHz 
                        # prenant en compte l'effet Doppler du 2ème ordre
                        # et les déplacements Stark des niveaux 1S et 3S
    V1=Vs[:,2]
    coupl1 = 0
    for x in range(12):       # 12 niveaux 3P couples aux 4 niveaux 3S 
        coupl1 += V1[x]**2/(-gamma3p/2 + 1j*(ecart1+niv3s[2]-nivp[x]))
        # Coefficient B dans EPJD2010 p.251 et thèse Gaëtan p.119 :
        BB[x] = -V1[x]**2*(gamma3s+gamma3p)/((niv3s[2]-nivp[x])**2+((gamma3s+gamma3p)/2)**2)
    # Formule (4-46) de Gaëtan et (13) de EPJD2010 :
    A = 1j*ecart1-gamma3s/2+coupl1   
    AA = -1/A
    K = AA.real
    
    for x in range(12):          
        CC[x] = (V1[x]**2/(A*(ecart1+niv3s[2]-nivp[x]+1j*gamma3p/2)*(niv3s[2]-nivp[x]
                     +1j*(gamma3s+gamma3p)/2))).real
    
    # En mieux :
    
    
    # Population de l'état 3S(F=1,mF=+1) :
    # formule (4-47) de la thèse de Gaëtan 
    num = K
    den = gamma3s 
    for x in range(12):
        num = num - CC[x]*(1+BB[x]/(gamma3p-BB[x]))
        den = den - BB[x]*(1+BB[x]/(gamma3p-BB[x]))
    popul3s = num/den
    fluo1 = gamma3s*popul3s
    
    # Population des états 3P excités :
    # formule (4-48) de la thèse de Gaëtan :
    # et calcul de la fluorescence (formule (22) de EPJD2010) :
    popul3p = np.zeros(12, dtype = float)
    for x in range(12):
        popul3p[x] = (CC[x]-popul3s*BB[x])/(gamma3p-BB[x])
        fluo1 = fluo1+0.11834*gamma3p*popul3p[x]
        # on ne distingue pas les rapports de branchement 
        # pour les niveaux P3/2 et P1/2   