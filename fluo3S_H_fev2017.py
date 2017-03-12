#-*- coding: utf-8 -*-
# fluo3S_b.py 
# (dérivé du programme fluo3S_corr_B0_SansDoppler.py de Sandrine)
# Calcul de la forme de raie 1S-3S
# en tenant compte de la distribution de vitesse
# avec le modèle de déplétion (Olander 1970)
# à sigma et B fixés
# Le résultat du calcul est la somme de trois courbes de 1001 pts en fréquence
# Fluo1 pour le niveau 3S1/2(F=1,mF=1) et Fluo2 pour le niveau 3S1/2(F=1,mF=-1)
# pour B=0 on tient compte aussi de Fluo3 pour le niveau 3S1/2(F=1,mF=0)
# le résultat est la somme Fluo1 + Fluo2 + Fluo3
# Ce programme ne distingue pas 3P1/2 et 3P3/2
# pour le rapport de branchement et la largeur naturelle
# on a laissé tomber la prise en compte de la nature gaussienne du faisceau UV
# (thèse d'Olivier)

# calcul des courbes théoriques
# avec 2 boucles sur B et vo 

from __future__ import division
from math import *
import numpy as np
import pylab as pl
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from stark import *
from zeeman import *
from distrib_vitesse import *
import time

c = 299792.458  # en km/s
nu0 = 2922742937 # en Mhz
coefdop = nu0/(2.*c**2)

# La nature gaussienne du faisceau UV est prise en compte (thèse d'Olivier)
xd = 6.5 # taille de la zone de détection/2 en mm
zr = 35 # longueur de Rayleigh en mm
taue = 1/(2*pi)

#vo = 1
#sigma = 1

def Oliv(v) :
    return atan((xd-v*taue)/zr)+atan((xd+v*taue)/zr)
    
#sigma = 1.523#1.6      # sigma en km/s 
# B en Gauss :
#listBexp = [-0.299,170.227,-170.825,190.306,-190.904,164.989,-165.587,174.883,-175.481]#,0.26,-0.26]#[-0.255,169.94,-171.34,-191.12,-159.7,159.18,190.31]
## sigma en km/s :
#listsigma = [1.2,1.4,1.6]
## v0 en km/s (exposant -v/v0 dans la distribution de vitesse)
#listvo = [0.1,0.5,1.,1.5,2.,2.5,3,3.5,4.]#[0.326],5.

def coef(v) :
   return fdepl(v,sigma,vo)*Oliv(v)
#   return fdepl(v,sigma,vo)
   
def lorentz(x,x0,S,gamma):
    return S/(1+((x-x0)/(gamma/2))**2)
        
# Largeurs des niveaux en MHz (valeurs de la thèse de Gaëtan)
gamma3s = 1.004945452
gamma3p = 30.192
#for sigma in listsigma:
#    for ivo in range(len(listvo)):# (1):#0):
#        vo=listvo[ivo]    
        # Coefficient de normalisation de l'integrale sur les vitesses :
#Norm = quad(coef,0.,10.)[0] 


#liste_B = np.arange(0,251,2)
liste_B = np.loadtxt('Resultats_Lor.txt')[:,0]
fit = np.zeros((len(liste_B),4))

for vo in [0.8,1,1.5,2,3]:
    for sigma in [0.8,1,1.3,1.5,1.7,1.9]:
        Norm = quad(coef,0.,10.)[0] 

        for iB, B in enumerate(liste_B):
        #            range (len(listBexp)):#(1):#7):
        #            Bexp=listBexp[iB]
        #            B=abs(Bexp)
            Vstark = coef_Stark(B)
            niv1s,niv3s = zeeman_S(B)[:2]
            niv1s0,niv3s0 = zeeman_S(0.0015)[:2]
            nivp = zeeman_P(B)[0]
            nivp0 = zeeman_P(0.0015)[0]
            
            Fluo1 = np.zeros(1001, dtype = float)  
            Fluo2 = np.zeros(1001, dtype = float)  
            Fluo3 = np.zeros(1001, dtype = float)
            fluo_array = np.zeros(1001, dtype = float) 
            
            stringfile = 'Fluo3S_%g_%g_%g_mar2017.txt'%(B,sigma,vo)
            print 'calcul de', stringfile
            debut = time.time()
            
            for inc in range(1001) :        # Boucle sur les fréquences 
                                            # (1001 pts de -5 à +5 MHz)                                
                delta = -5.+inc*0.01        # delta est l'écart à résonance en MHz 
        #        for inc in range(3) :           # Boucle sur les fréquences 
        #                                        # (seulement les 3 premiers pts de la courbe)                                
        #            delta = -5.+inc*0.01        # delta est l'écart à résonance en MHz                                    
                fluo1v = np.zeros(101, dtype = float) 
                fluo2v = np.zeros(101, dtype = float) 
                fluo3v = np.zeros(101, dtype = float) 
                
                for incv in range(101) :    # Boucle sur les vitesses 
                                            # (101 pts de 0 à 10 km/s)
                    v = 0.1*(incv+1)       
                    Vs = Vstark*v
            #        dop = 16.25995046e-3*v**2       # en MHz
                    dop = coefdop*v**2
                    coupl = np.zeros(12, dtype = complex)
                    BB = np.zeros(12, dtype = float)
                    CC = np.zeros(12, dtype = float)
                    
                    ############## Niveau 3S1/2 (F=1,mF=1)
                    ecart1 = delta + dop-(niv3s[2]-niv3s0[2]-niv1s[2]+niv1s0[2])  
                                        # écart à résonance en MHz 
                                        # prenant en compte l'effet Doppler du 2ème ordre
                                        # et les déplacements Stark des niveaux 1S et 3S
                    V1=Vs[:,2]
                    coupl1 = 0
                    for x in range(12):       # 12 niveaux 3P couples aux 4 niveaux 3S 
                        coupl[x] = V1[x]**2/(-gamma3p/2 + 1j*(ecart1+niv3s[2]-nivp[x]))
                        coupl1 = coupl1 + coupl[x]
                        # Coefficient B dans EPJD2010 p.251 et thèse Gaëtan p.119 :
                        BB[x] = -V1[x]**2*(gamma3s+gamma3p)/((niv3s[2]-nivp[x])**2+((gamma3s+gamma3p)/2)**2)
                    # Formule (4-46) de Gaëtan et (13) de EPJD2010 :
                    A = 1j*ecart1-gamma3s/2+coupl1   
                    AA = -1/A
                    K = AA.real
                    
                    for x in range(12):          
                        CC[x] = (V1[x]**2/(A*(ecart1+niv3s[2]-nivp[x]+1j*gamma3p/2)*(niv3s[2]-nivp[x]
                                     +1j*(gamma3s+gamma3p)/2))).real
                    
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
            
                    
                    ################### Niveau 3S1/2 (F=1,mF=-1)
                    # Les commentaires sont les mêmes que pour mF=+1
                    ecart2 = delta + dop-(niv3s[3]-niv3s0[3]-niv1s[3]+niv1s0[3])
                                                                               
                    V2=Vs[:,3]
                    coupl2 = 0
                    for x in range(12):
                        coupl[x] = V2[x]**2/(-gamma3p/2 + 1j*(ecart2+niv3s[3]-nivp[x]))
                        coupl2 = coupl2 + coupl[x]
                        BB[x] = -V2[x]**2*(gamma3s+gamma3p)/((niv3s[3]-nivp[x])**2+((gamma3s+gamma3p)/2)**2)
                    
                    A = 1j*ecart2-gamma3s/2+coupl2
                    AA = -1/A
                    K = AA.real
                    
                    for x in range(12):  
                        CC[x] = (V2[x]**2/(A*(ecart2+niv3s[3]-nivp[x]+1j*gamma3p/2)*(niv3s[3]-nivp[x]
                                     +1j*(gamma3s+gamma3p)/2))).real
                    
                    # Population de l'état 3S(F=1,mF=-1) : 
                    num = K
                    den = gamma3s 
                    for x in range(12):
                        num = num - CC[x]*(1+BB[x]/(gamma3p-BB[x]))
                        den = den - BB[x]*(1+BB[x]/(gamma3p-BB[x]))
                    popul3s = num/den
                    fluo2 = gamma3s*popul3s
                    
                    # Population des états 3P excités : 
                    # et calcul de la fluorescence :
                    popul3p = np.zeros(12, dtype = float)
                    for x in range(12):
                        popul3p[x] = (CC[x]-popul3s*BB[x])/(gamma3p-BB[x])
                        fluo2 = fluo2+0.11834*gamma3p*popul3p[x]
            
                    ################### Niveau 3S1/2 (F=1,mF=0)
                    # Les commentaires sont les mêmes que pour mF=+1
                    ecart3 = delta + dop-(niv3s[1]-niv3s0[1]-niv1s[1]+niv1s0[1])
                                                                               
                    V3=Vs[:,1]
                    coupl3 = 0
                    for x in range(12):
                        coupl[x] = V3[x]**2/(-gamma3p/2 + 1j*(ecart3+niv3s[1]-nivp[x]))
                        coupl3 = coupl3 + coupl[x]
                        BB[x] = -V3[x]**2*(gamma3s+gamma3p)/((niv3s[1]-nivp[x])**2+((gamma3s+gamma3p)/2)**2)
                    
                    A = 1j*ecart3-gamma3s/2+coupl3
                    AA = -1/A
                    K = AA.real
                    
                    for x in range(12):   
                        CC[x] = (V3[x]**2/(A*(ecart3+niv3s[1]-nivp[x]+1j*gamma3p/2)*(niv3s[1]-nivp[x]
                                     +1j*(gamma3s+gamma3p)/2))).real
                    
                    # Population de l'état 3S(F=1,mF=0) : 
                    num = K
                    den = gamma3s 
                    for x in range(12):
                        num = num - CC[x]*(1+BB[x]/(gamma3p-BB[x]))
                        den = den - BB[x]*(1+BB[x]/(gamma3p-BB[x]))
                    popul3s = num/den
                    fluo3 = gamma3s*popul3s
                    
                    # Population des états 3P excités : 
                    # et calcul de la fluorescence :
                    popul3p = np.zeros(12, dtype = float)
                    for x in range(12):
                        popul3p[x] = (CC[x]-popul3s*BB[x])/(gamma3p-BB[x])
                        fluo3 = fluo3+0.11834*gamma3p*popul3p[x]
            
                    # Fluorescence pour une vitesse donnée : 
                    coefv = coef(v)   
                    fluo1v[incv] = coefv*fluo1
                    fluo2v[incv] = coefv*fluo2
                    fluo3v[incv] = coefv*fluo3
             
                # on a rempli deux tableaux de 101 pts donnant la fluorescence pour chaque classe de vitesse
                # il reste à les intégrer sur les vitesses entre 0 et 10 km/s :
                Fluo1[inc] = quad(interp1d(np.linspace(0,10.,101),fluo1v,kind='cubic'),0,10.)[0]/Norm
                Fluo2[inc] = quad(interp1d(np.linspace(0,10.,101),fluo2v,kind='cubic'),0,10.)[0]/Norm
                Fluo3[inc] = quad(interp1d(np.linspace(0,10.,101),fluo3v,kind='cubic'),0,10.)[0]/Norm
                fluo_array[inc]= (Fluo1[inc] + Fluo2[inc] + Fluo3[inc])*1000
                # print fluo_array[inc]
                
            # print fluo_array
                  
            f7 = open(stringfile, 'w')
            np.savetxt(f7,fluo_array,fmt='%10.2f',delimiter='     ' )
            f7.close()
            print 'calcul fini pour','B=',B, ', sigma=',sigma, ', vo=', vo
            fin = time.time()
            duree = fin-debut
            print 'temps de calcul : %6.4f min' %(duree/60) 
        
            fit[iB,0] = B
            fit[iB,1:],err = curve_fit(lorentz,np.linspace(-5,5,1001),fluo_array)
    
        header = 'date : 12/03  \tvo = '+str(vo)+' km/s  \tsigma = '+str(sigma)+' km/s \nLorentzienne : S/(1+((x-x0)/(gamma/2))**2)\n B (G) \t||\t x0 (MHz) \t||\t S (1/s) \t||\t gamma (MHz)'
        np.savetxt('fit_12-03_vo='+str(vo)+'sigma='+str(sigma)+'.txt',np.array(fit),header=header,fmt='%10.6f')    

#plt.plot(fit[:,0],fit[:,1])
