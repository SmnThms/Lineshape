# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from fluo3S_H_mars2017_III import *
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import datetime

##### STRUCTURE DU PROGRAMME #####
# 1. Définition des bases et matrices de passage
# 2. Définition des hamiltoniens
# 3. Calcul de la forme de raie
# 4. Ajustement, enregistrement, affichage

##### 4. Ajustement, enregistrement, affichage

class raie():
    def __init__(self,B,sigma,vo=0):
        self.B = B
        self.sigma = sigma        
        self.vo = vo
        self.frequences,self.fluo = forme_de_raie(B,sigma,vo)
        self.date = datetime.date.today().isoformat()
        self.resonance = self.ajuster()[0]
        
    def ajuster(self):
        parametres,erreur = curve_fit(lorentz,self.frequences,self.fluo)
        return parametres
        
    def enregistrer(self):
        nom = 'fluo3S_H_'+self.date+'_'+str(self.B)+'_'+str(self.sigma) \
              +'_'+str(self.vo)+'.txt'   
        header = ''
        np.savetxt(nom,self.array,header=header,fmt='%10.2f')
        return self
        
    def afficher(self):
        plt.plot()
        plt.xlabel('Fréquence (MHz)')
        plt.title('B = '+str(self.B)+', sigma = '+str(self.sigma) \
                  +' et vo = '+str(self.vo))
        return self    

def fit_B(liste_B,sigma,vo):
    parametres = np.array([raie(B,sigma,vo).ajuster() for B in liste_B])
    if len(liste_B) is 1:
        print 'x0 =',parametres[0][0]
    else:
        array_B = np.array(liste_B).reshape(len(liste_B),1)
        date = datetime.date.today().isoformat()    
        nom = date+'_vo='+str(vo)+'_sigma='+str(sigma)+'_nb_B=' \
              +str(len(liste_B))+'.txt'
        header = date+'\tvo='+str(vo)+' km/s\tsigma='+str(sigma)+' km/s' \
                 +'\n Lorentzienne : S/(1+((x-x0)/(gamma/2))**2)' \
                 +'\n B (G) \t||\t x0 (MHz) \t||\t S (1/s) \t||\t gamma (MHz)'
        resultat = np.concatenate((array_B,parametres),1)
        np.savetxt(nom,resultat,header=header,fmt='%10.6f') 
        return resultat
    
def lorentz(x,x0,S,gamma):
    return S/(1+((x-x0)/(gamma/2))**2)
    
def test(M,fig=0):
    if np.max(M)<1E-12:
        print 'Matrice nulle'
    else:
        if fig is 0:
            plt.close('all')
        else:
            plt.figure(fig)
        plt.imshow(M, interpolation='nearest',cmap=plt.jet())
        plt.colorbar()
        
def diagramme_Zeeman(liste_B,types_niveaux,fig=1,text=False):
    plt.figure(fig)    
    for B in liste_B:  
        
        H0 = H_HFS().additionner(H_Zeeman(B).convert(LSI_vers_LJI()).convert(LJI_vers_LJF()))
        H0.diagonalise()        
        niveaux = getattr(H0,type_niveaux)
        plt.plot(B*np.ones(len(niveaux)),niveaux,'o',color='b')
#        plt.ylim(np.min(niveaux)*1.1,np.max(niveaux)*1.1)
        if text:        
            for niveau in niveaux:
                plt.text(B,niveau,'  '+str(niveaux.tolist().count(niveau)))
    plt.xlabel('B')
    plt.ylabel('E')