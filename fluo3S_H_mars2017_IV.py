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
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


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
        self.array = forme_de_raie(B,sigma,vo)
        self.date = str(14-03-17)
        self.resonance = self.ajuster()[0]
        
    def ajuster(self):
        parametres,erreur = curve_fit(lorentz,np.linspace(-5,5,1001),fluo_array)[0]
        return parametres
        
    def enregistrer(self):
        nom = 'fluo3S_H_'+self.date+'_'+str(self.B)+'_'+str(self.sigma)+'_'+str(self.vo)+'.txt'   
        header = ''
        np.savetxt(nom,donnees,header=header,fmt='%10.2f')
        
    def afficher(self):
        plt.plot()

def ajustement():
    
def enregistrement(donnees,date,B,sigma,vo=0):

    
def affichage():
    
def lorentz(x,x0,S,gamma):
    return S/(1+((x-x0)/(gamma/2))**2)