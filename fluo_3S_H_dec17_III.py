# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:05 2017 @author: Simon
"""

from __future__ import division
import numpy as np
from fluo_3S_H_dec17_II import *
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import datetime

##### STRUCTURE DU PROGRAMME #####
# I.   Définition des bases et matrices de passage
# II.  Définition des hamiltoniens, calcul matrice densité et forme de raie
# III. Exploitation (ajustement, enregistrement, affichage)

##### STRUCTURE DU FICHIER #######
# class Raie
# fitB
# test matrices (affichage en couleurs)
# diagramme Zeeman

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
        
def diagramme_Zeeman(liste_B=np.linspace(0,200,300),text=False):
    plt.figure()    
    for B in liste_B:  
        H0 = H_SFHF() + convert(convert(H_Zeeman(B),LSI_vers_LJI()),LJI_vers_LJF())
        E, Hdiag = np.linalg.eigh(H0)
        E3S3P = E[4:]
        plt.plot(B*np.ones(len(E3S3P)),E3S3P,'.',color='b')
        if text:        
            for niveau in niveaux:
                plt.text(B,niveau,'  '+str(niveaux.tolist().count(niveau)))
    plt.xlabel('B')
    plt.ylabel('E')
    
def dispersion(liste_B=np.linspace(160,200,10)):
#    liste_B=[180]
    f_centrale, freq = np.zeros(len(liste_B)), np.linspace(-4,4,30)
    for i,B in enumerate(liste_B):
        pop3S = np.array([np.sum(np.diag(convert( \
                          matrice_densite(f=f,B=B),LJI_vers_LJF())).real[4:8])
                          for f in freq])
#        plt.plot(freq,pop3S)
        p,e = curve_fit(lorentz,freq,pop3S)
        f_centrale[i] = p[0]
        print B
    plt.figure()
    plt.plot(liste_B,f_centrale)
    
dispersion()
    
#diagramme_Zeeman()
    
#A = np.array([5.78721395e-07, 7.34881081e-38, 1.09454904e-02, 0, 
#              1.11377096e-34, 1.32100112e-06, 4.94040862e-11, 5.28484825e-38,
#              3.16018493e-07, 0.0000e+00, 0.0000e+00, 0.00000+00])

#B = 170
#H0 = H_SHF().additionner(H_Zeeman(B).convert(LSI_vers_LJI()) \
#                .convert(LJI_vers_LJF()))
#H0.diagonaliser()
#matrice = np.dot(H0.LJF_vers_baseH0.M3S3P,LJI_vers_LJF().M3S3P)

def composantes(vecteur,titre='',fig=1):
    yy = vecteur
    yy[yy<1e-10] = np.zeros(len(yy[yy<1e-10]))
    x_tick_labels = ['L\nJ\nmJ\nmI']
    for niv in LJmJmI():
        x_tick_labels.append(str(niv.L) + '\n' + str(niv.J) + '\n' + \
                             str(niv.mJ)+ '\n' + str(niv.mI))
    xx = np.arange(1,len(yy)+1,1)
#    plt.close('all')
    plt.figure(fig)
    plt.bar(xx,yy,0.5,color=(5/255,53/255,82/255),align='center',log=True)
    plt.xticks(range(len(yy)+1),x_tick_labels)
    plt.title(titre)
    plt.xlabel('Vecteurs de base LJmJmI')
    plt.ylabel('Composantes')
    
#composantes(matrice[3,:],'no 1')
#composantes(populations(),fig=1)
#composantes(populations_2(),fig=2)