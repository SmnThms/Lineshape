# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 14:07:33 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import os.path

    
def cmap(nb_plots):
    colormap = plt.cm.jet
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, nb_plots)])

exp = np.loadtxt('Resultats_Lor.txt') # Points expérimentaux, en Hz
liste_vo = [0,1,2,3,4,5,6,7,8,9] # en km/s
liste_sigma = [0.9,1.1,1.3,1.5,1.7,1.9] # ainsi que 0.7 et 0.9 ; en km/s
liste_sigma_additionnelle = [1.63,1.55,1.48,1.42,1.36,1.32,1.27,1.23,1.688,1.587,1.506,1.445,1.375,1.324,1.274,1.223]
yy = np.linspace(0.7,1.9,100)


###### Génération des données à afficher ######################################
calc, p, vallee, fluo = {}, {}, {}, {}
liste_delta = np.linspace(-5*exp[0,2],5*exp[0,2],1000) # en Hz
chi2vo,f_1S3Svo = np.zeros(len(liste_delta)),np.zeros(len(liste_delta))
for vo in liste_vo:
    calc[vo] = []
    for sigma in liste_sigma + liste_sigma_additionnelle:
        fichier = 'vo='+str(vo)+'_sigma='+str(sigma)+'.txt'
        if os.path.isfile(fichier):
            fluo[vo] = np.loadtxt(fichier)[:,1]*1E6 # en Hz
########### f_exp(B) = f_1S3S(B=0) + fluo(B,vo,sigma) + delta(vo,sigma) #######
            for i,delta in enumerate(liste_delta):
                chi2vo[i] = np.sum(((exp[:,1]-(fluo[vo][:]+exp[0,1]-fluo[vo][0]+delta))/exp[:,2])**2)
            calc[vo].append([vo, sigma, np.min(chi2vo), liste_delta[np.argmin(chi2vo)]])
    calc[vo] = np.array(calc[vo])
    p[vo] = np.polyfit(calc[vo][:,1],calc[vo][:,2],2)
    sqrtchi2 = np.sqrt(np.poly1d(p[vo])(yy)/13)
    vallee[vo] = [vo, yy[np.argmin(sqrtchi2)], np.min(sqrtchi2)] 


###### Figure 1 : chi2 = f(vo,sigma) ##########################################
plt.close('all')       
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('vo (km/s)')
ax.set_ylabel('sigma (km/s)')
ax.set_zlabel('chi2')
cmap(len(liste_vo))
for vo in liste_vo:
    ax.scatter(calc[vo][:,0],calc[vo][:,1],calc[vo][:,2])
    ax.plot(vo*np.ones(len(yy)),yy,np.poly1d(p[vo])(yy))
    
    
###### Figure 2 : sqrt(chi2/n-1) = f(vo,sigma) ################################
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('vo (km/s)')
ax.set_ylabel('sigma (km/s)')
ax.set_zlabel('sqrt(chi2/n-1)')
cmap(len(liste_vo))
for vo in liste_vo:
    ax.scatter(calc[vo][:,0],calc[vo][:,1],np.sqrt(calc[vo][:,2]/13))
    zz = np.sqrt(np.poly1d(p[vo])(yy)/13)
    ax.plot(vo*np.ones(len(yy)),yy,zz)
    
    
##### Figure 3 : chi2 dans la vallée théorique ################################
plt.figure(3)
plt.plot(liste_vo,[vallee[vo][2] for vo in liste_vo],'o--',color='b')
plt.xlabel('vo (km/s)')
plt.ylabel('sqrt(chi2/n-1)')
ax1 = plt.gca()
ax1.set_ylabel('sqrt(chi2/n-1)',color='b')
plot_sigma = True
if plot_sigma:
    ax2 = ax1.twinx()
    ax2.plot(liste_vo,[vallee[vo][1] for vo in liste_vo],'o--',color='r')
    ax2.set_ylabel('sigma (km/s)',color='r')  
plt.title('Dans la vallee \'theorique\', des minima des polynomes')
    


###### Ajustement dans la vallée ##############################################
###### f_exp(B) = f_1S3S(B=0) + fluo(B,vo,sigma) + delta(vo,sigma) ############

liste_vo = [0,1,2,3,4,5,6,7,8,9]
#liste_vo = [5,6]
liste_sigma_vallee = [1.688,1.587,1.506,1.445,1.375,1.324,1.274,1.233,1.197,1.160] # Vallée avec delta
#liste_sigma_vallee = [1.63,1.55,1.48,1.42,1.36,1.32,1.27,1.23] # Vallée sans delta
fluo,delta_min,f_1S3S,chi2,sig,moyenne,std = {},{},{},{},{},{},{}
liste_delta = np.linspace(-5*exp[0,2],5*exp[0,2],1000) # en Hz
chi2vo,f_1S3Svo = np.zeros(len(liste_delta)),np.zeros(len(liste_delta))

for vo in liste_vo:
    for sigma in liste_sigma_vallee:
        fichier = 'vo='+str(vo)+'_sigma='+str(sigma)+'.txt'
        if os.path.isfile(fichier):
            fluo[vo] = np.loadtxt(fichier)[:,1]*1E6 # en Hz
            for i,delta in enumerate(liste_delta):
                chi2vo[i] = np.sum(((exp[:,1]-(fluo[vo][:]+exp[0,1]-fluo[vo][0]+delta))/exp[:,2])**2)
            delta_min[vo] = liste_delta[np.argmin(chi2vo)]
            f_1S3S[vo] = exp[:,1] - fluo[vo][:] - delta_min[vo]
            chi2[vo] = np.min(chi2vo)
            sig[vo] = sigma
            moyenne[vo] = np.mean(f_1S3S[vo])
            std[vo] = np.std(f_1S3S[vo])
moyenne_tous = np.mean([f_1S3S[vo] for vo in liste_vo])


###### Figure 4 : résonance = f(B) dans la vallée #############################
plt.figure(4)
start = 1 # Pour afficher ou non le point B=0
plt.errorbar(exp[start:,0],exp[start:,1],yerr=exp[start:,2],fmt='o')
cmap(len(liste_vo))
for vo in liste_vo:
    f_fit = f_1S3S[vo][0] + fluo[vo][start:] + delta_min[vo]
    plt.plot(exp[start:,0],f_fit,label='vo='+str(vo)+'  sigma='+str(sig[vo]))
plt.title('vo_sigma (km/s)')
plt.xlabel('B (G)')
plt.ylabel('Resonance (Hz)')
plt.legend(loc=3)


###### Figure 5 : f_1S3S = f(B) dans la vallée ################################
plt.figure(5)
cmap(len(liste_vo))
for vo in liste_vo:
    plt.plot(exp[:,0],f_1S3S[vo]-moyenne_tous,'o',label='vo = '+str(vo)+'  sigma='+str(sig[vo]))
cmap(len(liste_vo))
for vo in liste_vo:   
    plt.plot([0,200],[moyenne[vo],moyenne[vo]])
plt.xlabel('B (G)')
moyenne_tous_str = '2 922 742 936 '+str(int((moyenne_tous-int(moyenne_tous/1E6)*1E6)//1000))+' '+str(int((moyenne_tous-int(moyenne_tous/1E3)*1E3)))
plt.ylabel('f_1S3S - '+moyenne_tous_str+' Hz')
plt.legend(loc=3)
plt.title('f_1S3S = f_exp - fluo - delta')
for vo in liste_vo:
    print 'vo = '+str(vo)+' km/s  :  moyenne(f_1S3S) = '+str(moyenne[vo])+' Hz'
            
            
###### Figure 6 : delta,chi2 = f(vo) dans la vallée ###########################
plt.figure(6)
plt.plot(liste_vo, np.sqrt(np.array([chi2[vo] for vo in liste_vo])/13),'o--')
ax1 = plt.gca()
ax1.set_ylabel('sqrt(chi2/n-1)',color='b')
plot_delta = True
if plot_delta:
    ax2 = ax1.twinx()
    ax2.plot(liste_vo,[delta_min[vo] for vo in liste_vo],'o--',color='r')
    ax2.set_ylabel('delta (Hz)',color='r')  
plt.title('Dans la vallee \'echantillonnee\'')