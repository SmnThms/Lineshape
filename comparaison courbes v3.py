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

def afficher(vo,sigma,figure=2):
    plt.figure(figure)
    start = 1
    fit = np.loadtxt('vo='+str(vo)+'_sigma='+str(sigma)+'.txt')[:,1]*1E6
    fit += exp[0,1] - fit[0]
    plt.plot(exp[start:,0],fit[start:],label=str(vo)+'_'+str(sigma))
    plt.title('Pour vo='+str(vo)+' et sigma='+str(sigma))
    plt.xlabel('B (G)')
    plt.ylabel('Resonance (Hz)')
    plt.errorbar(exp[start:,0],exp[start:,1],yerr=exp[start:,2])#,linestyle=None)
    plt.legend()


exp = np.loadtxt('Resultats_Lor.txt') # Points expérimentaux
liste_vo = [0.3,1,2,3,4,5]
liste_sigma = [0.9,1.1,1.3,1.5,1.7,1.9]

calc = {}
p = {}
for vo in liste_vo:
    calc[vo] = []
    for sigma in liste_sigma:
        fichier = 'vo='+str(vo)+'_sigma='+str(sigma)+'.txt'
        if os.path.isfile(fichier):
            fit = np.loadtxt(fichier)[:,1]
            fit *= 1E6 # car exp en Hz et fit en MHz
            fit += exp[0,1] - fit[0] # normalisation à B = 0 G
            chi2 = np.sqrt(np.sum(((exp[:,1]-fit)/exp[:,2])**2))/np.sqrt(13)
            calc[vo].append([vo, sigma, chi2])
    calc[vo] = np.array(calc[vo])
    p[vo] = np.polyfit(calc[vo][:,1],calc[vo][:,2],2) 

plt.close()       
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('vo (km/s)')
ax.set_ylabel('sigma (km/s)')
ax.set_zlabel('sqrt(chi2/n-1)')
yy = np.linspace(0.9,1.9,100)
for vo in liste_vo:
    ax.scatter(calc[vo][:,0],calc[vo][:,1],calc[vo][:,2])
    ax.plot(vo*np.ones(len(yy)),yy,np.poly1d(p[vo])(yy))
    
afficher(4,1.3)

ROI_vo    = [0,  1,  2,  3,  3,  4,  4,  5  ]
ROI_sigma = [1.6,1.7,1.5,1.5,1.3,1.5,1.3,1.5,1.3]
for vo,sigma in zip(ROI_vo,ROI_sigma):
    afficher(vo,sigma,3)


    
