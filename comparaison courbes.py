# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 14:07:33 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def lorentz(x,x0,S,gamma):
    return S/(1+((x-x0)/(gamma/2))**2)

#fit = np.zeros((14,4))
#for vo in [0.8,1,2,3]:
#    for sigma in [1.3,1.5,1.7,1.9]:
#        for iB,B in enumerate(np.loadtxt('Resultats_Lor.txt')[1:,0]):
#            data = np.loadtxt('Fluo3S_'+str(B)+'_'+str(sigma)+'_'+str(vo)+'_fev2017.txt')
#            fit[iB,0] = B
#            fit[iB,1:],err = curve_fit(lorentz,np.linspace(-5,5,1001),data)        
#        
#        header = 'date : 10/03  \tvo = '+str(vo)+' km/s  \tsigma = '+str(sigma)+' km/s \nLorentzienne : S/(1+((x-x0)/(gamma/2))**2)\n B (G) \t||\t x0 (MHz) \t||\t S (1/s) \t||\t gamma (MHz)'
#        print 'coucou'
#        np.savetxt('fit_11-03_sigma='+str(sigma)+'_vo='+str(vo)+'.txt',np.array(fit),header=header,fmt='%10.6f')    

#
exp = np.loadtxt('Resultats_Lor.txt')
fit = {}
for i,vo in enumerate([0.8,1,2,3]):
    sigmas = [1.3,1.5,1.7,1.9]#[1.2,1.4,1.6,1.7,1.8]
    for sigma in sigmas:
        fit[sigma] = np.loadtxt('fit_11-03_sigma='+str(sigma)+'_vo='+str(vo)+'.txt')[:,1]
        fit[sigma] *= 1E6 # car exp en Hz et fit en MHz
        fit[sigma] += exp[0,1]# - fit[sigma][0]
    
    chi2 = np.zeros(len(sigmas))
    for i,sigma in enumerate(sigmas):
        chi2[i] = np.sum(((exp[1:,1]-fit[sigma])/exp[1:,2])**2)
        
    p = np.polyfit(sigmas,chi2,2)
    meilleur_sigma = -p[1]/(2*p[0])
    meilleur_chi2 = np.poly1d(p)(meilleur_sigma)
    print 'sqrt(chi2/(n-1)) =', np.sqrt(meilleur_chi2/13)
    print 'vo=',vo,'meilleur sigma : ',meilleur_sigma,' km/s'
    
#    plt.close()
#    plt.figure(i)
    xx = np.arange(1,2,0.01)
    plt.plot(xx,np.poly1d(p)(xx),label=str(vo)+'km/s')
    plt.plot(sigmas,chi2,'o')
    plt.legend()

#meilleur_fit = np.loadtxt('fit_10-03_sigma=1.64130978851.txt')
##meilleur_fit = np.loadtxt('fit_10-03_sigma=1.6.txt')
#meilleur_fit[:,1] *= 1E6
#meilleur_fit[:,1] += exp[0,1] - meilleur_fit[0,1]
#
#plt.figure(1)
##plt.plot(meilleur_fit[:,0],meilleur_fit[:,1])
#plt.plot(meilleur_fit[10:,0],meilleur_fit[10:,1])
#plt.errorbar(exp[:,0],exp[:,1],yerr=exp[:,2],linestyle=None)
#
##plt.plot(exp[:,0],exp[:,1],'+')
#plt.xlabel('Champ magnetique (G)')
#plt.ylabel('Resonance (Hz)')