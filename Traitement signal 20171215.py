# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:30:24 2017
@author: Simon
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

Nfemto_Tisa = 1338660
Nfemto_1064 = 1126128
frep = 245000000+20396223/4     
f0 = 20e6         
figsize = (12,8)
f_centre = 1.7961684965e9 #MHz
critere = 5

class Run:
    def __init__(self,mois,jour,numero,plot=True,save=False):
        self.frequences,self.photons = np.zeros(31),np.zeros(31)
        self.pts_inclus = np.zeros(31)
        folder = '.\\'+mois+'\\'+jour+'\\'
        metadata = np.loadtxt(folder+'metadata.txt')
        chemin = folder+'run'+format_nb(numero)+'\\result.dat'
        l1 = open(chemin).readline().split('\t') 
        data = np.loadtxt(chemin)
        
        ### Pour tous les points ###
        self.nb_pts = len(data)
        self.f_AOM = data[:,l1.index('Marconi')]
        self.f_TiSa1 = data[:, l1.index('#freq_batt_896')] \
                       + (Nfemto_Tisa*frep + 2*f0)
        self.f_TiSa2 = data[:, l1.index('freq_batt_896_VCO')] \
                       + (Nfemto_Tisa*frep + 2*f0)
        self.f_10641 = -data[:, l1.index('freq_batt_1064_VCO1')] \
                       + (Nfemto_1064*frep + 2*f0)
        self.f_10642 = -data[:, l1.index('freq_batt_1064_VCO2')] \
                       + (Nfemto_1064*frep + 2*f0)
        self.f_V6 = 2*(data[:, l1.index('freq_batt_v6')] \
                       - data[:, l1.index('freq_batt_1064_VCO1')] \
                       + (Nfemto_1064*frep + 2*f0))
        self.f_Rb = data[:, l1.index('freq_batt_Rb')]
        self.f = (self.f_TiSa1+self.f_TiSa2)/2 \
                       + 2*(self.f_10641+self.f_10642)/2 \
                       - 2*self.f_AOM
        self.f *= 2*1e-6 # MHz en unités atomiques
        self.counts = data[:, l1.index('comptage')]
        self.t_fluoresceine = data[:, l1.index('205level')]
#        self.t_photodiode = data[:, l1.index('sortie205_photodiode')]
#        self.t_wattmetre = data[:, l1.index('sortie205_wattmetre')]

        ### Définition des seuils ###
        self.median,self.ecart,self.inf,self.sup = {},{},{},{}
        self.names = ['f_TiSa1','f_TiSa2','f_10641','f_10642','f_V6','t_fluoresceine']
        for name in self.names:
            self.median[name] = np.median(getattr(self,name))
            self.ecart[name] = np.median(abs(getattr(self,name)-self.median[name]))
            self.inf[name] = self.median[name]-critere*self.ecart[name]
            self.sup[name] = self.median[name]+critere*self.ecart[name]
        self.bons_pts = np.ones(self.nb_pts)
#        self.bons_pts[abs(self.f_TiSa1-self.f_TiSa2)>10] = 0
#        self.bons_pts[abs(self.f_10641-self.f_10642)>10] = 0
        for name in self.names:
            self.bons_pts[abs(getattr(self,name)-self.median[name]) \
                          > critere*self.ecart[name]] = 0
                          
        ### Par valeur de f_AOM ###
        self.liste_AOM = np.unique(self.f_AOM[self.f_AOM>0])
        for i in range(self.nb_pts):
            if self.bons_pts[i]==1:
                index = np.where(self.liste_AOM==self.f_AOM[i])
                self.frequences[index] += self.f[i]
                self.photons[index] += self.counts[i]
                self.pts_inclus[index] += 1
        self.frequences /= self.pts_inclus
        self.photons /= self.pts_inclus
        self.pts_faux = int(self.nb_pts-np.sum(self.pts_inclus))
        
        self.B = metadata[numero,1]
        self.P = metadata[numero,2]
        self.R = metadata[numero,3]
        self.qualite_run = metadata[4]
        self.reference = str(mois)+'-'+str(jour)+'-'+format_nb(numero)
        print 'Run',self.reference,' | ', self.pts_faux, \
              'pts non inclus sur',int(self.nb_pts)  
        self.plot_run(plot,save)
              
    def plot_run(self,plot=True,save=False):
        if plot:
            plt.figure(figsize=figsize)
            plt.subplot(1,3,1)
            plt.plot(self.frequences-f_centre,self.photons,'.')
            plt.xlabel(r'$Fr\'{e}quence\ atomique$ (+ '+str(f_centre)+' MHz)')
#            plt.ylabel('$Fluorescence$')
            for i,name in enumerate(self.names):
                plt.subplot(3,3,2+i+i//2)
                plt.plot(getattr(self,name),label=name)
                plt.plot(self.inf[name]*np.ones(self.nb_pts),'--',color='y')
                plt.plot(self.sup[name]*np.ones(self.nb_pts),'--',color='y')
                plt.ylim((self.inf[name]-8*self.ecart[name],
                          self.sup[name]+8*self.ecart[name]))
                plt.yticks([],[])
                plt.xticks([],[])
                plt.ylabel(name)
                if i==0:
                    plt.title(self.reference+'  B='+str(int(self.B)))
                if i==4:
                    plt.xlabel(r'$Crit\`{e}re$ : '+str(int(critere))+ \
                               r'$\ x\ \'{e}cart\ m\'{e}dian\ \`a\ la\ m\'{e}diane$')
                if i==5:
                    plt.xlabel(r'$Pts\ faux$ : '+str(self.pts_faux)+'/'+ \
                               str(int(self.nb_pts)))
            if save:
                plt.savefig(self.reference+'.png')
        
def signal(plage,plot=True,save=False):
    frequences,photons = np.zeros(31),np.zeros(31)
    k,pts = 0,0
    for date in plage:
        for numero in date[2]:
            run = Run(date[0],date[1],numero,plot=plot,save=save)
            frequences += run.frequences
            photons += run.photons
            k += 1
            pts += np.sum(run.pts_inclus)
    frequences /= k
    frequences -= f_centre
    photons /= k
    plt.figure(figsize=figsize)
    plt.plot(frequences,photons,'.k')
    p0 = [270,200,1.2,0]
    param,cov = curve_fit(lorentzienne,frequences,photons,p0=p0)
    erreur = np.sqrt(np.diag(cov))
    xx = np.linspace(np.min(frequences),np.max(frequences),100)
    plt.plot(xx,lorentzienne(xx,*param),
             color='midnightblue',alpha=0.3)
    dates = ''
    for date in plage:
        dates += date[0]+'-'+date[1]+' '
    titre = 'Moyenne des signaux du '+dates+'\n' \
          + '$f_0$ = '+str('%d' % (param[3]*1000+f_centre)) \
          + '$\pm$'+str('%d' % (erreur[3]*1000))+' kHz' \
          + ',  $\Gamma$ = '+str('%d' % (param[2]*1000)) \
          + '$\pm$'+str('%d' % (erreur[2]*1000))+' kHz' \
          + '\nsur '+str(k)+' runs, '+str(int(pts))+' points.'
    plt.title(titre)
    plt.xlabel(r'$Fr\acute{e}quence\ atomique$'+ \
               ' (+ '+str(f_centre)+' MHz)')
    plt.ylabel('$Fluorescence$')
    if save:
        plt.savefig(dates+'.png')
        print 'Figures sauvegardees'

def format_nb(i):
    if len(str(i))==1:
        return '00'+str(i)
    if len(str(i))==2:
        return '0'+str(i)  

def lorentzienne(x,a0,a1,a2,a3):
    return a0 + a1*(2/(a2*np.pi))/(1+((x-a3)/(a2/2))**2)

plt.close('all')
plage = [['12','08',[2,3,4,5,6,8,10]],
        ['11','28',[1,2,3,4,5,6,7,8,9,10,11]],
        ['12','01',[1,2,6,7,8,9,10]]]
#         [12,[4,[1,2,3,4,5,6,7]]]]
#plage = [['12','01',[1,2,6,7,8,9,10]]]    
#plage = [['11','28',[1,2,3,4,5,6,7,8,9,10,11]]]  
#plage = [['11','13',[1]]]          
signal(plage,plot=False,save=False)

#run = Run('12','08',6)                       