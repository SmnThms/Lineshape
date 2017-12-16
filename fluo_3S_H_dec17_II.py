# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:27:38 2017 @author: Simon
"""

##### STRUCTURE DU PROGRAMME #####
# I.   Définition des bases et matrices de passage
# II.  Définition des hamiltoniens, calcul matrice densité et forme de raie
# III. Exploitation (ajustement, enregistrement, affichage)

##### STRUCTURE DU FICHIER #######
# valeurs numériques
# niveaux d'énergie hydrogène
# transition dipolaire électrique
# hamiltoniens
# matrice densité
# forme de raie

from __future__ import division
from fluo_3S_H_dec17_I import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import hyp2f1
from scipy.linalg import expm
import time
from scipy.optimize import curve_fit

N = len(LJmJmI())

##### VALEURS NUMERIQUES #####
mub = 1.3996245042      # magnéton de Bohr en MHz/G        (CODATA14)
me = 9.10938356e-31     # masse de l'électron en kg        (CODATA14)
mN = 1.672621898e-27    # masse du proton en kg            (CODATA14)
qe = 1.6021766208e-19   # charge de l'électron en C        (CODATA14)
a0 = 0.52917721067e-10  # rayon de Bohr en m               (CODATA14)
h = 6.626070040e-34     # constante de Planck en J.s       (CODATA14)
gN = 5.585694702        # facteur de Landé du noyau        (CODATA14)
ge = 2.00231930436182   # facteur de Landé de l'électron   (CODATA14)
c = 299792458           # vitesse de la lumière dans le vide en m/s
alpha = 7.2973525664e-3 # constante de structure fine      (CODATA14)
eps0 = 8.85418782e-12   # permittivité du vide en A².s^4/(m³.kg)
#Rinf = me*alpha**2*c/(2*h) # constante de Rydberg en 1/m
# Structure fine (site du Nist, CODATA14) en MHz :
SF = {(1,0,1/2): -3288066857.1276, 
      (2,0,1/2): -822025443.9405, 
      (2,1,1/2): -822026501.7845, 
      (2,1,3/2): -822015532.7429,
      (3,0,1/2): -365343578.4560,
      (3,1,1/2): -365343893.3338,
      (3,1,3/2): -365340643.2445}
#SF = {(0,1/2):0, (1,1/2):-314.784, (1,3/2):2934.968}
# Lamb shift en MHz (n,L) (Galtier thèse) :
LS = {(1,0):8172.840, (2,0):1045.009, (2,1):0, (3,0):311.404, (3,1):0, (4,0):8172/4**3}
# Constante de couplage en MHz (n) :
A_SHF = {1:1420.405751768, 3:52.6094446}
# Facteur de Landé de l'électron (n) (Indelicato)
gS = {1:2.00228377, 3:2.0023152}
# Largeur des niveaux en MHz (n,L) :
gamma = {(1,0):0, (3,0):1.004945452, (3,1):30.192}

##### NIVEAUX D'ENERGIE HYDROGENE #####
def E(n,L,J): # Dirac + recul + Lamb Shift, pour Z=1, en MHz
    mu = me*mN/(me+mN)
    epsilon = J + 1/2 - np.sqrt((J+1/2)**2-alpha**2)
    inter = alpha**2/(n-epsilon)**2
    E = -(mu*c**2)*inter/(1+inter)/(1+1/np.sqrt(1+inter))
    E -= (mu**2*c**2)*alpha**4 / ((me+mN)*8*n**4)
    E *= 1e-6/h # conversion en MHz
    E += LS[(n,L)]
    return E

##### TRANSITION DIPOLAIRE ELECTRIQUE #####
def A(d,a,theta=np.pi/2): # (base LJmJmI)
    # pi/2 : Polarisation du champ motionnel normale à l'axe de quantification
    k, S = 1, 1/2 # ordre, spin
    alpha = {0:np.cos(theta), 1:-np.sin(theta)/np.sqrt(2),
             -1:np.sin(theta)/np.sqrt(2)}
    q = a.mI - d.mI
    return np.sum([alpha[q] * (-1)**(S+d.mJ) \
           * np.sqrt((2*d.J+1)*(2*a.J+1)*(2*d.L+1)*(2*a.L+1)) \
           * wigner6j(d.J,k,a.J,a.L,S,d.L) \
           * wigner3j(d.J,k,a.J,-d.mJ,q,a.mJ) \
           * wigner3j(d.L,k,a.L,0,0,0) \
           for q in [-1,0,1]]) # q = delta_mI
           
def R(n1,L1,n2,L2): # (Bethe-Salpeter p.262)
    L = max(L1,L2)
    if L2 is L:
        n2,n1 = n1,n2
    if n1==n2 and np.abs(L1-L2)==1:
        return 3/2*n1*np.sqrt(n1**2-L**2)
    elif n1!=n2 and np.abs(L1-L2)==1:
        return (-1)**(n2-L)/(4*factorial(2*L-1)) \
               *np.sqrt((factorial(n1+L)*factorial(n2+L-1)) \
                       /(factorial(n1-L-1)*factorial(n2-L))) \
               *(4*n1*n2)**(L+1)*(n1-n2)**(n1+n2-2*L-2)/(n1+n2)**(n1+n2) \
               *(hyp2f1(-n1+L+1,-n2+L,2*L,-4*n1*n2/(n1-n2)**2) \
                 -((n1-n2)/(n1+n2))**2 \
                  *hyp2f1(-n1+L-1,-n2+L,2*L,-4*n1*n2/(n1-n2)**2))
    else:
        return 0
    
#def W(d,a,theta): # base LJmJmI
#    deltaE = (E(d.n,d.L,d.J) - E(a.n,a.L,a.J))*1E6 # en Hz
#    if deltaE>0:
#        return (2/3)*a0**2*qe**2*(2*np.pi)**2/(eps0*h*c**3) \
#               *deltaE**3 \
#               *(R(d.n,d.L,a.n,a.L)*A(d,a,theta))**2 \
#               *1E-6 # en MHz
#    else:
#        return 0
    
def gam(d,n_a=False): # largeur du niveau (n_a: seul niveau de retombée)
    coef = 0
    for a in LJmJmI2(): # Avec n=2 (mais sans les différents mI)
        if not (n_a and a.n!=n_a):
            coef += np.sum([W(d,a,theta) for theta in [0,np.pi/2,np.pi/2]])
    return coef

##### HAMILTONIENS #####
def H_SFHF(E0=SF[(3,0,1/2)]): # en MHz (Hagel thèse, Brodsky67, Glass thèse) 
    # base = 'LJFmF' 
    H = np.zeros((N,N))
    for n, niv1 in enumerate(LJFmF()):
        for m, niv2 in enumerate(LJFmF()):
            if n == m: # (I·J)
                H[n,m] = SF[(niv1.n, niv1.L, niv1.J)] - E0
                H[n,m] += (3/16)*A_SHF[niv1.n] \
                  *(niv1.F*(niv1.F+1)-niv1.J*(niv1.J+1)-niv1.I*(niv1.I+1)) \
                  /(niv1.J*(niv1.J+1)*(niv1.L+1/2))
            if niv1.L != 0 and niv1.J != niv2.J \
            and niv1.L==niv2.L and niv1.F==niv2.F and niv1.mF==niv2.mF: # (I·L)
                H[n,m] = (3/16)*A_SHF[3] \
                  *(-1)**(2*niv1.J+niv1.L+niv1.F+niv1.I+3/2) \
                  *np.sqrt((2*niv1.J+1)*(2*niv2.J+1) \
                          *(2*niv1.I+1)*(niv1.I+1)*niv1.I \
                          *(2*niv1.L+1)*(niv1.L+1)*niv1.L) \
                  *wigner6j(niv1.F,niv1.I,niv2.J,1,niv1.J,niv1.I) \
                  *wigner6j(niv1.L,niv2.J,1/2,niv1.J,niv1.L,1) \
                  /(niv1.L*(niv1.L+1)*(niv1.L+1/2))
    return H 
    
def H_Zeeman(B): # en MHz (Hagel thèse, Glass thèse)
    # base = 'LmSmLmI'
    H = np.zeros((N,N))
    for n, niv in enumerate(LmSmLmI()):
        H[n,n] = (gS[niv.n]*niv.mS + (1-me/mN)*niv.mL - gN*me/mN*niv.mI)*mub*B
        H[n,n] -= diamagnetique(niv.n,niv.L,niv.mL)*B**2
    return H
    
def diamagnetique(n,L,mL): # en MHz/G² (Delande thèse)
    r_perp_2 = n**2*(5*n**2+1-3*L*(L+1))*(L**2+L-1+mL**2)/((2*L-1)*(2*L+3))
    return r_perp_2 * qe**2*a0**2/(8*me*h) * 1e-14  # Hz/T² -> MHz/G²

def H_Stark(B): # en MHz/(km/s) (Hagel thèse, Glass thèse)
    # base = 'LJmJmI'
    H = np.zeros((N,N))
    for n, niv1 in enumerate(LJmJmI()):
        for m, niv2 in enumerate(LJmJmI()):
            if niv1.mI != niv2.mI:
                H[n,m] = 0
            else:
                H[n,m] = R(niv1.n, niv1.L, niv2.n, niv2.L) \
                        *A(niv1,niv2) \
                        *a0*qe*B/h * 1e-7  # Hz/(T*m/s) -> MHz/(G*km/s)
    return H
    
def H_2photons(rabi):
    # base = 'LJmJmI'
    H = np.zeros((N,N),dtype=complex)       
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            if a.n!=d.n and a.L==d.L and a.mJ==d.mJ and a.mI==d.mI:
                H[i,j] = rabi
    return H
    
def convert(H,P):
    return np.dot(P,np.dot(H,P.transpose()))

##### MATRICE DENSITE #####
def matrice_densite(f=0,B=0,v=3,rabi=1E-3,t=10): 
    # base = 'LJmJmI'
    # hamiltonien
    H = np.zeros((N,N),dtype=complex)
    H += convert(H_SFHF(),LJF_vers_LJI()) \
      + convert(H_Zeeman(B),LSI_vers_LJI()) \
      + H_Stark(B)*v \
      + H_2photons(rabi)
    
    # désexcitation spontanée
    B = np.zeros((N**2,N**2))
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            B[i,j] = -1/2*(gamma[(a.n,a.L)] + gamma[(d.n,d.L)])*2*np.pi
    
    # fréquence de balayage
    for i,u in enumerate(LJFmF()):
        if getattr(u,'n')==1 and getattr(u,'mF')==1:
            E1S = H_SFHF()[i,i]
        if getattr(u,'n')==3 and getattr(u,'L')==0 and getattr(u,'mF')==1:
            E3S = H_SFHF()[i,i]
    f += (1 + (v*1E3)**2/(2*c**2))*(E3S-E1S) # avec v en km/s
    
    # référentiel tournant
    C = np.zeros((N**2,N**2),dtype=complex)
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            if d.n==1 and a.n==3:
                C[i,j] = -2j*np.pi*f
            if d.n==3 and a.n==1:
                C[i,j] = 2j*np.pi*f
    
    # réécriture du système en dX/dt=AX        
    A = np.zeros((N**2,N**2),dtype=complex)
    k = 0
    for i in range(N):
        for j in range(N):
            A_ij = np.zeros((N,N),dtype=complex)
            A_ij[:,j]  = -2j*np.pi*H[i,:].transpose()
            A_ij[i,:] -= -2j*np.pi*H[:,j].transpose()
            A_ij[i,j] += B[i,j] - C[i,j]
            A[k,:] = A_ij.reshape((1,N**2))
            k += 1
                            
    # résolution du système
    X0 = np.zeros((N**2,1),dtype=complex)
    for k in range(4):
        X0[k*(N+1),0] = 0.25
    X = np.dot(expm(t*A),X0)
    return X.reshape((N,N))
    
##### FORME DE RAIE #####    
def coefv(v,sigma,vo):      #(Olander70, Arnoult thèse, Galtier thèse)
    xd = 6.5e-6             # taille de la zone de détection/2 en km
    zr = 35e-6              # longueur de Rayleigh en km
    taue = 1e-6/(2*np.pi)   # durée de vie en s    
    z = v/(np.sqrt(2)*sigma)
    psi = (z*np.exp(-z**2)+np.sqrt(np.pi)/2.*(1+2*z**2)*erf(z)) \
          /(np.sqrt(2*np.pi)*z**2)
    K = 0.01
    maxwell = 4./np.sqrt(np.pi)*z**2*np.exp(-z**2)
    olander = np.sqrt(np.pi)/2.*np.sqrt(erf(psi/(2*K)))/np.sqrt(psi/(2*K))
    olivier = np.arctan((xd-v*taue)/zr)+np.arctan((xd+v*taue)/zr)
    return maxwell*olander*olivier*np.exp(-vo/v)

def forme_de_raie0(B,sigma,v0,nb_pts=50):
    debut = time.time()
    balayage = np.linspace(-5,5,nb_pts)    # en MHz
    vitesses = np.linspace(0.1,10.1,11)    # en km/s (v non nul pour coefv)
    normalisation = quad(lambda x:coefv(x,sigma,v0),0.1,10.1)[0] 
    fluo = np.zeros(len(balayage))
    for i,delta in enumerate(balayage):
        for j,v in enumerate(vitesses):
            fluo_v = np.zeros(len(vitesses))
            pop = np.diag(matrice_densite(f=delta,B=B,v=v)).real
            for k,d in enumerate(LJmJmI()):
                for a in LJmJmI2():
                    for theta in [0,np.pi/2,np.pi/2]:
                        l_fluo = 1e9/(Rinf*abs(1/d.n**2-1/a.n**2))
                        if abs(l_fluo-656)<1:
                            fluo_v[j] += np.sqrt(pop[k])*ampl_fluo(d,a,theta)
#            fluo_v[j]  = gamma[(3,0)]*np.sum(pop[4:8]) \
#                         + gamma[(3,1)]*np.sum(pop[8:])
            fluo_v[j] *= coefv(v,sigma,v0)
        fluo[i] = quad(interp1d(vitesses,fluo_v,kind='cubic'),0.1,10.1)[0]
        print i
    fluo /= normalisation
    print 'Calcul fini pour B =',B,', sigma =',sigma,', v0 =',v0, \
          ', en ',int(time.time()-debut),' s'
    return balayage,fluo*1000
    
def forme_de_raie_2(B=170,nb_pts=50):
    debut = time.time()
    balayage = np.linspace(-3,3,nb_pts) # en MHz
    fluo = np.zeros(len(balayage))
    for i,delta in enumerate(balayage):
        rho = matrice_densite(f=delta,B=B)
        for a in LJmJmI2():
            terme = 0
            for k,d in enumerate(LJmJmI()):
                for theta in [0,np.pi/2,np.pi/2]: # détection toutes directions
#                    terme = 0
                    deltaE = SF[(d.n,d.L,d.J)] - SF[(a.n,a.L,a.J)]
                    if deltaE>0 and abs(c*1e3/deltaE-656)<1: # filtre à 656 nm
#                        print abs(c*1e3/deltaE-656)
#                        if d.L==0:#1 and d.J==3/2:
                            terme += np.sqrt(abs(rho[k,k]))\
                                     *np.exp(1j*np.angle(rho[k,0]))\
                                     *deltaE**(3/2)\
                                     *R(d.n,d.L,a.n,a.L)*A(d,a,theta)
#                            fluo[i] += abs(terme)**2
            fluo[i] += abs(terme)**2
        print i
    proba_fluo = fluo/np.max(fluo)#/2
    print 'Calcul fini en ',int(time.time()-debut),' s'
    return balayage,proba_fluo

def lorentz(x,x0,S,gamma,offset):
    return S/(1+((x-x0)/(gamma/2))**2)+offset

#freq,raie = forme_de_raie_2()
plt.figure(3)
#plt.plot(freq,raie)
plt.plot(freq,fluo2)

#H1 = np.diag(H_SFHF())[4:]
#plt.figure(2)
#plt.plot(H1-H2)
#plt.figure(3)
#plt.plot(H2)
#H3 = np.diag(H_Zeeman(170))[4:]
#plt.figure(2)
#plt.plot(H4-H3)
#H5 = np.diag(H_Stark(170))[4:]
#plt.figure(2)
#plt.plot(H6-H5)
#
#freq,fluo1 = forme_de_raie_2()
#plt.close('all')
#plt.figure(3)
##freq,fluo2=forme_de_raie_2()
#p1,e = curve_fit(lorentz,freq,fluo1) #1 : le bon
#f1 = p1[0]
#p2,e = curve_fit(lorentz,freq,fluo2) #2 : somme des probas
#f2 = p2[0]
#print 'delta f centrale = ',f1-f2,' MHz'
#plt.plot(freq,lorentz(freq,*p1))
#plt.plot(freq,fluo1,'--')
#plt.plot(freq,lorentz(freq,*p2))
#plt.plot(freq,fluo2,'--')
#plt.figure(4)
#plt.plot(freq,fluo1-fluo2)
#plt.figure(3)
#plt.plot(freq,fluo1,'-')
#plt.figure(5)
#plt.plot(freq,fluo2,'-')
#plt.figure(6)
#plt.plot(freq,fluo8)


#plt.plot(freq,fluo,'.r',label='$|\sum$ ampl.|$^2$')#'|$\sum_f$ amplitudes ac $\phi$|^2')
#plt.legend()
#plt.title('B=170 G, v=3 km/s')
#plt.xlabel('frequence atomique (MHz)')

##### Test rho(f) #####
#freq = np.linspace(-2,2,50)
#pop3S = np.zeros(len(freq))
#for i,f in enumerate(freq):
#    pop3S[i] = abs(np.sum(np.diag(matrice_densite(f=f,B=0))[4:8]))
#plt.close('all')
#plt.figure()
#plt.plot(freq,pop3S)

##### Proba de désexcitations 4S->2S #####
#N3S = Niveau(n=3, L=0, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 )
#N2P = Niveau(n=2, L=1, J=3/2, mJ=3/2,  I=1/2 )
#N3P = Niveau(n=3, L=1, J=3/2, mJ=3/2,  mI=1/2,  I=1/2 )
#N4S = Niveau(n=4, L=0, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 )
#N2S = Niveau(n=2, L=0, J=1/2, mJ=1/2,  mI=1/2,  I=1/2 )
#print gam(N4S), 'MHz'
#print gam(N4S,3)/gam(N4S) * gam(N3P,2)/gam(N3P)
#print gam(N4S,3)/gam(N4S)
#print gam(N4S,2)/gam(N4S)

##### Michele Glass p.47 #####
#def D(L1,J1,J2,mJ,theta):
#    terme = 0
#    d=Niveau(L=L1,J=J1,mJ=mJ,S=1/2,I=1/2,mI=1/2)
#    for a in LJmJmI2():
#        if abs(a.L-d.L)==1:
#            terme += A(d,a,theta=0)
#    return terme**2
#print D(0,1/2,1/2,1/2,np.pi/2)