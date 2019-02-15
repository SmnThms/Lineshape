# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:27:38 2017 @author: Simon
Based on the old 1S-3S program; derivation in Arnoult et al., EPJD 60 (2010)
Version dated Feb 15 2019, for Python3.6+

Structure of the program:
    1. Physical constants
    2. Basis (and changes of basis)
    3. Hamiltonians (and electric dipole matrix elements)
    4. Lineshape calculation
    5. Bonus functions (linewidths, Zeeman diagram...)
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import hyp2f1
from scipy.special import factorial as fact
from scipy.optimize import curve_fit
import time

#-----------------------------------------------------------------------------#
#                           1. Physical constants                             #
#-----------------------------------------------------------------------------#

h = 6.62607015e-34          # Planck constant, J.s
c = 299792458               # Speed of light in vacuum, m/s
qe = 1.602176634e-19        # Elementary charge, C
kB = 1.380649e-23           # Boltzmann constant, J/K

alpha = 7.2973525664e-3     # Fine structure constant                (CODATA14)
mN = 1.672621898e-27        # Nucleus mass, kg                       (CODATA14)
me = 9.10938356e-31         # Electron mass, kg                      (CODATA14)
gN = 5.585694702            # Landé g-factor of the nucleus          (CODATA14)
ge = 2.00231930436182       # Landé g-factor of the electron         (CODATA14)
mub = 1.3996245042          # Bohr magneton, MHz/G                   (CODATA14)
S = 1/2                     # Electron spin
I = 1/2                     # Nucleus spin
a0 = h/(2*np.pi*me*c*alpha) # Bohr radius, m

# Fine structure of the hydrogen atom (n,L,J), MHz
# (from physics.nist.gov/PhysRefData/HDEL/energies.html)
SF = {(1,0,1/2): -3288066857.1276, 
      (2,0,1/2): -822025443.9405, 
      (2,1,1/2): -822026501.7845, 
      (2,1,3/2): -822015532.7429,
      (3,0,1/2): -365343578.4560,
      (3,1,1/2): -365343893.3338,
      (3,1,3/2): -365340643.2445}
# Hyperfine splitting (n), MHz:
A_SHF = {1:1420.405751768, 3:52.6094446}
# Linewidths (n,L), MHz:
gamma = {(1,0):0, (3,0):1.004945452, (3,1):30.1917}
branching_ratio_3P = 0.11834

#-----------------------------------------------------------------------------#
#                                  2. Basis                                   #
#-----------------------------------------------------------------------------#

class Level:
    def __init__(self,n=False, S=False, L=False, I=False, J=False, F=False, 
                 mS=False, mL=False, mI=False, mJ=False, mF=False):
        self.n, self.S, self.L, self.I, self.J, self.F = n, S, L, I, J, F
        self.mS, self.mL, self.mI, self.mJ, self.mF = mS, mL, mI, mJ, mF

# Three basis are useful: {|L,mS,mL,mI>}, {|L,J,mJ,mI>} and {|L,J,F,mF>}
def LmSmLmI():
    list_levels = []
    for (n,L) in [(1,0),(3,0),(3,1)]:
        for mL in np.arange(-L,L+1,1):
            for mS in [-S,S]:
                for mI in [-I,I]:
                    list_levels.append(Level(n=n,L=L,mS=mS,mL=mL,mI=mI,S=S))
    return list_levels
            
def LJmJmI():
    list_levels = []
    for (n,L) in [(1,0),(3,0),(3,1)]:
        for J in np.unique(np.abs([L+S,L-S])):
            for mJ in np.arange(-J,J+1,1):
                for mI in [-I,I]:
                    list_levels.append(Level(n=n,L=L,J=J,mJ=mJ,mI=mI,I=I))
    return list_levels

def LJFmF():
    list_levels = []
    for (n,L) in [(1,0),(3,0),(3,1)]:
        for J in np.unique(np.abs([L+S,L-S])):
            for F in np.unique(np.abs([J+I,J-I])):
                for mF in np.arange(-F,F+1,1):
                    list_levels.append(Level(n=n,L=L,J=J,F=F,mF=mF,I=I))
    return list_levels

N = len(LJmJmI())  # dimension of our state space

# Change of basis:
def LSI_to_LJI(): 
    P = np.zeros((N,N))
    for m,i in enumerate(LmSmLmI()): # initial level
        for n,f in enumerate(LJmJmI()): # final level
            if i.L != f.L or i.mI != f.mI or i.n != f.n:
                P[n,m] = 0 # same mI and L required for both levels
            else:
                P[n,m] = clebsch(j1=i.L,m1=i.mL,j2=i.S,m2=i.mS,J=f.J,M=f.mJ)
    return P
def LJI_to_LSI():
    return LSI_to_LJI().transpose()
            
def LJI_to_LJF(): 
    P = np.zeros((N,N))
    for m,i in enumerate(LJmJmI()): # initial level
        for n,f in enumerate(LJFmF()): # final level
            if i.L != f.L or i.J != f.J or i.n != f.n:
                P[n,m] = 0 # same J and L required for both levels
            else:
                P[n,m] = clebsch(j1=i.J,m1=i.mJ,j2=i.I,m2=i.mI,J=f.F,M=f.mF)
    return P
def LJF_to_LJI():
    return LJI_to_LJF().transpose()
    
def convert(H,P): return np.dot(P,np.dot(H,P.T))

def clebsch(j1,m1,j2,m2,J,M):
    return (-1)**(j1-j2+M)*np.sqrt(2*J+1)*wigner3j(j1,j2,J,m1,m2,-M)
    
def wigner3j(j1,j2,j3,m1,m2,m3):
    if m1+m2+m3!=0: return 0
    if j1-m1!=np.floor(j1-m1) or j2-m2!=np.floor(j2-m2) \
       or j3-m3!=np.floor(j3-m3): return 0
    if j3>j1+j2 or j3<abs(j1-j2): return 0
    t1, t2, t3, t4, t5 = j2-m1-j3, j1+m2-j3, j1+j2-j3, j1-m1, j2+m2
    tmin, tmax = max(0, max(t1,t2)), min(t3, min(t4,t5))
    wigner = 0
    for t in np.arange(tmin,tmax+1,1):
        wigner += (-1)**t/(fact(t)*fact(t-t1)*fact(t-t2) \
                          *fact(t3-t)*fact(t4-t)*fact(t5-t))
    return wigner * (-1)**(j1-j2-m3) * np.sqrt(fact(j1+j2-j3)*fact(j1-j2+j3) \
              *fact(-j1+j2+j3)/fact(j1+j2+j3+1)*fact(j1+m1)*fact(j1-m1) \
              *fact(j2+m2)*fact(j2-m2)*fact(j3+m3)*fact(j3-m3))
           
def wigner6j(j1,j2,j3,J1,J2,J3):
    if abs(j1-j2)>j3 or j1+j2<j3 or abs(j1-J2)>J3 or j1+J2<J3 \
    or abs(J1-j2)>J3 or J1+j2<J3 or abs(J1-J2)>j3 or J1+J2<j3: return 0
    if 2*(j1+j2+j3)!=round(2*(j1+j2+j3)) or 2*(j1+J2+J3)!=round(2*(j1+J2+J3)) \
    or 2*(J1+j2+J3)!=round(2*(J1+j2+J3)) or 2*(J1+J2+j3)!=round(2*(J1+J2+j3)):
        return 0
    t1, t2, t3, t4 = j1+j2+j3, j1+J2+J3, J1+j2+J3, J1+J2+j3
    t5, t6, t7 = j1+j2+J1+J2, j2+j3+J2+J3, j1+j3+J1+J3
    tmin, tmax = max(0,max(t1,max(t2,max(t3,t4)))), min(t5,min(t6,t7))
    wigner = 0
    for t in np.arange(tmin,tmax+1,1):
        wigner += (-1)**t*fact(t+1)/(fact(t-t1)*fact(t-t2)*fact(t-t3) \
                   *fact(t-t4)*fact(t5-t)*fact(t6-t)*fact(t7-t))
    return wigner*np.sqrt(TriCoef(j1,j2,j3)*TriCoef(j1,J2,J3) \
                 *TriCoef(J1,j2,J3)*TriCoef(J1,J2,j3))

def TriCoef(a,b,c): return fact(a+b-c)*fact(a-b+c)*fact(-a+b+c)/(fact(a+b+c+1))

#-----------------------------------------------------------------------------#
#                             3. Hamiltonians                                 #
#-----------------------------------------------------------------------------#

# Electric dipole matrix elements, in the {|L,J,mJ,mI>} basis :
def R(n1,L1,n2,L2): # radial part (Bethe-Salpeter Sect. 63)
    L = max(L1,L2)
    if L2 is L: n2,n1 = n1,n2
    if n1==n2 and np.abs(L1-L2)==1: 
        return 3/2*n1*np.sqrt(n1**2-L**2)
    elif n1!=n2 and np.abs(L1-L2)==1:
        return (-1)**(n2-L)/(4*fact(2*L-1)) \
               *np.sqrt((fact(n1+L)*fact(n2+L-1)) \
                       /(fact(n1-L-1)*fact(n2-L))) \
               *(4*n1*n2)**(L+1)*(n1-n2)**(n1+n2-2*L-2)/(n1+n2)**(n1+n2) \
               *(hyp2f1(-n1+L+1,-n2+L,2*L,-4*n1*n2/(n1-n2)**2) \
                 -((n1-n2)/(n1+n2))**2 \
                  *hyp2f1(-n1+L-1,-n2+L,2*L,-4*n1*n2/(n1-n2)**2))
    else: return 0
    
def A(d,a,theta,polar=[-1,0,1]): # angular part
    k = 1 # order
    alpha = {0:np.cos(theta), 1:-np.sin(theta)/np.sqrt(2),
             -1:np.sin(theta)/np.sqrt(2)}
    return np.sum([alpha[q] * (-1)**(S+d.mJ+d.L+a.L-k) \
           * np.sqrt((2*d.J+1)*(2*a.J+1)*(2*d.L+1)*(2*a.L+1)) \
           * wigner6j(d.J,k,a.J,a.L,S,d.L) \
           * wigner3j(d.J,k,a.J,-d.mJ,q,a.mJ) \
           * wigner3j(d.L,k,a.L,0,0,0) \
           for q in polar])

# Fine and hyperfine structure :           
def H_SFHF(E0=SF[(3,0,1/2)]): # MHz (Glass or Hagel PhD thesis, Brodsky67) 
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
    return H # in the {|L,J,F,mF>} basis
    
def H_Zeeman(B): # MHz (Glass or Hagel PhD thesis)
    H = np.zeros((N,N))
    for n, niv in enumerate(LmSmLmI()):
        gS = ge*(1-alpha**2/(3*niv.n**2))
        H[n,n] = (gS*niv.mS + (1-me/mN)*niv.mL - gN*me/mN*niv.mI)*mub*B
        H[n,n] -= diamagnetic(niv.n,niv.L,niv.mL)*B**2
    return H # in the {|L,mS,mL,mI>} basis
    
def diamagnetic(n,L,mL): # MHz/G² (Delande PhD thesis p.189)
    r_perp_2 = n**2*(5*n**2+1-3*L*(L+1))*(L**2+L-1+mL**2)/((2*L-1)*(2*L+3))
    return r_perp_2 * qe**2*a0**2/(8*me*h) * 1e-14  # Hz/T² -> MHz/G²

def H_Stark(B): # MHz/(km/s) (Glass or Hagel PhD thesis)
    H = np.zeros((N,N))
    theta = np.pi/2  # motionnal E field perpendicular to the quantization axis
    for n, niv1 in enumerate(LJmJmI()):
        for m, niv2 in enumerate(LJmJmI()):
            if niv1.n==niv2.n and niv1.mI==niv2.mI:
                H[n,m] = R(niv1.n,niv1.L,niv2.n,niv2.L)*A(niv1,niv2,theta) \
                        *a0*qe*B/h * 1e-7  # Hz/(T*m/s) -> MHz/(G*km/s)
    return H # in the {|L,J,mJ,mI>} basis
    
def H_2photons(rabi=1): 
    H = np.zeros((N,N),dtype=complex)       
    for i,a in enumerate(LJmJmI()):
        for j,d in enumerate(LJmJmI()):
            if a.n!=d.n and a.L==d.L and a.mJ==d.mJ and a.mI==d.mI:
                H[i,j] = rabi
    return H # in the {|L,J,mJ,mI>} basis

#-----------------------------------------------------------------------------#
#                      4. Calculation of the lineshape                        #
#-----------------------------------------------------------------------------#

def thermal_velocity_flux(v,T): # v in km/s, T in Kelvin
    return (v*1e3)**3*np.exp(-(mN+me)*(v*1e3)**2/(2*kB*T))

def lkb_velocity_flux(v,sigma,v0): # v, sigma, v0 in km/s
    # (Olander70, Arnoult PhD thesis, Galtier PhD thesis)
    xd = 6.5e-6             # half length of the detection zone, km
    zr = 35e-6              # Rayleigh length, km
    tau = 1e-6/(2*np.pi)    # lifetime of the excited state, s    
    K = 0.01                # Knudsen number
    z = v/(np.sqrt(2)*sigma)
    psi = (z*np.exp(-z**2)+np.sqrt(np.pi)/2.*(1+2*z**2)*erf(z)) \
          /(np.sqrt(2*np.pi)*z**2)
    maxwell = 4./np.sqrt(np.pi)*z**2*np.exp(-z**2)
    olander = np.sqrt(np.pi)/2.*np.sqrt(erf(psi/(2*K)))/np.sqrt(psi/(2*K))
    arnoult = np.arctan((xd-v*tau)/zr)+np.arctan((xd+v*tau)/zr)
    return maxwell*olander*arnoult*np.exp(-v0/v)

def lineshape(B,frequency_scan,velocity_flux): # B in Gauss, frequencies in MHz
    start = time.time()
    velocities = np.linspace(0.1,10,100)   # in km/s
    H = H_SFHF() + convert(convert(H_Zeeman(B),LSI_to_LJI()),LJI_to_LJF())
    E1S,_   = np.linalg.eigh(H[:4,:4])     # sorted diagonalization
    E3S,M3S = np.linalg.eigh(H[4:8,4:8])
    E3P,M3P = np.linalg.eigh(H[8:,8:])
    E_B0,_  = np.linalg.eigh(H_SFHF()[:8,:8])
    M = np.block([ [M3S.T,np.zeros((4,N-8))], [np.zeros((N-8,4)),M3P.T] ])
    coef_Stark_by_v = convert(convert(H_Stark(B),LJI_to_LJF())[4:,4:],M)[4:,:4]
    hfs = {True:[1,2,3], False:[1,3]}[B<5] # relevant hyperfine components
    fluo,fluo_v = np.zeros(len(frequency_scan)),np.zeros((len(velocities),4))
    for i,delta in enumerate(frequency_scan):
        for j,v in enumerate(velocities):
            coef_Stark = coef_Stark_by_v*v
            for k in hfs: 
                detuning = (E1S[k]-E3S[k]) - (E_B0[k]-E_B0[4+k]) + delta 
                detuning += (v*1e3)**2/(2*c**2)*2.922742e9 # Doppler
                coupl = np.sum(coef_Stark[:,k]**2 \
                        /(-gamma[(3,1)]/2 + 1j*(detuning + E3S[k] - E3P)))
                BB = -coef_Stark[:,k]**2*(gamma[(3,0)]+gamma[(3,1)]) \
                     /((E3S[k]-E3P)**2+((gamma[(3,0)]+gamma[(3,1)])/2)**2)
                A = coupl - gamma[(3,0)]/2 + 1j*detuning
                CC = np.real(coef_Stark[:,k]**2 \
                     /(A*(detuning+E3S[k]-E3P+1j*gamma[(3,1)]/2) \
                     *(E3S[k]-E3P+1j*(gamma[(3,0)]+gamma[(3,1)])/2)))
                num = np.real(-1/A) - np.sum(CC*(1+BB/(gamma[(3,1)]-BB)))
                den = gamma[(3,0)] - np.sum(BB*(1+BB/(gamma[(3,1)]-BB)))
                pop3S = num/den
                pop3P = (CC-pop3S*BB)/(gamma[(3,1)]-BB)
                fluo_v[j,k] = gamma[(3,0)]*pop3S
                fluo_v[j,k] += branching_ratio_3P*gamma[(3,1)]*np.sum(pop3P)
                fluo_v[j,k] *= velocity_flux(v)
        fluo[i] = np.sum([quad(interp1d(velocities,fluo_v[:,k],kind='cubic'), \
                          min(velocities),max(velocities))[0] for k in hfs])
    normalization = quad(velocity_flux,min(velocities),max(velocities))[0]
    print(f'Lineshape computed for B = {B} G in {time.time()-start:.1f} s')
    return frequency_scan,fluo/normalization

def save_thermal_lineshape(B,T=300):
    freq,fluo = lineshape(B,np.arange(-4.8,4.8,0.1),
                          lambda v:thermal_velocity_flux(v,T))
    np.savetxt(f'thermal_fast_lineshape_{B}.txt',
               np.transpose([freq,fluo]),fmt='%.5e\t%.6e',
               header=f'T={T}K\nAtomic frequency (MHz)\tFluorescence')
    return freq,fluo
    
def save_lkb_lineshape(B,sigma,v0):
    freq,fluo = lineshape(B,np.arange(-4.8,4.8,0.1),
                          lambda v:lkb_velocity_flux(v,sigma,v0))
    np.savetxt(f'lkb_fast_lineshape_{B}_{sigma}_{v0}.txt',
               np.transpose([freq,fluo]),header=f'')
    return freq,fluo

def lorentzian_fit(freq,fluo):
    lor = lambda x,a0,a1,a2,a3: a0 + a1*(2/(a2*np.pi))/(1+((x-a3)/(a2/2))**2)
    p,_ = curve_fit(lor,freq,fluo/np.max(fluo),p0=[0,1,1,0]) # for freq in MHz
    return p[3]

#-----------------------------------------------------------------------------#
#                            5. Bonus functions                               #
#-----------------------------------------------------------------------------#
    
def Zeeman_diagram(list_B=np.linspace(0,200,100)):
    E_vs_B = []
    for B in list_B:  
        H = convert(H_SFHF(),LJF_to_LJI())
        H += convert(H_Zeeman(B),LSI_to_LJI())
        E,M = np.linalg.eigh(np.real(H))
        E_vs_B.append(E)
    plt.figure()
    for i in np.arange(4,N):
        plt.plot(list_B,[E_vs_B[j][i] for j in range(len(list_B))])
    offset1S = E_vs_B[0][0] + 6000 # For the sake of readability
    for i in np.arange(4):
        plt.plot(list_B,[E_vs_B[j][i] - offset1S for j in range(len(list_B))])
    plt.xlabel('B (G)')
    plt.ylabel('E (MHz)')
    return

def W(d,a,theta): # Probability of spontaneous emission, basis {|L,J,mJ,mI>}
    deltaE = (SF[(d.n,d.L,d.J)] - SF[(a.n,a.L,a.J)])*1e6 # in Hz
    if deltaE>0:
        return (R(d.n,d.L,a.n,a.L)*A(d,a,theta))**2 * deltaE**3 \
               *(4/3)*alpha*a0**2*(2*np.pi)**2/c**2*1e-6 # in MHz
    else: return 0
    
def linewidth(i,n_f=False): 
    # i is the initial level 
    # the decay can be restricted to levels of principal quantum number n_f
    coef = 0
    decay_basis = []
    for (n,L) in [(1,0),(2,0),(2,1)]:
        for J in [L+S,L-S]:
            for mJ in np.arange(-J,J+1,1):
                decay_basis.append(Level(n=n,L=L,J=J,mJ=mJ))
    for f in decay_basis:
        if not (n_f and f.n!=n_f):
            coef += np.sum([W(i,f,theta) for theta in [0,np.pi/2,np.pi/2]])
    return coef # in MHz

#print(f'Gamma_3S = {linewidth(Level(n=3,L=0,J=1/2,mJ=1/2)):.5f} MHz')
#print(f'Gamma_3P = {linewidth(Level(n=3,L=1,J=1/2,mJ=1/2)):.5f} MHz')
#print('Branching ratio from 3P to 2S =',
#       '%.5f'%(linewidth(Level(n=3,L=1,J=1/2,mJ=1/2),2) \
#              /linewidth(Level(n=3,L=1,J=1/2,mJ=1/2))))