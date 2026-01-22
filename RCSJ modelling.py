# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 16:32:37 2026

@author: annae
"""

import numpy as np
import astropy.constants as c
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm


################ ADD MODEL THERMAL NOISE sqrt(4k_bT/R)??

# Font of plots
plt.rcParams['font.family'] = 'serif'

# Variables:
Ic = 30e-6      # A
R = 6         # Ohm
C = 1e-12       # F
#Ib = 30e-6      # A

# Constants:
hbar = c.hbar.value
e = c.e.value
Phi0 = c.h.value/(2*e)

@njit
def solve_RCSJ(t, y, Ib):
    phi, dphi = y
    ddphi = (2*e/(hbar*C))*(Ib - Ic*np.sin(phi) - (hbar/(2*e*R))*dphi)
    return [dphi, ddphi]

def second_JE(dphi):
    return (hbar/(2*e)) * np.mean(dphi)         # Mean for the time average due to DC voltage ????????

def plot_RCSJ_model(Ibs, voltages, split = False, name = None, ohm_voltages = None):
    if split == False:
        plt.scatter(Ibs*1e6, np.array(voltages)*1e6, color = 'darkblue', label = 'Model data')
        
        if ohm_voltages != None:
            plt.plot(Ibs*1e6, np.array(ohm_voltages)*1e6, color = 'red', label = "Ohm's law", linestyle = '--')
            plt.legend()
        
        plt.xlabel("Current [µA]", size = 11.5)
        plt.ylabel("Voltage [µV]", size = 11.5)
        plt.title("RCSJ model IV curve", size = 14)
        plt.grid()
        plt.tight_layout()
        
        if name != None:
            plt.savefig(name)
        
    else:
        half = len(Ibs)//2 + 1
        plt.scatter(Ibs[:half]*1e6, np.array(voltages[:half])*1e6, facecolor='blue', edgecolor='blue', label = 'Up-sweep')
        plt.scatter(Ibs[half:]*1e6, np.array(voltages[half:])*1e6, facecolor='none', edgecolor='black', label = 'Down-sweep')

        if ohm_voltages != None:
            plt.plot(Ibs*1e6, np.array(ohm_voltages)*1e6, color = 'red', label = "Ohm's law", linestyle = '--')
        
        plt.legend()
        plt.xlabel("Current [µA]", size = 11.5)
        plt.ylabel("Voltage [µV]", size = 11.5)
        plt.title("RCSJ model IV curve", size = 14)
        plt.grid()
        plt.tight_layout()
        
        if name != None:
            plt.savefig(name)
    
    plt.show()


def run_single_JJ_model():
    """
    This function models an example Josephson junction. 
    """
    integration_interval = [0, 1e-6]
    initial_state = [0,0]
    dt_max = 1e-9
    Ibs = np.linspace(0, 60e-6, 40)   # 0 to 40 micro_ampere
    voltages = []
    name = r"C:\Users\annae\Documents\Bachelor project\RCSJ model\Single JJ model.pdf"
    
    # Sweeping the current:
    for Ib in Ibs:
        solution = solve_ivp(solve_RCSJ, integration_interval, initial_state, args = (Ib,), max_step= dt_max)     # phi, dphi
    
        # We discard transient:
        steady_dphi = solution.y[1]
        #steady_dphi = dphi[len(dphi)//2:]     ????? NECESSARY????
            
        voltage = second_JE(steady_dphi)
        voltages.append(voltage)
    
    # Adding Ohm's law to the plot:
    ohm_voltages = []
    for Ib in Ibs:
        ohm_voltages.append(Ib * R)
    
    
    plot_RCSJ_model(Ibs, voltages, name, ohm_voltages)


def betaC(Ic, R, C):
    """This function calculates the Stewart-McCumber parameter.
    If betaC << 1, then the Josephson junction is overdamped.
    If betaC approx 1, then the Josephson junction is underdamped. Hystersis may occur.
    """    
    return 2*e*Ic*C*(R**2)/hbar

def run_JJ_model_with_memory():
    sweep_up = np.linspace(0, 40e-6, 40)
    sweep_down = sweep_up[::-1]
    current_sweep = np.concatenate((sweep_up[:-1], sweep_down))
    voltages_of_sweep = []
    
    integration_interval = [0, 5e-7]
    initial_state = [0,0]
    dt_max = 3e-9
        
    R = 20.0      # ohm
    C = 5e-12     # F
    
    bc = betaC(Ic, R, C)
    print(f"The Stewart-McCumber parameter is {np.round(bc,3)}.")
    
    for Ib in tqdm(current_sweep, desc = "Current sweep"):
        solution = solve_ivp(solve_RCSJ, integration_interval, initial_state, args = (Ib,), max_step= dt_max)     # phi, dphi
        
        # We discard transient:
        dphi = solution.y[1]
        steady_dphi = dphi[len(dphi)//2:]     
                
        voltage = second_JE(steady_dphi)
        voltages_of_sweep.append(voltage)
        
        initial_state = [solution.y[0, -1], solution.y[1, -1]]
    
    name = r"C:\Users\annae\Documents\Bachelor project\RCSJ model\Single JJ model with hysteresis.pdf"
    plot_RCSJ_model(current_sweep, voltages_of_sweep, name = name, split = True)
    
@njit
def SQUID_function(t, y, Ib, Phi_ext, Ic1, Ic2, R1, R2, C1, C2, L):
    
    #print(f'y = {y}')
    phi1, dphi1, phi2, dphi2 = y
    
    
    #Icirc = (1/L)*((hbar/(2*e))*(phi1 - phi2) - Phi_ext)
    delta_phi = phi1 - phi2
    Icirc = (Phi_ext - Phi0 * delta_phi / (2*np.pi)) / L

    I1 = Ib/2 + Icirc
    I2 = Ib/2 - Icirc
    
    
    ddphi1 = (2*e/(hbar * C1))*(I1 - Ic1 * np.sin(phi1) - hbar/(2*e*R1)*dphi1)
    ddphi2 = (2*e/(hbar * C2))*(I2 - Ic2 * np.sin(phi2) - hbar/(2*e*R2)*dphi2)

    return [dphi1, ddphi1, dphi2, ddphi2]
    

def betaL(Ic, L):
    return 2*L*Ic/Phi0

def run_SQUID_model_iterations(current_sweep, flux_sweep, Ic1, Ic2, R1, R2, C1, C2, L, name_data = None):
    bc1 = betaC(Ic1, R1, C1)
    bc2 = betaC(Ic2, R2, C2)
    bL1 = betaL(Ic1, L)
    bL2 = betaL(Ic2, L)
    print(f"""The Stewart-McCumber parameters are {np.round(bc1,3)} and {np.round(bc2,3)}. 
          The Beta_L variables are {np.round(bL1,3)} and {np.round(bL2,3)}.
          """)
    
    voltages_of_sweep = np.zeros((len(flux_sweep), len(current_sweep)))
    
    for i,Phi_ext in enumerate(tqdm(flux_sweep, desc = 'Flux sweep')):
        initial_state = [1e-3,0,-1e-3,0]
        for n,Ib in enumerate(current_sweep):
            solution = solve_ivp(SQUID_function, integration_interval, initial_state, args = (Ib,Phi_ext, Ic1, Ic2, R1, R2, C1, C2, L), max_step= dt_max)     # phi, dphi
            # We discard transients:
            dphi1 = solution.y[1]
            steady_dphi1 = dphi1[len(dphi1)//2:]     
                 
            # We discard transients:
            dphi2 = solution.y[3]
            steady_dphi2 = dphi2[len(dphi2)//2:]    
               
            voltage = second_JE((steady_dphi1 + steady_dphi2)/2)
            #voltage = (hbar/(2*e)) * np.mean(steady_dphi1+steady_dphi2)
            voltages_of_sweep[i,n] = voltage
            
            initial_state = [solution.y[0, -1], solution.y[1, -1], solution.y[2, -1], solution.y[3, -1]]
            #initial_state = [0,0,0,0]
            
        #name = r"C:\Users\annae\Documents\Bachelor project\RCSJ model\SQUID model.pdf"
        plot_RCSJ_model(current_sweep, voltages_of_sweep[i,:])
    if name_data != None:
        np.save(name_data, voltages_of_sweep)
        
    return voltages_of_sweep
#%%

#def run_SQUID_model_with_memory():
current_sweep = np.linspace(0, 40e-6, 30)
#sweep_down = sweep_up[::-1]
#current_sweep = np.concatenate((sweep_up[:-1], sweep_down))

flux_sweep = np.linspace(-1.2*Phi0, 1.2*Phi0, 30)
integration_interval = [0, 1e-7]
initial_state = [1e-3,0,-1e-3,0]
dt_max = 5e-6
    
# New variables:
Ic1 = Ic2 = 10e-6
R1 = R2 = 10.0
C1 = C2 = 1e-12
L = 100e-12

current_sweep = np.linspace(-25e-6, 25e-6, 20)
flux_sweep = np.linspace(0, 1.2*Phi0, 50)

name_data = None
#voltages_of_sweep = run_SQUID_model_iterations(current_sweep, flux_sweep, Ic1, Ic2, R1, R2, C1, C2, L, name_data)
voltages_of_sweep = np.load("voltage_sweep_data.npy")

volts = voltages_of_sweep.T * 1e6
Vmax = np.max(np.abs(volts))

plt.imshow(
    volts,
    vmin=-Vmax,
    vmax=Vmax,
    extent=[
        flux_sweep[0]/Phi0,
        flux_sweep[-1]/Phi0,
        current_sweep[0]*1e6,
        current_sweep[-1]*1e6
    ],
    cmap='inferno',   # or 'RdBu_r'
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='Voltage [µV]')
plt.ylabel('Bias current [$\mu$A]')
plt.xlabel(r'External flux ($\Psi_{\text{ext}} / \Psi_0$)')

name = r"C:\Users\annae\Documents\Bachelor project\RCSJ model\SQUID model.pdf"
plt.savefig(name)

plt.show()


# R1 = 6.0      # ohm
# C1 = 1e-12     # F
# R2 = 6.0      # ohm
# C2 = 1e-12     # F
# L = 5e-12      # H
# Ic1 = 30e-6      # A
# Ic2 = 30e-6      # A


# Variables that worked sort of:
# R1 = 20.0        # ohm
# R2 = 25.0        # ohm

# C1 = 5e-12       # F
# C2 = 6e-12       # F

# Ic1 = 10e-6      # A
# Ic2 = 10.2e-6      # A

# L = 20e-12       # H






#run_SQUID_model_with_memory()


# m = 10
# plt.plot(flux_sweep/Phi0, voltages_of_sweep[:, m]*1e6)
# plt.xlabel(r'$\Phi_{\rm ext}/\Phi_0$')
# plt.ylabel('Voltage [µV]')
# plt.title(f'V–Φ curve at I = {current_sweep[m]*1e6:.1f} µA')
# plt.grid()
# plt.show()  
    
# plt.imshow(voltages_of_sweep.T * 1e6,
#            vmin=0,
# vmax=np.percentile(voltages_of_sweep*1e6, 99),extent=[
#     flux_sweep[0]/Phi0,
#     flux_sweep[-1]/Phi0,
#     current_sweep[0]*1e6,
#     current_sweep[-1]*1e6
# ],
# cmap = 'inferno',
# origin='lower',
# aspect='auto'
# )
# plt.colorbar(label = r'Voltage [$\mu$V]')
# plt.ylabel('Bias current [$\mu$A]')
# plt.xlabel(r'External flux ($\Psi_{\text{ext}} / \Psi_0$)')
# plt.show()


