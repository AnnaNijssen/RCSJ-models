# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 16:32:37 2026

@author: annae
"""

import numpy as np
import astropy.constants as c
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

def solve_RCSJ(t, y, Ib):
    phi, dphi = y
    ddphi = (2*e/(hbar*C))*(Ib - Ic*np.sin(phi) - (hbar/(2*e*R))*dphi)
    return [dphi, ddphi]

def second_JE(dphi):
    return (hbar/(2*e)) * np.mean(dphi)         # Mean for the time average due to DC voltage ????????

def plot_RCSJ_model(Ibs, voltages, name = None, ohm_voltages = None):
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
    
    plt.show()


def run_single_JJ_model():
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




















