# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:10:37 2023

@author: shwhassing
"""
import numpy as np

def sweep(dt, f0, f1, length, method = 'long'):
    """
    Generate a sweep function when given a starting frequency, ending frequency,
    signal length and time step. Three different functions are available that 
    generate a faster or slower sweep:
        long - includes each frequency roughly evenly
        short - includes the low frequencies more
        cos - uses a cosine function and fits between short and long
    
    Based on:
        https://www.recordingblogs.com/wiki/sine-sweep

    Parameters
    ----------
    f0 : float
        Starting frequency in Hz.
    f1 : float
        Ending frequency in Hz.
    length : float
        Signal length in seconds.
    dt : float
        Time step in seconds.
    method : str, optional
        Which function is used to calculate the sweep function. Possibilities
        are 'long', ' short' and 'cos'.
        . The default is 'long'.

    Returns
    -------
    t : np.ndarray
        Time array for the signal.
    u : np.ndarray
        Resulting sweep signal.

    """
    t = np.arange(0,length,dt)
    
    if method == 'long':
        u = np.sin(2*np.pi*(f0*t+(f1-f0)/(2*length)*t**2))
    elif method == 'short':
        u = np.sin(2*np.pi*f0*length * ((f1/f0)**(t/length)-1)/np.log(f1/f0))
    elif method == 'cos':
        u = np.cos((f0 + (f1-f0)/length * t)*t)
    
    return t, u

def ricker(dt, f):
    """
    Generates a Ricker wavelet when given a central frequency and time step. 
    Length of the wavelet is calculated so that the first and last time should
    be close to zero compared to the amplitude of the wavelet.

    Parameters
    ----------
    f : float
        Central frequency of the Ricker wavelet.
    dt : float
        Time step used for the Ricker wavelet.

    Returns
    -------
    t : np.ndarray
        Time array for the signal.
    u : np.ndarray
        Resulting Ricker wavelet.

    """
    length = np.sqrt(6)/np.pi/f*2.5
    
    # length = (np.floor(1.1/f/dt)*2+1)/2*dt
    
    t = np.arange(-length/2, (length-dt)/2, dt)
    u = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, u