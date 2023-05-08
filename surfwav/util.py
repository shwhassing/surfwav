# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:29:06 2023

@author: Sverre Hassing
"""

import numpy as np
from struct import calcsize
from sklearn.linear_model import LinearRegression
import os

def get_idx(value, min_val, dx):
    idx = np.round((value - min_val)/dx)
    
    if isinstance(idx, np.ndarray):
        return idx.astype(np.int32)
    return int(idx)

def freq_ft(signal, dt, n=None, axis=0, give='full'):
    """
    Calculate a Fourier transform of a real signal. 

    Parameters
    ----------
    signal : np.ndarray
        The evenly spaced time series for which the Fourier transform is 
        calculated.
    dt : float
        The distance between each time sample in seconds.
    n : int, optional
        Frequency axis length. If less than the signal length, it is cropped 
        and if more, it is padded with zeroes. The default takes the length
        of the signal. The default is None.
    give : str, optional
        Which part of the resulting spectrum to return. This simply cuts part
        the spectrum, which could be done with a single function, but is 
        available for convenience. The default is 'full'. Options are:
            - full
            Both the real and imaginary part
            - abs
            The absolute values of the complex numbers
            - real
            Only the real part
            - imag
            Only the imaginary part

    Raises
    ------
    ValueError
        If no valid value for 'give' is provided.

    Returns
    -------
    f : np.ndarray
        Frequency axis of the result.
    spec : np.ndarray
        Spectrum of the result.

    """
    
    if n == None:
        n = signal.shape[axis]
        
    spectrum = np.fft.rfft(signal, n=n, axis=axis)*dt
    f = np.fft.rfftfreq(n, dt)
    
    if give == 'full':
        return f, spectrum
    elif give == 'abs':
        return f, abs(spectrum)
    elif give == 'real':
        return f, np.real(spectrum)
    elif give == 'imag':
        return f, np.imag(spectrum)
    else:
        raise ValueError(f"'{give}' is not a valid value for give")

def full_ft(signal, dt, n=None, give='full'):
    """
    Convert the full Fourier transform of a given signal

    Parameters
    ----------
    signal : np.ndarray
        The evenly spaced time series for which the Fourier transform is 
        calculated.
    dt : float
        The distance between each time sample in seconds.
    n : int, optional
        Frequency axis length. If less than the signal length, it is cropped 
        and if more, it is padded with zeroes. The default takes the length
        of the signal. The default is None.
    give : TYPE, optional
        Which part of the resulting spectrum to return. This simply cuts part
        the spectrum, which could be done with a single function, but is 
        available for convenience. The default is 'full'. Options are:
            - full
            Both the real and imaginary part
            - abs
            The absolute values of the complex numbers
            - real
            Only the real part
            - imag
            Only the imaginary part

    Raises
    ------
    ValueError
        If no valid value for 'give' is provided.

    Returns
    -------
    f : np.ndarray
        Frequency axis of the result.
    spec : np.ndarray
        Spectrum of the result.

    """
    
    
    if n == None:
        n = len(signal)
    spectrum = np.fft.fft(signal, n=n)*dt
    f = np.fft.fftfreq(n, dt)
    
    if give == 'full':
        return f, spectrum
    elif give == 'abs':
        return f, abs(spectrum)
    elif give == 'real':
        return f, np.real(spectrum)
    elif give == 'imag':
        return f, np.imag(spectrum)
    else:
        raise ValueError(f"'{give}' is not a valid value for give")

def conv_endian(endian):
    """
    Convert the string "little" or "big" to "<" or ">" respectively. To get
    an endian indicator that can be used to write binary with pack

    Parameters
    ----------
    endian : str
        String indicating endian with text.

    Returns
    -------
    str
        String indicating endian with '<' or '>'.

    """
    if endian == 'little':
        return '<'
    elif endian == 'big':
        return '>'
    else:
        return endian
    
def check_binary_fit(value, bin_val):
    """
    If the specified value is larger than the limits of the specified binary
    format, return the saturated value. So for example when giving 1000000 as 
    's' or short, it can only represent values between -32768 and 32768. Then
    the function returns 32768. It is based on the standard sizes from:
        https://docs.python.org/3/library/struct.html
    
    Function is only really meant for packing integers the Seismic Unix files, 
    so it includes 'i', 'h', 'f' and 'q' and the unsigned variants.

    Parameters
    ----------
    value : int
        Value that will be encoded in binary.
    bin_val : str
        Binary type, must be a format that fits with the function pack from struct

    Returns
    -------
    int
        The input value either capped at the limits for the binary format or
        simply the original value.

    """

    if bin_val in ['x','c','b','B','?','n','N','e','f','d','s','p','P']: # Ignore non-integer formats
        return value
    elif bin_val in ['i','I','l','L','h','H','q','Q']:
        lim = 2**(calcsize(bin_val)*8-1)
    else:
        raise ValueError(f"bin_val ('{bin_val}') is not one of the supported formats")

    # Unsigned integers are indicated with capitals, so separate the two
    if bin_val.isupper():
        low_lim, upp_lim = 0, lim*2
    else:
        low_lim, upp_lim = -lim, lim
    
    # Return the saturated values
    if value < low_lim:
        return low_lim
    elif value >= upp_lim:
        return upp_lim - 1 # the minus 1 is to account for 0 if you were wondering like me
    # Or return just the starting value
    else:
        return value

def fk_transform(signal, dt, dx, max_f=None):
    """
    Compute the f-k transform of the supplied signal. Is assumed to have 
    variation over space over the first axis and time over the second axis. 

    Parameters
    ----------
    signal : np.ndarray
        Array for which the fk transform is computed.
    dt : float
        Time step of the data.
    dx : float
        Spatial step of the data.
    max_f : float, optional
        The maximum frequency to use. Defaults to the Nyquist frequency. The 
        default is None.

    Returns
    -------
    f : np.ndarray
        Frequency axis of the transform.
    k : np.ndarray
        Wavenumber axis of the transform.
    fk : np.ndarray
        Frequency-wavenumber transform of the input data.

    """
    # Get the shape of the original data
    nx = signal.shape[0]
    nt = signal.shape[1]
    
    # Compute the Fourier transform over time
    spectrum = np.fft.rfft(signal, axis=1)*dt
    f = np.fft.rfftfreq(nt, dt)
    
    if max_f == None:
        max_f = max(f)
    
    # Apply the maximum frequency
    mask = f <= max_f
    f = f[mask]
    spectrum = spectrum[:,mask]
    
    # Compute the wavenumber step
    spectrum = np.fft.fftshift(spectrum, axes=0)
    fk = np.fft.fftshift(np.fft.ifft(spectrum, axis=0)*dx*signal.shape[0], axes=0)
    
    # Compute the wavenumber axis
    k = np.fft.fftfreq(nx, dx)
    k = np.fft.fftshift(k)
    
    return f, k, fk

def fit_line(distances, elevations):
    """
    Fit a line through a set of points with linear regression. Used to estimate
    the elevation based on distance to some station. Can also be used to fit
    elevation to local coordinates.

    Parameters
    ----------
    distances : np.ndarray
        Array containing the distances from different stations to some station.
    elevations : np.ndarray
        Array containing the elevations of these stations.

    Returns
    -------
    coef : list or float
        The coefficient(s) of the linear regression fitting the elevations.
    intercept : float
        Intercept of the linear regression fitting the elevations.
    """
    # If there is only a single input value add a dimension so the regression
    # behaves
    if distances.ndim == 1:
        distances = distances[:,np.newaxis]
    # Set up a linear regression model
    model = LinearRegression()
    # Fit the model
    model.fit(distances, elevations)
    # Get the coefficients and intercept out
    coef = model.coef_
    intercept = model.intercept_
    
    if coef == 0 and np.all(distances == distances[0,:]):
        raise ValueError("Vertical line")
    
    return coef, intercept

def root_mean_square(array, axis=None):
    """
    Return the root-mean-square value of the given array

    Parameters
    ----------
    array : TYPE
        The array for which the rms value is compputed.
    axis : int, optional
        Over which axis to compute the rms value. By default, the array is
        flattened

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    square = np.square(array)
    mean = np.mean(square, axis=axis)
    return np.sqrt(mean)

def mkdir(path):
    """
    Make the folder at path if it not already exists

    Parameters
    ----------
    path : str
        Path to the folder.

    Returns
    -------
    None.

    """
    if not os.path.exists(path):
        os.mkdir(path)
        
def point_to_line(coef, intercept, points):
    """
    Project provided points onto the line defined by coef and intercept in 2D
    space. Line is defined as y = coef*x + intercept
    Based on:
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    Parameters
    ----------
    coef : float
        Coefficient of line.
    intercept : float
        Intercept with x=0 of the line.
    points : np.ndarray
        Array of shape [amt_points, 2] with the points that are projected onto
        the line.

    Returns
    -------
    points_on_line : np.ndarray
        Output array with each point in points projected onto the provided line.

    """
    a = coef
    b = -1
    c = intercept
        
    if points.shape == (2,):
        points = points[np.newaxis,:]
    
    points_on_line = np.array([(b*( b*points[:,0] - a*points[:,1]) - a*c)/(a**2+b**2),
                               (a*(-b*points[:,0] + a*points[:,1]) - b*c)/(a**2+b**2)])
    
    return points_on_line

def rfftcorrelate(signal1, signal2, axis=-1):
    
    shape = 2*signal1.shape[axis]
    
    spectrum1 = np.fft.rfft(signal1, shape, axis=axis)
    spectrum2 = np.fft.rfft(signal2, shape, axis=axis)
    
    return np.fft.fftshift(np.fft.irfft(spectrum1*np.conjugate(spectrum2), shape, axis=axis), axes=axis)

def rfftconvolve(signal1, signal2, axis=-1):
    
    shape = 2*signal1.shape[axis]
    
    spectrum1 = np.fft.rfft(signal1, shape, axis=axis)
    spectrum2 = np.fft.rfft(signal2, shape, axis=axis)
    
    return np.fft.fftshift(np.fft.irfft(spectrum1*spectrum2, shape, axis=axis), axes=axis)

def rfftcoherence(signal1, signal2, stab_val = 1e-2, axis=-1):
    
    shape = 2*signal1.shape[axis]
    
    spectrum1 = np.fft.rfft(signal1, shape, axis=axis)
    spectrum2 = np.fft.rfft(signal2, shape, axis=axis)
    
    coh = spectrum1*np.conjugate(spectrum2) / (abs(np.conjugate(spectrum1))*abs(spectrum2)+stab_val*np.max(abs(spectrum1)))
    
    return np.fft.fftshift(np.fft.irfft(coh, shape, axis=axis), axes=axis)