# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:44:02 2023

@author: Sverre Hassing
"""

import numpy as np
from surfwav.header import Header
from surfwav.gather import Gather
from surfwav.SU_write import write_su, read_header

def make_su_header(data,dt,x):
    """
    Creates a header for a SU file and fills in some information. For possible
    entries see class Header. 

    Parameters
    ----------
    data : np.ndarray
        Array containing the numerical information. Should be organised so that
        it has shape [amt. of traces x amt. of time steps]
    dt : float
        Time between samples in seconds.
    x : np.ndarray
        Location of each trace in desired distance unit.

    Returns
    -------
    trace_headers : Header
        Header file with the relevant information placed in.

    """
    # Initialise list for the headers
    trace_headers = []
    
    # Go over each row in the data
    for i,line in enumerate(data):
        # Initialise an empty header
        su_header = Header()
        
        # Set the relevant quantities
        su_header.dt        = int(np.floor(dt*1e6))
        su_header.d1        = dt
        su_header.ns        = data.shape[1]
        # su_header.trace     = data[i,:]
        su_header.tracl     = i+1
        su_header.offset    = int(round(x[i]))
        su_header.counit    = 1
        
        # Add the header to the list
        trace_headers.append(su_header)
    
    return trace_headers

def make_su(filename, data, dt, x, endian = 'little', method_offset='line'):
    """
    Function that makes a Seismic Unix .su file when given raw data. 

    Parameters
    ----------
    filename : str
        Path to output file.
    data : np.ndarray
        Array containing the data. Can be either 1-dimensional or if 2D, should
        have shape [amt. of traces x amt. of samples]. 
    dt : float
        Time step of the data, can be distance for velocity model data.
    x : np.ndarray
        Array containing the locations of each trace. 
    endian : str, optional
        Endian format for the data. Can be the strings 'little' or 'big' or the
        identifiers '<' or '>'. The default is 'little'.

    Returns
    -------
    None.

    """
    
    # If the data is 1-dimensional, add an axis at the front
    if data.ndim == 1:
        data = data[np.newaxis,:]
    
    # If there is only a single location, make x into an array to fit with later
    # indexing
    if np.isscalar(x):
        x = np.array([x])
    
    # Check if the amount of traces and locations is the same
    if data.shape[0] != x.shape[0]:
        raise IndexError(f"Amount of traces ({data.shape[0]}) is not the same as the amount of location values ({x.shape[0]})")
    
    # Make the headers for the SU file
    headers = make_su_header(data, dt, x)
    
    gather = Gather(data, headers, method_offset=method_offset)
    
    # Write the data to the file
    write_su(filename, gather, endian)

def make_su_model(filename, data, dx, dz, origin=(0,0), endian = 'little'):
    """
    Function that makes a Seismic Unix .su file when given raw data. 

    Parameters
    ----------
    filename : str
        Path to output file.
    data : np.ndarray
        Array containing the data. Can be either 1-dimensional or if 2D, should
        have shape [amt. of traces x amt. of samples]. 
    dt : float
        Time step of the data, can be distance for velocity model data.
    x : np.ndarray
        Array containing the locations of each trace. 
    endian : str, optional
        Endian format for the data. Can be the strings 'little' or 'big' or the
        identifiers '<' or '>'. The default is 'little'.

    Returns
    -------
    None.

    """
    
    # If the data is 1-dimensional, add an axis at the front
    if data.ndim == 1:
        data = data[np.newaxis,:]
    
    # Initialise list for the headers
    headers = []
    
    # Go over each row in the data
    for i,line in enumerate(data):
        # Initialise an empty header
        header = Header()
        
        # Set the relevant quantities
        header.ns        = data.shape[1]
        header.tracl     = i+1
        header.counit    = 1
        header.f1 = origin[1]
        header.f2 = origin[0]
        header.d1 = dz
        header.d2 = dx
        
        # Add the header to the list
        headers.append(header)
    
    gather = Gather(data, headers, method_offset='simple')
    
    # Write the data to the file
    write_su(filename, gather, endian)
    
def stream_to_gather(stream):
    """
    Convert an obspy stream to a Gather object

    Parameters
    ----------
    stream : obspy.Stream
        Stream that will be converted to Gather.

    Returns
    -------
    Gather
        Converted Gather.

    """
    
    # Initialise header list and data array
    headers = []
    data = np.zeros([len(stream),stream[0].stats.npts])
    
    # Go over all traces in stream
    for i, trace in enumerate(stream):
        # Get the data format
        fmt = trace.stats._format
        # And the header from obspy
        obspy_header = getattr(trace.stats, fmt.lower())
        
        # Read the endian and bytes from the header
        endian = obspy_header['trace_header']['endian']
        byte_header = obspy_header['trace_header']['unpacked_header']
        
        # Read the bytes
        headers.append(read_header(byte_header, endian))
        
        # Enter the data
        data[i,:] = trace.data
    
    # Make the gather
    return Gather(data, headers)