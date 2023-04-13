# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:16:02 2023

@author: Sverre Hassing
"""

import numpy as np
from time import time
from scipy.special import hankel1

def give_blocks(array, amt_blocks, padding=None):
    """
    Split the first axis of the given array into a set amount of blocks. 
    Yields data blocks with padding per block. By default no actual padding is 
    applied. If padding is set, that value is added at the end of blocks that
    are too short. 

    Parameters
    ----------
    array : np.ndarray
        Array for which the first axis is split into data blocks.
    amt_blocks : int
        The amount of data blocks.
    padding : float, optional
        Which value is set at the padded values. The default is None.

    Yields
    ------
    np.ndarray
        Array of size [nx, ..., amt_blocks], where nx is the size of the 
        original first axis of the data

    """
    # Get the indices for the start of each block
    idcs = np.linspace(0, array.shape[0], amt_blocks+1, dtype=int)
    # Initialise loop variable
    i = 0
    
    cont = True
    while cont:
        if padding is None:
            # With no padding, give the data of this block
            yield array[idcs[i]:idcs[i+1],...]
        else:
            # Calculate how long the block should be
            desired_length = int(np.ceil(array.shape[0]/amt_blocks))
            # Get the data in this block
            data_block = array[idcs[i]:idcs[i+1],...]
            
            if data_block.shape[0] == desired_length:
                # If the block is already the desired length, give it back
                yield data_block
            else:
                # Else get the shape of the array
                new_size = list(array.shape)
                # Set the length to the desired one
                new_size[0] = desired_length
                
                # Initialise an array with the new size
                block_padded = np.zeros(new_size) + padding
                # Put in the data of the block
                block_padded[:data_block.shape[0],...] = data_block
                
                yield block_padded
            
        i += 1
        
        # If the last block is done, stop
        cont = i < amt_blocks

def block_reshape(gather, amt_blocks, padding_method='end', padding_value = 0):
    """
    Reshape the data in the gather into blocks for the spatiospectral correlation
    matrix. 

    Parameters
    ----------
    gather : surfwav.Gather
        Gather containing the seismic data.
    amt_blocks : int
        Into how many blocks to convert the data.
    padding_method : str, optional
        How to pad the blocks of data. The default is 'end'. Options are:
            - end
            Add values at the end of the data before splitting into blocks
            - block
            Add values at the end of each block
    padding_value : float, optional
        With what value to pad the data. The default is 0.

    Raises
    ------
    NotImplementedError
        If an invalid method for padding_method is provided.

    Returns
    -------
    data : np.ndarray
        The data reshaped into [ntr, ceil(ns/amt_blocks), amt_blocks].

    """
    
    # Get the maximum length of a block
    length_block = int(np.ceil(gather.ns/amt_blocks))
    # How long should the arrays be to accommodate the blocks
    new_size = int(length_block*amt_blocks)
    
    if padding_method == 'end':
        # If the padding is at the end, add enough zeroes to the array
        data = np.zeros([gather.ntr, new_size]) + padding_value
        data[:,:gather.ns] = gather.data
        
        # Then reshape the data to obtain the blocks
        data = data.reshape([gather.ntr, length_block, amt_blocks])
    elif padding_method == 'block':
        # If each block is padded separately, initialise a new data array
        data = np.zeros([gather.ntr, int(new_size/amt_blocks), amt_blocks])
        
        # Go over each block in the data
        for i,block in enumerate(give_blocks(gather.data.T, amt_blocks, padding=padding_value)):
            data[:, :, i] = block.T
    else:
        raise NotImplementedError(f"This padding method ({padding_method}) has not been implemented, can be ['end', 'block']")
    
    return data

def spatiospectral_correlation_matrix(gather, amt_blocks=1, f_range=None, padding_method='end'):
    """
    Compute the spatiospectral correlation matrix for the Frequency Domain 
    Beamforming. For every evaluation frequency a square matrix as large as the
    amount of receiver locations is given. This gives an array of size 
    [nf, ntr, ntr]. The data can be split into blocks and the average over the
    blocks is taken. Does not seem to work well for field data. 
    
    See the function frequency_domain_beamforming for more context.

    Parameters
    ----------
    gather : surfwav.Gather
        Gather containing the data for which the sscm is calculated.
    amt_blocks : int, optional
        How many blocks of data are used. The default is 1.
    f_range : list, optional
        List containing the minimum and maximum frequency used for evaluation. 
        By default every frequencies between 0 and the Nyquist frequency are 
        used. It is recommended to limit this, further calculations become 
        expensive otherwise. The default is None.
    padding_method : str, optional
        Which padding method is used when splitting the data into blocks to keep
        the data at the same size. The default is 'end'. Options are:
            - end
            Add zeroes at the end of the data before splitting into blocks
            - block
            Add zeroes at the end of each block

    Returns
    -------
    np.ndarray
        A sscm for every evaluation frequency.

    """
    
    # Reshape the data into blocks
    data_blocks = block_reshape(gather, amt_blocks, padding_method=padding_method)
    
    # Get the frequency axis
    f = np.fft.rfftfreq(data_blocks.shape[1], gather.dt)
    # And compute the Fourier transform over the time axis
    spectrum = np.fft.rfft(data_blocks, axis=1)
    
    # Cut the frequency axis for the desired values
    if f_range is not None:
        mask = np.logical_and(f >= f_range[0], f <= f_range[1])
        f = f[mask]
        spectrum = spectrum[:,mask,:]
    
    # Compute the dot product between the spectrum and its complex conjugate 
    # for each combination of receiver positions
    sscm = spectrum[:,np.newaxis,...] * np.conjugate(spectrum[np.newaxis,...])
    
    # Then sum over each block and remove the leftover axis
    sscm = sscm.sum(axis=-1).squeeze()
    # Divide by the amount of blocks
    sscm /= amt_blocks
    
    # Rotate the variation over frequency to be the first axis, this is better
    # for later matrix multiplications, be careful playing around with it.
    # It's very easy to accidentally create a transposed SSCM for each frequency
    return f, sscm.swapaxes(-1,0).swapaxes(1,2)

# Alternative calculation of the sscm for speed tests
# def alt_sscm(data, f_index):
    
#     transform = np.fft.fft(data, axis=1)
    
#     transform = transform[:,f_index,:]
    
#     nchannels, nfrqs, nblocks = transform.shape
#     spatiospectral = np.empty((nchannels, nchannels, nfrqs), dtype=complex)
#     scm = np.zeros((nchannels, nchannels), dtype=complex)
#     tslice = np.zeros((nchannels, 1), dtype=complex)
#     tslice_h = np.zeros((1, nchannels), dtype=complex)
#     for i in range(nfrqs):
#         scm[:, :] = 0
#         for j in range(nblocks):
#             tslice[:, 0] = transform[:, i, j]
#             tslice_h[0, :] = np.conjugate(tslice)[:, 0]
#             scm += np.dot(tslice, tslice_h)
#         scm /= nblocks
#         spatiospectral[:, :, i] = scm[:]
    
#     return spatiospectral

def steering_vector(k, rec_pos, method='plane'):
    """
    Create a steering vector for the beamforming. See equations (12) and (14) 
    in Zywicki and Rix, 2005
    
    References:
    Zywicki, D.J. and Rix, G.J. 2005. Mitigation of near-field effects for 
    seismic surface wave velocity estimation with cylindrical beamformers. 
    Journal of Geotechnical and Geoenvironmental Engineering 131(8), p. 970-977

    Parameters
    ----------
    k : np.ndarray
        Wavenumbers for which the steering vectors are determined.
    rec_pos : np.ndarray
        Array containing the receiver positions.
    method : str, optional
        Which steering vector to give, can be 'plane' or 'cylindrical'. The 
        default is 'plane'.

    Raises
    ------
    ValueError
        If no valid value for method is given.

    Returns
    -------
    np.ndarray
        The steering vector.
    """
    
    kx = k*rec_pos
    
    # Get the steering vector
    if method == 'plane':
        return np.exp(-1.j * kx)
    elif method == 'cylindrical':
        return np.exp(-1.j * np.angle(hankel1(0,kx)))
    else:
        raise ValueError(f"Invalid value for method ({method}) is given, can be ['plane', 'cylindrical']")
    
def frequency_domain_beamforming(gather, 
                                 vel, 
                                 f_range=[1,100], 
                                 amt_blocks=1, 
                                 steering_method='plane',
                                 padding_method='end',
                                 weighting='sqrt',
                                 directional=False,
                                 verbose=0):
    """
    Apply Frequency-Domain Beamforming to the data in Gather at the specified 
    phase velocities in 'vel' and the frequency range given in 'f_range'. The
    method is an implementation of the one in Zywicki and Rix, 2005. As such,
    both plane wave and cylindrical steering vectors can be used. The first 
    take longer to compute, but the latter are more accurate in 3D media. 
    
    Splitting the data into blocks as described in the paper is implemented, 
    but does not seem to give good results. 
    
    References:
    Zywicki, D.J. and Rix, G.J. 2005. Mitigation of near-field effects for 
    seismic surface wave velocity estimation with cylindrical beamformers. 
    Journal of Geotechnical and Geoenvironmental Engineering 131(8), p. 970-977

    Parameters
    ----------
    gather : surfwav.Gather
        Gather containing the data that is converted.
    vel : np.ndarray
        Array containing the phase velocities that are evaluated in the FDBF.
    f_range : list, optional
        List containing the minimum and maximum frequency used. It is better to
        keep the limits relatively low, as there is generally not much 
        constructive information at higher frequencies, while the computation 
        time quickly increases. The default is [1,100].
    amt_blocks : int, optional
        The amount of blocks the data is split into. The default is 1.
    steering_method : str, optional
        Which method to use to create the steering vectors, can be 'plane' or 
        'cylindrical'. The default is 'plane'.
    padding_method : str, optional
        Which padding method to use when creating the data blocks. Can be padded
        per block ('block') or just at the end of the data ('end'). The default 
        is 'end'.
    weighting : str, optional
        How to weight the sscm. Based on the adaptation in swprocess. The 
        default is 'sqrt'.
    directional : bool, optional
        Not yet implemented. The default is False.
    verbose : int, optional
        How much output to give. The default is 0.

    Raises
    ------
    NotImplementedError
        If directional is set to True.

    Returns
    -------
    f : np.ndarray
        The frequency axis of the resulting data.
    fdbf_transform : np.ndarray
        The FDBF transformed data.

    """
    
    if verbose > 0:
        start = time()
    
    # Make sure the traces are in the right order
    gather = gather.sort_offset()
    
    offsets = np.array(gather.get_item('offset'))

    if not (np.all(offsets <= 0) or np.all(offsets >= 0)):
        print("WARNING: Gather contains both negative and positive offsets, is unlikely to give good results")    

    if np.sum(offsets >= 0) <= len(offsets)/2:
        # If mostly negative offsets are used, flip the gather
        gather = gather.reverse()
        
    # Get the sscm
    f, sscm = spatiospectral_correlation_matrix(gather,
                                                amt_blocks=amt_blocks, 
                                                f_range=f_range, 
                                                padding_method=padding_method)
    
    if verbose > 0:
        print(f"Calculated sscm after {round(time()-start, 3)} s")
    
    # XXX The theory says that the wavenumber vector should be multiplied with 
    # the receiver locations. Angular dependency is not a priority, so is 
    # ignored for now 
    if directional:
        raise NotImplementedError("Directional dependency of the wavenumber has not been implemented yet")
    else:
        # Get a test wavenumber for every frequency and velocity
        k = 2*np.pi*f[np.newaxis,:]/vel[:,np.newaxis]
        # Get all offsets
        offsets = np.array(gather.get_item('offset'))
        
        # Determine the weighting for the sscm
        if weighting == 'sqrt':
            sscm *= offsets[:,np.newaxis].dot(offsets[np.newaxis,:])
        else:
            # If no method is specified, use only ones
            sscm *= np.ones([gather.ntr, gather.ntr])
        
        # Calculate the steering vector
        steer_vec = steering_vector(k[:,:,np.newaxis], offsets[np.newaxis,np.newaxis,:], method=steering_method)
    
    # Multiply the sscm with the weights
    # sscm *= weights[np.newaxis,:,:]
    
    if verbose > 0:
        print(f"Determined steering vector after {round(time()-start, 3)} s")
    
    fdbf_transform = np.empty([len(vel), len(f), 1, 1], dtype=np.complex128)
    for i in range(len(f)):
        for j in range(len(vel)):
            fdbf_transform[j,i,0,0] = np.linalg.multi_dot([np.conjugate(steer_vec[j,i]), sscm[i,:,:], steer_vec[j,i]])
            
    if verbose > 1:
        print(f"Computed transform after {round(time()-start, 3)} s")
    
    # Remove any leftover dimensions
    return f, fdbf_transform.squeeze()