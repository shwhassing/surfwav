# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:21:44 2023

@author: Sverre Hassing
"""

import numpy as np
from scipy.signal import convolve2d
from copy import deepcopy
from surfwav.header import Header
from surfwav.util import fit_line, freq_ft, fk_transform, point_to_line, rfftcorrelate, rfftcoherence, rfftconvolve
from surfwav.SU_write import write_su, read_su
from surfwav.FDBF import frequency_domain_beamforming
from surfwav.plot import disp_power, plot_section_map, plot_section_wiggle, plot_ft, plot_fk
import matplotlib.pyplot as plt

class Gather:
    
    def __init__(self, data=None, headers=None, filename=None, tol=1e-5, method_offset='line'):
        
        # If an empty gather is initialised, just set the data to zeroes and
        # the type to empty
        if data is None and headers is None and filename is None:
            self.data = np.zeros(1)
            self.headers = [Header()]
            
            self.shape = [1]
            self.type = 'empty'
        # If the data is provided as arrays, use those
        elif ((data is not None and headers is not None) or filename is not None):
            
            # If a filename is provided, set the data from this file
            if filename is not None:
                data, headers = read_su(filename)
            
            if data.shape[0] == 0 and len(headers) == 0:
                self.data = np.zeros(1)
                self.headers = [Header()]
                
                self.shape = [1]
                self.type = 'empty'
            else:
            
                # If the data has a single dimension, set up a first axis, so the 
                # shape is constant
                if data.ndim <= 1:
                    data = data[np.newaxis,:]
                # If only a single header is provided, put it in a list for 
                # consistency
                if isinstance(headers,Header):
                    headers = [headers]
                   
                # Check if the amount of headers is the same as the amount of traces
                if len(headers) != data.shape[0]:
                    raise IndexError(f"Amount of headers ({len(headers)}) and traces ({data.shape[0]}) is not equal")
                
                # Now set some values for the gather
                self.headers = headers
                self.data = data
                self.ns = data.shape[1]
                self.shape = data.shape
                self.ntr = data.shape[0]
                self.dt = headers[0].dt/1e6
                self.t = np.linspace(0,self.dt*self.ns, self.ns, endpoint=False)
                self.tol = tol
                self.upd_offsets(method_offset)
                self._update()
                # self.type = self._check_type(tol=self.tol)
        else:
            raise ValueError("No valid combination of inputs was provided")
        
    def __repr__(self):
        return "Gather(data, headers)"
    
    def __str__(self):
        if self.type == "empty":
            return "Empty gather"
        else:
            return f"{self.type.capitalize()} type gather with {len(self)} traces of {self.ns} samples with timestep {self.dt*1e3} ms"
        
    def __len__(self):
        # As the amount of traces should be the same as the amount of headers
        # when loading the data, the amount of traces is simply used
        if self.type == 'empty':
            return 0
        else:
            return len(self.headers)
    
    def __getitem__(self, idx):
        
        # Set up the new data
        new_data = self.data[idx,:]
        
        # If a list or array with booleans is used as a masking index, each
        # item is added according to the list
        
        try:
            new_headers = self.headers[idx]
        except TypeError:
            if isinstance(idx, np.ndarray):
                if idx.dtype == bool:
                    new_headers = [x for i,x in enumerate(self.headers) if idx[i]]
                elif idx.dtype.kind == 'i':
                    new_headers = [self.headers[i] for i in idx]
            else:
                TypeError(f"Indexing with type ({type(idx)}) is not supported")
        
        # Set up a gather with the sliced data
        new_gather = Gather(new_data, deepcopy(new_headers))
        
        return new_gather
    
    def __iter__(self):
        # When iterating over the gather, provide each line of the array together
        # with its header
        return zip(self.data,self.headers)
    
    def __setitem__(self, key, value):
        self.data[key,:] = value[0]
        self.headers[key] = value[1]
        
        return self
    
    def __add__(self, *args):
        """
        Add traces to the gather. Can be done by providing the new data in three
        ways. The first is by adding another Gather to the object. The second
        is by providing a header and a an array with the trace data. The third
        is by providing a list of headers and an array with the data for all 
        traces, similar to when creating a new Gather. 

        Parameters
        ----------
        *args : tuple
            An input tuple containing the input variables on the first index. 
            Input can be:
                Gather
                    Gather containing the traces that are added
                np.ndarray, Header
                    Array containing the data as an array and a (list of) 
                    header(s) that can be used to initialise a Gather
                    
        Raises
        ------
        IndexError
            If the amount of samples in the original gather is not the same as
            in the added data, meaning that the arrays that are added do not 
            have the same shape
        ValueError
            If too many (or no) input values are provided.
            If the time step in the original gather is not the same as in the 
            new data

        Returns
        -------
        Gather
            New data that contains the provided traces added at the end.

        """
            
        # Unpack the data
        args = args[0]
        
        # Now if there is a single input, it should be a Gather
        if len(args) == 1:
            if isinstance(args[0], Gather):
                new_gather = args[0]
            else:
                ValueError(f"Invalid input of type {type(args[0])} provided")
        elif len(args) == 2:
            # If there are two inputs, create a gather with the data
            new_gather = Gather(args[0], args[1])
        else:
            raise ValueError("Wrong amount of variables provided, can be a single gather or a Numpy array with data and a (list of) Header(s)")
        
        # If the original gather is empty, the new gather can be returned
        if self.type == 'empty':
            return new_gather
        else:
            # Else we check if certain values are the same
            if self.ns != new_gather.ns:
                raise IndexError(f"New trace ({new_gather.ns}) is not the same length as traces in gather ({self.ns})")
            elif self.dt != new_gather.dt:
                raise ValueError(f"Time step of new data ({new_gather.dt} ms) is not the same as time step in original gather ({self.dt} ms)")
            
            # Then initialise a new data array for the data and put in the 
            # total data
            new_shape = (self.data.shape[0] + new_gather.data.shape[0],
                         self.data.shape[1])
            new_data = np.zeros(new_shape)
            new_data[:len(self),:] = self.data
            new_data[len(self):,:] = new_gather.data
            
            self.data = new_data
            self.headers.extend(new_gather.headers)
            
            # Update values in gather
            self._update()            
            
            return self
    
    def __eq__(self, other):
        if not isinstance(other, Gather):
            raise ValueError(f"Cannot compare Gather object with other object of type {type(other)}")
            
        # Check the base values of the gather against the other
        if self.ns != other.ns or self.dt != other.dt or self.tol != other.tol:
            return False
        
        # Check if all the headers are the same
        for own_header, other_header in zip(self.headers, other.headers):
            if not own_header == other_header:
                return False
        
        # Finally check if all of the data is the same
        return np.all(self.data == other.data)
    
    def _check_type(self, tol):
        """
        Check the type of gather from the locations of source and receiver.
        A tolerance for the location must be set. This tolerance reflects the
        distance from a point that is still considered the same location. It 
        is smart to not set it to zero, to prevent small numerical differences
        to make a difference. 
        Possible results in order of precedence are:
            single      - gather containing a single trace
            offset      - a common offset gather
            midpoint    - a common midpoint gather
            source      - a common shot gather
            receiver    - a common receiver gather
            mixed       - a gather with no common features

        Parameters
        ----------
        tol : float
            Tolerance of choosing a common point. If the distance from all points
            to their average is within this tolerance it counts as the same. It
            can be used for binning or to prevent small numerical differences 
            from having an effect

        Returns
        -------
        str
            Descriptor of the gather.

        """
        offsets = self.get_item('offset')
        midpoints = (self.src_pos()+self.rec_pos())/2
        
        if self.data.shape[0] == 1:
            typ = 'single'
        elif np.all(check_tol(offsets, tol)):
            typ = 'offset'
        elif np.all(check_tol(midpoints, tol)):
            typ = 'midpoint'
        elif np.all(check_tol(self.unq_src_pos(), tol)):
            typ = 'source'
        elif np.all(check_tol(self.unq_rec_pos(), tol)):
            typ = 'receiver'
        else:
            typ = 'mixed'
            
        self.type = typ
        
    def _set_rec_to_src(self, idx):
        
        new_sx = self.get_item('gx')[idx]
        new_sy = self.get_item('gy')[idx]
        new_selev = self.get_item('gelev')[idx]
        
        self.set_item('sx', new_sx)
        self.set_item('sy', new_sy)
        self.set_item('selev', new_selev)
        
    def _update(self):
        self._check_type(self.tol)
        self.set_item('ntr',len(self))
        self.set_item('ns', self.ns)
        
        return self
    
    ############################## BASIC ####################################
    # Basic operations on the object itself. Generally do not change the data
    # inside. 
    
    def reverse(self):
        new_gather = self[::-1]
        offsets = new_gather.get_item("offset")
        new_gather.set_item('offset', -offsets)
        
        return new_gather
    
    def copy(self):
        return deepcopy(self)
    
    def write(self, filename, endian = 'little'):
        """
        Create a .su file from the gather

        Parameters
        ----------
        filename : str
            Path and filename of the output .su file.
        endian : str, optional
            String giving the endian of the file. Can be '<', '>', 'little' or
            'big'. The default is 'little'.

        Returns
        -------
        None.

        """
        if self.type == 'empty':
            print("Warning: Attempting to write empty Gather, ignoring instead")
            return False
        
        # Write the data to the file
        write_su(filename, self, endian)
        return True
    
    def get_item(self, attr_name):
        """
        Get an array with the value of a certain attribute from each header in 
        the gather

        Parameters
        ----------
        attr_name : str
            Name of the attribute that is gotten from the headers. See the 
            format of the headers for the names

        Returns
        -------
        attr_list : np.ndarray
            Array with the values extracted from the headers.

        """
        attr_list = []
        
        for header in self.headers:
            attr_list.append(getattr(header,attr_name))
            
        return np.array(attr_list)
    
    def set_item(self, attr_name, values):
        """
        Set a value in each header with a provided list

        Parameters
        ----------
        attr_name : str
            Name of the header value that must be changed.
        values : list or np.ndarray
            List containing the new values that replace the old ones.

        Returns
        -------
        None.

        """
        try:
            iter(values)
        except TypeError:
            dtype = self.headers[0].get_dtype(attr_name)
            values = np.zeros(len(self), dtype=dtype) + values
        # if isinstance(values, (int, float, complex)):
        #     values = np.zeros(len(self), dtype = int) + values
        
        for i,header in enumerate(self.headers):
            setattr(header,attr_name,values[i])
        
        return self
    
    def sort_dists(self, end_line=0):
        """
        Create a version of this gather, sorted by the distance to one of the
        outer stations. Can be used to sort over distance along line. 

        Parameters
        ----------
        end_line : int, optional
            Index for self.find_outer_rec(). The default is 0.

        Returns
        -------
        Gather
            Gather with equal values, but traces reordered to align with the 
            distance along the line.

        """
        sort_idx = np.argsort(self.dists(end_line))
        return self[sort_idx]
    
    def sort_offset(self):
        offsets = np.array(self.get_item('offset'))
        sort_idx = np.argsort(offsets)
        return self[sort_idx]
    
    def split_src(self):
        """
        Splits off each source position into a separate gather and returns a 
        list with all unique source position gathers

        Returns
        -------
        list
            List containing Gather objects .

        """
        if self.type == 'source':
            return [self]
        
        # Get the shot positions
        shot_locs = self.src_pos()
        
        # XXX Think of a way to use the tolerance
        
        # Get the unique positions and the index belonging to each shot location
        shot_pos, shot_idx = np.unique(shot_locs, return_inverse=True, axis=0)
        
        # Now go over each shot position and create a gather with only the 
        # traces created from this shot position
        gathers = []
        for i in range(len(shot_pos)):
            mask = shot_idx == i
            
            gathers.append(self[mask])
        
        return gathers
    
    def split_rec(self):
        """
        Splits off each receiver position into a separate gather and returns a 
        list with all unique receiver position gathers

        Returns
        -------
        list
            List containing Gather objects .

        """
        if self.type == 'receiver':
            return [self]
        
        # All receiver positions of the traces
        rec_locs = self.rec_pos()
        
        # Get the unique receiver positions and the index of each trace
        rec_pos, rec_idx = np.unique(rec_locs, return_inverse=True, axis=0)
        
        # Go over the receiver positions and created a gather for each
        gathers = []
        for i in range(len(rec_pos)):
            mask = rec_idx == i
            
            gathers.append(self[mask])
            
        return gathers
    
    def split_line(self):
        """
        Split up the gather into two gathers for negative and positive offset.
        The negative offsets are always given at the first index, the positive
        at the second. If all traces have a positive/negative offset, the other
        Gather is empty.

        Raises
        ------
        ValueError
            If the gather is not a common-source gather.

        Returns
        -------
        list with Gather
            List containing the Gather with negative offsets at the first 
            position and the positive offsets at the second index.

        """
        
        if self.type != 'source':
            raise ValueError('Multiple source positions available in gather')
        
        self.upd_offsets('line')
        
        # Get the offset from the headers
        offsets = np.array(self.get_item("offset"))
        # Get a mask for negative offset
        mask = offsets < 0
        
        # If one Gather contains all traces, handle the result
        if all(mask):
            return [self, Gather()]
        elif not any(mask):
            return [Gather(), self.set_item("offset", abs(offsets))]
        
        # Slice the two gathers
        neg_offsets = self[mask].set_item('offset', abs(offsets)[mask])
        pos_offsets = self[~mask].set_item('offset', abs(offsets)[~mask])
        
        return [neg_offsets, pos_offsets]
    
    ############################ OPERATIONS ##################################
    # Operations that change the data inside the object
    
    def select(self, attr_name, start=None, end=None):
        """
        Create a new gather with only those traces selected where a certain
        attribute in the header falls within a specified range. If no value for
        start or end is defined, it defaults to the minimum or maximum, 
        respectively for the attribute. 

        Parameters
        ----------
        attr_name : str
            Name of the attribute, see surfwav.Header().format for options.
        start : float, optional
            Start of the selection range. The default is None.
        end : float, optional
            End of the selection range. The default is None.

        Returns
        -------
        Gather
            Gather with only the specified traces included.

        """
        # Get an array with the attribute in it
        attr = np.array(self.get_item(attr_name))
        
        # Set the defaults
        if start is None:
            start = min(attr)
        if end is None:
            end = max(attr)
        
        # Make a mask for the gather
        mask = np.logical_and(attr >= start,
                              attr <= end)
        
        if not np.any(mask):
            return Gather()
        
        return self[mask]
    
    def trim(self, end=None, start=0.,):
        """
        Trim this gather to a specified time range. End will default to 
        self.t[-1].

        Parameters
        ----------
        start : float, optional
            Starting time of the new gather. The default is 0.
        end : float, optional
            Ending time of the new gather. The default is None.

        Returns
        -------
        new_gather : Gather
            Gather cut to the specified time range.

        """
        if end is None:
            end = self.t[-1]
        
        # Create a mask for the required time range
        mask = np.logical_and(self.t >= start,
                              self.t <= end,
                              dtype=bool)
                
        # Cut the data
        new_data = self.data[:,mask]
        # Create a new gather
        new_gather = Gather(new_data, deepcopy(self.headers))
        # Update the header
        new_gather = new_gather.set_item('ns', np.sum(mask))
        
        return new_gather
    
    def offset(self, sel_offset):
        """
        Select only traces that fall within a certain range of offset:

        Parameters
        ----------
        sel_offset : float or list
            The offset range selected. If indexable, uses first two indices for
            as left and right offset respectively. Otherwise applies value on
            both sides.

        Returns
        -------
        Gather
            A gather with the specified offset range

        """
        try: 
            sel_offset[0], sel_offset[1]
        except (IndexError, TypeError):
            return self.select("offset", start=-sel_offset, end=sel_offset)
        else:
            return self.select("offset", start=sel_offset[0], end=sel_offset[1])
    
    def filter(self, method, f_range):
        """
        Apply a frequency filter to the data. method and f_range are used together
        to modify the energy at certain frequencies. f_range decides at which
        frequencies the filter is defined and method defines the strength of the
        filter. 
        For example, if f_range is [4,8,24,32] and method is [0,1,1,0], 
        everything below 4 Hz is removed, between 4 and 8 Hz there is a linear
        increase while accepting all energy at 8 Hz, etc. This example would
        be a bandpass filter. This can be extended for more complex filters.
        The first value in method indicates what happens below the first 
        frequency in f_range and the last value in method indicates what happens
        above the last frequency in f_range.
        
        For the standard uses, there are predefined values for method:
            - bandpass
            method will become [0,1,1,0]
            - bandstop
            method will become [1,0,0,1]
            - lowpass
            method will become [1,0]
            - highpass
            method will become [0,1]
        

        Parameters
        ----------
        method : str or list with float
            Either a string for one of the predefined values or a list containing
            floats indicating the pass rate at the frequencies defined in f_range.
        f_range : list
            List with frequency values for which the pass rate is defined in 
            method

        Returns
        -------
        Gather
            Gather containing the filtered data.

        """
        # Determine the Fourier transform
        f, spec = freq_ft(self.data, self.dt, axis=1)
        
        # Set the predefined values or take the one from the input
        if method == 'bandpass':
            stops = [0,1,1,0]
        elif method == 'bandstop':
            stops = [1,0,0,1]
        elif method == 'lowpass':
            stops = [1,0]
        elif method == 'highpass':
            stops = [0,1]
        else:
            stops = method
        
        # If the amount of stops is not equal to the amount of frequencies
        if len(stops) != len(f_range):
            raise ValueError(f'Length of stops ({len(stops)}) and length of frequency values ({len(f_range)}) do not match')
        
        # Initialise a multiplier array
        mult = np.zeros(len(f))
        
        # Set the values at frequencies below the first one to stops[0]
        mask = np.where(f <= f_range[0], True, False)
        mult[mask] = stops[0]
        # Now go over all other stops and set the range between them
        for i in range(len(f_range)):
            if i == len(f_range)-1:
                # For the last one, set all values above the last frequency to
                # stops[-1]
                mask = np.where(f >= f_range[i], True, False)
                mult[mask] = stops[i]
            else:
                mask = np.where(np.logical_and(f >= f_range[i], f <= f_range[i+1]), True, False)
                mult[mask] = np.linspace(stops[i],stops[i+1], np.sum(mask))
        
        # Multiply the data with the filter
        spec *= mult[np.newaxis,:]
        
        # Compute the inverse Fourier transform to get the filtered data
        new_data = np.fft.irfft(spec*self.ns, self.ns, axis=1)
        
        return Gather(new_data, deepcopy(self.headers))
    
    def normalise(self, method='rms-trace'):
        """
        Normalise each trace in a section by dividing each trace by its root-mean-
        square value. Traces with no data are left as is

        Returns
        -------
        new_record : obspy.core.stream.Stream
            Normalised record.

        """
        if method == 'rms-trace':
            data = self.data
            
            # Calculate the rms of the trace. A trace with no data is just divided by 1
            squares = np.square(data)
            mean_squares = np.mean(squares, axis=1)
            mean_squares[mean_squares == 0] = 1.
            root_mean_squares = np.sqrt(mean_squares)
            
            # Divide each trace by its rms
            new_data = data / root_mean_squares[:,np.newaxis]
            
        elif method == 'max-trace':
            data_max = np.max(abs(self.data), axis=1)
            data_max = np.where(data_max == 0, 1, data_max)
            
            new_data = self.data / data_max[:,np.newaxis]
        elif method == 'cyl-spread':
            offsets = np.array(self.get_item('offset'))
            
            rms_gather = self.normalise('rms-trace')
            new_data = rms_gather.data/(offsets[:,np.newaxis]**2)
            
        elif method == 'none':
            return self
        else:
            raise ValueError(f"Invalid method ({method}) provided, can be ['rms-trace', 'max-trace', 'none']")
        
        new_gather = Gather(new_data, deepcopy(self.headers))
        
        return new_gather
    
    def mute(self, method, start, vel=None, length=0.1, top=False):
        """
        Apply muting to the data as a cone spreading from the source location
        with a slope, defined by vel. If no vel is defined, the starting time
        is applied universally, effectively using an infinite velocity. The
        muting can be used with three methods that describe the step from
        accepting no energy and all energy. These are:
            - step
            A step function is used
            - ramp
            A ramp function is used
            - sigmoid
            A sigmoid function specifically designed to be similar to the ramp
            function is used

        Parameters
        ----------
        method : str
            Which method to use to go from muting all energy to passing all
            energy.
        start : float
            The starting time to apply.
        vel : float, optional
            The slope of the muting cone, defined as a velocity. The default is 
            None.
        length : float, optional
            The length of the zone where the muting goes from muting everything
            to passing everything. The default is 0.1.
        top : bool, optional
            Whether to pass all energy at lower traveltimes than the cone (True)
            or the energy at higher traveltimes (False). The default is False.

        Returns
        -------
        Gather
            Gather containing the muted data.

        """
        
        # If no velocity is set, act as if an infinite velocity was used (or just
        # a vertical line)
        if not isinstance(method, str):
            mute_mult(method, 0., 200, 0.1, self)
        
        if vel is None:
            mult = np.where(self.t >= start, 1., 0.)[np.newaxis,:]
        else:
            # Else get the muting multiplier 
            mult = mute_mult(method, start, vel, length, self)
        
        # If you want to use the top, invert the multiplier
        if top:
            mult = 1 - mult
        
        # Multiply the data with the multiplier
        new_data = self.data * mult
        
        # Return the new gather
        return Gather(new_data, deepcopy(self.headers))
    
    def AGC(self,oper_len,basis='centred'):
        """
        Automatic Gain Control balances the gain based on the amplitude in a local
        window. The function is based on the AGC function from SeisSpace ProMAX. 
        Scaling can be done based on the inverse of:
            mean
            median
            RMS 
        The location of the window can be set as:
            trailing:
                Following the sample
            leading:
                Preceding the sample
            centred:
                The sample is located at the centre of the window
        
        See also:
        https://esd.halliburton.com/support/LSM/GGT/ProMAXSuite/ProMAX/5000/5000_8/Help/promax/agc.pdf

        Parameters
        ----------
        record : obspy Stream
            The record that has to be balanced.
        oper_len : float
            Window length in seconds.
        basis : str
            Location of the window compared to each sample. Can be 'trailing', 
            'leading', 'centred'.

        Returns
        -------
        new_record : obspy Stream
            The new record after application of AGC.

        """   
        # The operator length in amount of data points
        oper_len_items = int(np.round(oper_len/self.dt+1))
        
        # The convolution operator for the mean
        operator = np.ones(oper_len_items)
        
        # Calculate how many data points are used for each point
        scal_vals = np.convolve(np.ones(self.ns),operator,'full')
        
        # Convolve the data with the operator and divide by the amount of points
        # used to get the mean
        convol = convolve2d(abs(self.data),operator[np.newaxis,:],'full')/scal_vals[np.newaxis,:]
        
        convol = np.where(convol==0,1,convol)
        
        # Now snip out the relevant part for each method
        if basis == 'trailing':
            snipped = 1/convol[:,oper_len_items-1:]
        elif basis == 'leading':
            snipped = 1/convol[:,:-oper_len_items+1]
        elif basis == 'centred':
            snipped = 1/convol[:,int(oper_len_items/2):int(-oper_len_items/2)]
        
        # Multiply the data with the scaling values
        new_data = self.data*snipped
            
        return Gather(new_data, deepcopy(self.headers))
    
    def remove_und_coords(self):
        return self[np.all(self.rec_pos() != 0, axis=-1)]
    
    def stack_src_rec_pairs(self, sort=True, end_line=0):
        """
        Stack source-receiver pairs, traces that have the same source and 
        receiver combination. At the end, the resulting gather can be sorted 
        over the distance to the source, which is recommended, because the 
        identification of overlapping source-receiver pairs shuffles around 
        some of the traces. 
        
        N.B. If there are three traces per source-receiver. Consider if these
        are not simply the three components!

        Parameters
        ----------
        sort : bool, optional
            Whether or not to sort the gather at the end of the stacking, 
            recommended.
        end_line : int, optional
            Index for self.find_outer_rec(). The default is 0.

        Returns
        -------
        Gather
            Gather containing the stacked data.

        """
        # Combine the source and receiver positions into one array
        comb_src_rec_pos = np.stack([self.src_pos(), self.rec_pos()], axis=-1)
        # Get the unique source and receiver positions (src_rec_pairs) and their
        # position in the original array
        src_rec_pairs, pair_idx = np.unique(comb_src_rec_pos, return_inverse=True, axis=0)

        # Initialise a new data array with the right amount of pairs
        new_data = np.empty([src_rec_pairs.shape[0], self.ns])

        # Initialise header list
        new_headers = []
        # Go over the index to each source-receiver pair
        for i,idx in enumerate(np.unique(pair_idx)):
            # Get a mask for all traces that fit with this pair
            mask = pair_idx == idx
            # Get the first trace that belongs to this stack
            trace_idx = np.argwhere(mask)[0][0]
            # Sum all traces for this pair and divide by the amount summed
            new_data[i,:] = self.data[mask,:].sum(axis=0) / np.sum(mask)
            # Append the header of the first trace
            new_headers.append(self.headers[trace_idx])
        # Make a new gather for the stacked data
        new_gather = Gather(new_data, new_headers)
        
        # If asked, sort the data, because np.unique shuffles a bit
        if sort:
            new_gather = new_gather.sort_dists(end_line)
        
        return new_gather
    
    def upd_offsets(self, method='simple'):
        """
        Update the offset value in the headers. Can be important for gathers
        where no offset is previously defined or when values are changed. Two
        methods are available:
            - simple
            Takes the absolute distance between the source and receiver pairs
            - line
            Assumes the sources and receivers lie on the same line. Gives 
            positive and negative offset along the line. Negative offset
            is defined as the direction of the first trace to the source

        Parameters
        ----------
        method : str, optional
            Which method to use to determine the offsets. The default is 
            'simple'.

        Raises
        ------
        ValueError
            If no valid value of method is provided.

        Returns
        -------
        None.

        """
        
        if method == 'simple':
            # Get the distance between all receivers and their source 
            new_offset = np.linalg.norm(self.rec_pos() - self.src_pos(), axis=1).squeeze()
            new_offset.astype(np.int32)
        elif method == 'line':
            try:
                # Fit a line through the receiver positions
                coef, intercept = self.fit_line('rec')
            # If there is a ValueError, a vertical line is found, then
            except ValueError:
                # The projection is simple
                recs_line = self.rec_pos()[:,:-1].T
                srcs_line = self.src_pos()[:,:-1].T
                srcs_line[0,:] = recs_line[0,0]
            else:
                # Project all source and receiver positions on the line
                recs_line = point_to_line(coef, intercept, self.rec_pos()[:,:-1])
                srcs_line = point_to_line(coef, intercept, self.src_pos()[:,:-1])
            
            pos_to_source = recs_line - srcs_line
            angles = np.arctan2(pos_to_source[1,:], pos_to_source[0,:])
            angle = angles[0]
            if angle != 0.:
                opposite = angle - angle/abs(angle) * np.pi
            else:
                opposite = np.pi
            neg_angle = min(angle, opposite)
            
            # Get the distance to the first receiver in the gather
            # dists = np.linalg.norm(recs_line - recs_line[:,0][:,np.newaxis], axis=0)
            # dists_src = np.linalg.norm(srcs_line - recs_line[:,0][:,np.newaxis], axis=0)
            
            # Now calculate the absolute offset
            new_offset = np.linalg.norm(recs_line - srcs_line, axis=0)

            # For stations that are less far along the line than the source,
            # set a negative offset
            new_offset = np.where(angles - neg_angle <= 1e-4, -new_offset, new_offset)
            # new_offset = np.where(dists < dists_src, -new_offset, new_offset)
        else:
            raise ValueError(f"Invalid method ({method}) provided, can be ['simple', 'line']")
        
        self.set_item('offset', new_offset)

        return self
        
    def add_noise(self, ampl, std, centre=0.):
        """
        Add Gaussian noise to the data. Gaussian is defined by the mean (centre)
        and the standard deviation (std). All noise values are multiplied with
        ampl. 

        Parameters
        ----------
        ampl : float
            All random noise samples are multiplied with ampl to change the
            amplitude of the noise.
        std : float
            Standard deviation of the Gaussian used to create the noise.
        centre : float, optional
            Mean of the Gaussian used to create the noise.

        Returns
        -------
        Gather
            New Gather with noise added to the data.

        """
        new_data = self.data + np.random.normal(centre,std,self.shape)*ampl
        
        return Gather(new_data, deepcopy(self.headers))
    
    ############################## COORDS ####################################
    # All methods that use the source and receiver positions
    
    def dx(self):
        """
        Determine the receiver spacing assuming an equal spacing between all
        receivers. 

        Returns
        -------
        float
            Receiver spacing.

        """
        return max(self.dists() - min(self.dists()))/len(self)
    
    def src_pos(self):
        """
        Gives an array containing the source coordinates of each trace in the 
        gather

        Returns
        -------
        np.ndarray
            [amt. traces, 3] Array containing the source coordinates.

        """
        # Get the values sx and sy from all headers
        sx = self.get_item("sx")
        sy = self.get_item("sy")
        selev = self.get_item("selev")
        # Get the coordinate scaling and elevation scaling
        scalco = self.get_item("scalco")
        scalel = self.get_item("scalel")
        
        # Apply the scaling to the values
        sx = su_div(sx, scalco)
        sy = su_div(sy, scalco)
        selev = su_div(selev, scalel)
        
        # And return one array with the source coordinates
        return np.array([sx,sy, selev]).T
    
    def unq_src_pos(self):
        """
        Gives the unique source positions in the gather

        Returns
        -------
        np.ndarray
            Array containing each unique source position in the array.

        """
        return np.unique(self.src_pos(), axis=0)
    
    def rec_pos(self):
        """
        Gives an array containing the receiver coordinates of each trace in the 
        gather

        Returns
        -------
        np.ndarray
            [amt. traces, 3] Array containing the receiver coordinates.

        """
        # Get the values gx and gy from all headers
        gx = self.get_item("gx")
        gy = self.get_item("gy")
        gelev = self.get_item("gelev")
        # and the coordinate scaling
        scalco = self.get_item("scalco")
        scalel = self.get_item("scalel")
        
        # Apply the scaling to the values
        gx = su_div(gx, scalco)
        gy = su_div(gy, scalco)
        gelev = su_div(gelev, scalel)
        
        return np.array([gx,gy,gelev]).T
    
    def unq_rec_pos(self):
        """
        Gives the unique receiver positions in the gather

        Returns
        -------
        np.ndarray
            Array containing each unique receiver position in the array.

        """
        return np.unique(self.rec_pos(), axis=0)
    
    def estimate_src(self):
        
        maxima = np.max(abs(self.data), axis=1)
        
        idx_max = np.argmax(maxima)
        if idx_max == 0 or idx_max == len(self) - 1:
            return self
        idx_side = idx_max - 1 if maxima[idx_max-1] >= maxima[idx_max+1] else idx_max+1
        dist_src = np.average(self.rec_pos()[[idx_max,idx_side],:], axis=0) 
        dist_src = dist_src.squeeze()[np.newaxis,:] + np.zeros([len(self),1])
        
        new_gx = su_mult(dist_src[:,0], self.get_item('scalco'))
        new_gy = su_mult(dist_src[:,1], self.get_item('scalco'))
        new_gz = su_mult(dist_src[:,2], self.get_item('scalel'))
        
        self.set_item('sx', new_gx)
        self.set_item('sy', new_gy)
        self.set_item('selev', new_gz)
        
        return self
    
    def fit_line(self, fit_to):
        """
        Fit a 2D line through a specified set of coordinates. Line has form
        y = a*x + b, where a is 'coef' and b is 'intercept'. fit_to can be:
            - rec
            For the receiver coordinates
            - src
            For the source coordinates

        Parameters
        ----------
        fit_to : str
            Shows what set of coordinates to fit the line through.

        Raises
        ------
        NotImplementedError
            If an invalid fit_to is given.

        Returns
        -------
        coef : float
            Coefficient (a) describing the line.
        intercept : float
            Intercept (b) describing the line

        """
        
        if fit_to == 'rec':
            points = self.rec_pos()
        elif fit_to == 'src':
            points = self.src_pos()
        else:
            raise NotImplementedError(f"fit_to ({fit_to}) is not a valid value, can be ['rec', 'src']")
                                    
        coef, intercept = fit_line(points[:,0], points[:,1])
            
        return coef, intercept
    
    def dist_mat(self):
        """
        Get a distance matrix between the receiver positions. The matrix 
        contains the distance between receiver i and receiver j at location ij.
        This means that the matrix is symmetric with zeroes along the diagonal.
        If the locations are given in global coordinates, this will not give
        meaningful results

        Returns
        -------
        np.ndarray
            Distance matrix for the receivers in the gather.

        """
        return np.linalg.norm(self.rec_pos()[np.newaxis,:,:] - self.rec_pos()[:,np.newaxis,:], axis=2)
    
    def find_outer_rec(self):
        """
        Find the two receiver stations that are the furthest away from each 
        other. When the receivers are located on a line, this would find the 
        outer ends of this line. 
        The result has the station closest to the source of the coordinate 
        system at the first index

        Returns
        -------
        list
            List with the indices of the outer receiver stations.

        """
        # Find the indices of the two receiver outer stations
        idcs = np.argwhere(self.dist_mat() == self.dist_mat().max())[0]
        # Now take the station closest to the source of the coordinate system
        # as the first index
        closest = np.argwhere(np.linalg.norm(self.rec_pos()[idcs,:],axis=1) == np.linalg.norm(self.rec_pos()[idcs,:],axis=1).min()).squeeze()
        return [idcs[closest], idcs[closest-1]]
    
    def dists(self, end_line=0):
        """
        Get an array with the distance of each station to one of the outer 
        stations (see self.find_outer_rec()). Works to get the distance along
        the line when all receivers are placed on a line

        Parameters
        ----------
        end_line : int, optional
            Index for self.find_outer_rec(). The default is 0.

        Returns
        -------
        np.ndarray
            Array with the distance to each stations.

        """
        return self.dist_mat()[self.find_outer_rec()[end_line]]
    
    def dist_src(self, end_line=0):
        """
        Finds the distance from one of the outer stations to the source 
        position, similar to self.dists(). When the sources and receivers are
        on the same line, this finds the along line distance of the source.

        Parameters
        ----------
        end_line : int, optional
            Index for self.find_outer_rec(). The default is 0.

        Raises
        ------
        ValueError
            If the gather is not a common-source gather.

        Returns
        -------
        float
            Distance between the specified outer station and the source.

        """
        # XXX Can probably be removed
        # if self.type != 'source':
        #     raise ValueError(f"Function not made for gather of type '{self.type}'")
        
        return np.linalg.norm(self.rec_pos()[self.find_outer_rec()[end_line],:] - self.unq_src_pos())
    
    def sweep(self, step, start=0, stop=None,  length=None):
        
        offsets = self.get_item('offset')
        if stop is None:
            stop = abs(offsets).max()
        # start_locs = np.arange(start, stop, step)
        start_loc = start
                
        while start_loc <= stop:
            if length is None:
                length = offsets.max()*2
            mask = np.logical_and(abs(offsets) >= start_loc,
                                  abs(offsets) <= start_loc+length)
            
            start_loc += step
            
            yield self[mask]
            
    ############################ MULTIPLICATIONS #############################
    
    def correlate(self, signal2, return_causal=True):
        
        new_data = rfftcorrelate(self.data, signal2)
        
        if return_causal is None:
            pass
        elif return_causal:
            new_data = new_data[:,self.ns-1:]
        elif not return_causal:
            new_data = new_data[:,:self.ns-1]
        
        return Gather(new_data, deepcopy(self.headers))
    
    def coherence(self, signal2, return_causal=True):
        
        new_data = rfftcoherence(self.data, signal2)
        
        if return_causal is None:
            pass
        elif return_causal:
            new_data = new_data[:,self.ns-1:]
        elif not return_causal:
            new_data = new_data[:,:self.ns-1]
        
        return Gather(new_data, deepcopy(self.headers))
    
    def convolve(self, signal2, return_causal=True):
        
        new_data = rfftconvolve(self.data, signal2)
        
        if return_causal is None:
            pass
        elif return_causal:
            new_data = new_data[:,self.ns-1:]
        elif not return_causal:
            new_data = new_data[:,:self.ns-1]
        
        return Gather(new_data, deepcopy(self.headers))
    
    def corr_in(self, idx, return_causal=True):
        
        new_gather = self.correlate(self.data[idx,:][np.newaxis,:], return_causal=return_causal)
        
        new_gather._set_rec_to_src(idx)
        
        return new_gather
    
    def coh_in(self, idx, return_causal=True):
        
        new_gather = self.coherence(self.data[idx,:][np.newaxis,:], return_causal=return_causal)
        
        new_gather._set_rec_to_src(idx)
        
        return new_gather
    
    def conv_in(self, idx, return_causal=True):
        
        new_gather = self.convolve(self.data[idx,:][np.newaxis,:], return_causal=return_causal)
        
        new_gather._set_rec_to_src(idx)
        
        return new_gather
    
    ############################ TRANSFORMS ##################################
    
    
    def ft(self):
        """
        Compute the Fourier transform of the data

        Returns
        -------
        np.ndarray
            Array containing the frequencies of the spectrum.
        np.ndarray
            Frequency spectrum of the data per trace

        """
        return freq_ft(self.data, self.dt, axis=1)
    
    def fk(self):
        """
        Transform the data to the frequency-wavenumber (f-k) domain. 

        Returns
        -------
        np.ndarray
            Array describing the frequency axis.
        np.ndarray
            Array describing the wavenumber axis
        np.ndarray
            Array containing the f-k data

        """
        return fk_transform(self.data, self.dt, self.dx())
    
    def FDBF(self, vel, **kwargs):
        """
        Transform the data to the frequency-phase velocity domain using 
        frequency domain beamforming. See surfwav.FDBF.frequency_domain_beamforming
        for more information.

        Parameters
        ----------
        vel : np.ndarray
            At which phase velocities the data is evaluated. See it as the 
            resolution along the phase-velocity axis
        **kwargs : dict
            Keyword arguments for the frequency_domain_beamforming function.

        Returns
        -------
        f : np.ndarray
            Array describing the frequency axis.
        fdbf : np.ndarray
            Array containing the transformed data

        """
        return frequency_domain_beamforming(self, 
                                         vel, 
                                         **kwargs)
    
    ############################## PLOTTING ###################################
    
    def plot(self, method, **kwargs):
        """
        Plot the data in the space-time domain. Can be done with two methods:
            - wiggle
            Plot wiggle traces for each trace in the section, see 
            surfwav.plot.plot_section_wiggle
            - map
            Plot the data with a colormap, see
            surfwav.plot.plot_section_map

        Parameters
        ----------
        method : str
            Which method to use to plot the data.
        **kwargs : dict
            Keyword arguments for the plotting function.

        Raises
        ------
        ValueError
            If no valid vlaue for method is provided.

        Returns
        -------
        plt.Figure
            Figure used to plot the data.
        plt.Axis
            Axis used to plot the data

        """
        if method == "wiggle":
            return plot_section_wiggle(self, **kwargs)
        elif method == "map":
            return plot_section_map(self, **kwargs)
        else:
            raise ValueError(f"Invalid method ({method}) provided, can be ['wiggle', 'map']")
            
    def plot_mute(self, start, vel, length=0.1):
        """
        Plot the muting lines over a wiggle plot of the gather. 

        Parameters
        ----------
        start : float
            The start time of the mute function at zero offset
        vel : float
            The slope of the muting cone.
        length : float, optional
            The ramp length of the muting function. The default is 0.1.

        Returns
        -------
        None.

        """
        lw_main = 1.2
        lw_side = 1.
        
        # Initialise a figure
        fig, ax = plt.subplots(dpi=300)
        # Plot the wiggle plot
        self.plot('wiggle', figure=(fig,ax))
        
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        
        # Plot the main centre mute line
        ax.plot([xlims[0], 0], [abs(xlims[0])/vel, start], c='r', lw=lw_main, alpha=0.6)
        ax.plot([0,xlims[1]], [start, abs(xlims[1]/vel)], c='r', lw=lw_main, alpha=0.6)
        
        # Plot the side lines
        for i in [-1, 1]:
            ax.plot([xlims[0], 0], 
                    [abs(xlims[0])/vel+0.5*i*length, start+0.5*i*length], 
                    c='orange', 
                    lw=lw_side, 
                    alpha=0.6)
            ax.plot([0, xlims[1]], 
                    [start+0.5*i*length, abs(xlims[1])/vel+0.5*i*length], 
                    c='orange', 
                    lw=lw_side, 
                    alpha=0.6)
        
        # Enforce the limits
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
    def plot_src_gathers(self):
        """
        Plot all of the receivers and source positions, colouring each for
        the source location. Only really helpful if there is not a lot of 
        overlap or to see the rough geometry

        Returns
        -------
        None.

        """
        
        # Get the source gathers in the original in separate gathers
        src_gathers = self.split_src()
        
        # Set up the plot
        fig, ax = plt.subplots(dpi=300)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Get the default color cycle for matplotlib
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        col_idx = 0
        for i,src_gather in enumerate(src_gathers):
            # Plot the source position for this source gather
            ax.scatter(src_gather.src_pos()[0,0], 
                       src_gather.src_pos()[0,1], 
                       s = 70,
                       marker="*",
                       zorder = 5,
                       color = colors[col_idx],
                       label=f"Source {i+1}")
            # and all receiver positions
            ax.scatter(src_gather.rec_pos()[:,0], 
                       src_gather.rec_pos()[:,1], 
                       s=2,
                       marker="v",
                       color = colors[col_idx])
            ax.legend()
            
            # Wrap the color cycle round when it reaches the end of the list
            col_idx += 1
            if col_idx >= len(colors):
                col_idx = 0
        
        # Change the plot to have the same scale on the x and y axes
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        
        spread = max(ylims[1]-ylims[0], xlims[1]-xlims[0])
        centre = [np.mean(xlims), np.mean(ylims)]
        
        xlims = [centre[0]-0.5*spread, centre[0]+0.5*spread]
        ylims = [centre[1]-0.5*spread, centre[1]+0.5*spread]
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        return fig, ax
    
    def plot_ft(self, figure=None):
        """
        Plot the Fourier spectrum of the data. A predefined figure can be given
        to plot the data on

        Parameters
        ----------
        figure : tuple, optional
            The matplotlib Figure and Axis to plot the data on. The default is 
            None.

        Returns
        -------
        matplotlib.pyplot.Figure
            Figure the data is plotted on.
        matplotlib.pyplot.Axes
            Axis the data is plotted on

        """
        
        f, spec = self.ft()
        return plot_ft(f, self.dists(), spec, figure=figure)
    
    def plot_avg_fourier(self, figure=None, fmax=None):
        """
        Plot the average frequency spectrum of the data. The individual traces
        are plotted with transparency in black, while the average is plotted in
        red.

        Parameters
        ----------
        figure : tuple, optional
            Tuple containing the figure and axis to plot the data on. The 
            default is None.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.

        """
        f, spec = self.normalise('rms-trace').ft()
        
        if figure is None:
            fig, ax = plt.subplots(dpi=300)
        else:
            fig, ax = figure
        
        if fmax is None:
            fmax = f[-1]
        
        for line in spec:
            ax.plot(f, abs(line), c='black', alpha=0.3)
        ax.plot(f, np.average(abs(spec), axis=0), c='r')
        ax.set_xlim([f[0],fmax])
        ylim = ax.get_ylim()
        ax.set_ylim([0,ylim[1]])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Normalised amplitude")
        
        return fig, ax
    
    def plot_fk(self, figure=None):
        f, k, fk = self.fk()
        return plot_fk(f, k, fk, figure=figure)
    
    def plot_FDBF(self, vel, figure=None, plot_max=False, **kwargs):
        f, fdbf = self.FDBF(vel, **kwargs)
        return disp_power(f, vel, fdbf, figure=figure, plot_max=plot_max)
        
def check_tol(array, tol, point = None):
    """
    Check if all given points fall within the tolerance distance of the given
    point. If no point is specified, the average location of the array is used.

    Parameters
    ----------
    array : np.ndarray
        Array containing all positions that are checked. For two-dimensional 
        arrays the norm of each coordinate set is taken for the distance. It is
        assumed that the last axis is used for this, so that the array has 
        shape [amt. points, 2]
    tol : float
        Tolerance for distance from the specified point.
    point : np.ndarray, optional
        Array containing the point from which the distance is checked. If no
        point is specified, the average of all points in array is used

    Returns
    -------
    np.ndarray
        Boolean array indicating which positions are close enough to the point.

    """
    if point == None:
        point = np.average(array, axis=0)
    if array.ndim >= 2:
        array = np.linalg.norm(array, axis=-1)
        point = np.linalg.norm(point, axis=-1)
    
    return abs(array - point) <= abs(tol)
    
def su_div(value, factor):
    
    # For arrays/lists 
    if isinstance(value, np.ndarray) or isinstance(value, list):
        # Enforce arrays
        value = np.array(value)
        factor = np.array(factor)
        
        # Check if the arrays have the same length
        if len(value) != len(factor):
            raise IndexError("Length of value array ({len(value)}) is not the same as factor array ({len(factor)})")
        
        # Get the index mask
        mask = factor < 0
        
        # Where the mask is True, divide values by the factor
        value[mask] = value[mask] / np.abs(factor[mask])
        # Elsewhere multiply with the factor
        value[~mask] = value[~mask] * factor[~mask]
        
        # Return the array
        return value
        
    else:
        if factor < 0:
            return value/abs(factor)
        else:
            return value*factor
        
def su_mult(value, factor):
    
    # For arrays/lists 
    if isinstance(value, np.ndarray) or isinstance(value, list):
        # Enforce arrays
        value = np.array(value)
        factor = np.array(factor)
        
        # Check if the arrays have the same length
        if len(value) != len(factor):
            raise IndexError("Length of value array ({len(value)}) is not the same as factor array ({len(factor)})")
        
        # Get the index mask
        mask = factor < 0
        
        # Where the mask is True, multiply values by the factor
        value[mask] = value[mask] * np.abs(factor[mask])
        # Elsewhere divide with the factor
        value[~mask] = value[~mask] / factor[~mask]
        
        # Return the array
        return value
        
    else:
        if factor < 0:
            return value*abs(factor)
        else:
            return value/factor
    
        
def mute_mult(method, start, vel, length, gather):
    """
    Create a muting multiplier for a given gather. The result is an array with
    the same shape as the gather. The muting is applied as a cone centred on 
    the source location where the angle is described by vel. How the change 
    from 0 to 1 is handled by setting method:
        - step
        A simple step function is used where the functionchanges from 0 to 1 in 
        a single time step
        - ramp
        A ramp function with defined length is used
        - sigmoid
        A sigmoid function designed to be similar to the ramp function is used

    Parameters
    ----------
    method : str
        Which method to use to handle the change from 0 to 1 in the muting 
        function.
    start : float
        The starting time of the source.
    vel : float
        The velocity used for muting. This is used as the slope in the x-t 
        domain. Unit is [offset unit]/[time unit].
    length : float
        Rough range over which the muting function goes from 0 to 1. This is
        ignored if the method is 'step'.
    gather : Gather
        Gather for which the muting multiplier is designed.

    Returns
    -------
    np.ndarray
        Array with the same size as gather. An element-wise multiplication 
        between gather.data and this provides the muted data.

    """
    # Get the axis values for the gather 
    t = gather.t[np.newaxis,:]
    offset = np.array(gather.get_item('offset'))[:,np.newaxis]
    
    # A length of 0 is the same as the step function
    if length == 0.:
        method = 'step'
    
    # Set the centre location for the muting in time
    locations = abs(offset)/vel + start
    # Now get the index belonging to this location
    loc_idx = np.round(locations/gather.dt).astype(int)
        
    # Initialise the multiplication array
    mult = np.where(locations <= t, 1., 0.)
    if method == 'step':
        return mult
    elif method == 'ramp':
        # Get the ramp
        ramp = np.linspace(0,1,int(length/gather.dt))
        
        # Apply the ramp function on each line
        for i,line in enumerate(mult):
            # The centre of the ramp
            idx_ramp = loc_idx[i]
            # Get the indices for the main matrix, also taking the edges into
            # account
            idx_start = max(0,
                            min(int(idx_ramp-0.5*len(ramp)),len(line)))
            idx_end = min(len(line),
                          max(int(idx_ramp+0.5*len(ramp)),0))
            # Slice the ramp to fit with the main matrix
            idx_start_ramp = max(0,int(-(idx_ramp-0.5*len(ramp))))
            idx_end_ramp = min(len(ramp),
                                max(int(len(line) - idx_ramp+0.5*len(ramp)),0))
            # Adapt the values
            line[idx_start:idx_end] = ramp[idx_start_ramp:idx_end_ramp]
                
        return mult
    elif method == 'sigmoid':
        # Calculate a sigmoid function for each entry in locations, designed
        # to be similar to the ramp function. If you want to make it steeper 
        # change the value before length to be lower and vice versa.
        return 1/(1+np.exp(-(t - locations)/(0.23*length)))
    else:
        raise ValueError(f"Invalid method ({method}) provided, can be ['step', 'ramp', 'sigmoid']")