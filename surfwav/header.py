# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:24:16 2023

@author: Sverre Hassing
"""

from struct import calcsize
import numpy as np
from surfwav.util import conv_endian

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
    
def convert_type(data, to_type):
    """
    Convert data to the required type for the header.

    Parameters
    ----------
    data : int or float or np.ndarray
        The data that is converted to another type.
    to_type : str
        The type to which the data is converted, must be one of the types 
        described in check_binary_fit.

    Raises
    ------
    ValueError
        If to_type is not one of the valid types.

    Returns
    -------
    int or float or np.ndarray
        The data converted to the right type.

    """
    # Convert to an integer
    if to_type in ['i', 'h', 'q']:
        return int(data)
    # Convert to a float
    elif to_type in ['f']:
        return float(data)
    # Or convert an array to a float or int
    elif to_type[-5:] == 'array':
        if to_type[0] in ['i', 'h', 'q']:
            return data.astype(int)
        elif to_type[0] in ['f']:
            return data.astype(float)
    else:
        raise ValueError(f"No valid value for to_type is given ({to_type})")

# Data types to python identifiers and the byte length
# int        -> 4 bytes  -> i
# short      -> 2 bytes  -> h
# float      -> 4 bytes  -> f
# unassigned -> 8 bytes  -> q (long long)

def SU_header_fmt():
    """
    Gives the format of a Seismic Unix .su file in a list. Each entry in the
    list gives the data type in the first index (given as a letter usable by
    'pack' from struct) and the variable name in the second index. 
    
    IMPORTANT: not all data types given in the first index are usable by pack.
    It is possible to use an array. An array of length 14 with floats in it
    would give the string 'h14array'. This can always be checked by indexing the
    entry of header_fmt as entry[0][-5:]. See that the example string would 
    return 'array', then entry[0][0] gives the data type 'h' and entry[0][1:-5]
    the length of the array (14).

    Returns
    -------
    header : list
        List containing a list for every header entry in the SU .su format. 
        In this smaller list, the first entry gives the data format of the entry
        and the second the name of the variable. 

    """
    # To make the list a bit easier to read, all header entries include the 
    # format in the binary file as a name, shown in 'fmt'. Afterwards these
    # are replaced with the identifier used in the pack function contained in 
    # struct, shown in 'fmt_id'. There is one exception for the unassigned bits
    # where 'h14array' indicates an array of 14 values encoded as a float ('h')
    fmt_id = ['i','h','f','q']
    fmt = ['int','short','float']
    
    header_format = [
        ['int','tracl'],
        ['int','tracr'],
        ['int','fldr'],
        ['int','tracf'],
        ['int','ep'],
        ['int','cdp'],
        ['int','cdpt'],
        ['short','trid'],
        ['short','nva'],
        ['short','nhs'],
        ['short','duse'],
        ['int','offset'],
        ['int','gelev'],
        ['int','selev'],
        ['int','sdepth'],
        ['int','gdel'],
        ['int','sdel'],
        ['int','swdep'],
        ['int','gwdep'],
        ['short','scalel'],
        ['short','scalco'],
        ['int','sx'],
        ['int','sy'],
        ['int','gx'],
        ['int','gy'],
        ['short','counit'],
        ['short','wevel'],
        ['short','swevel'],
        ['short','sut'],
        ['short','gut'],
        ['short','sstat'],
        ['short','gstat'],
        ['short','tstat'],
        ['short','laga'],
        ['short','lagb'],
        ['short','delrt'],
        ['short','muts'],
        ['short','mute'],
        ['short','ns'],
        ['short','dt'],
        ['short','gain'],
        ['short','igc'],
        ['short','igi'],
        ['short','corr'],
        ['short','sfs'],
        ['short','sfe'],
        ['short','slen'],
        ['short','styp'],
        ['short','stas'],
        ['short','stae'],
        ['short','tatyp'],
        ['short','afilf'],
        ['short','afils'],
        ['short','nofilf'],
        ['short','nofils'],
        ['short','lcf'],
        ['short','hcf'],
        ['short','lcs'],
        ['short','hcs'],
        ['short','year'],
        ['short','day'],
        ['short','hour'],
        ['short','minute'],
        ['short','sec'],
        ['short','timbas'],
        ['short','trwf'],
        ['short','grnors'],
        ['short','grnofr'],
        ['short','grnlof'],
        ['short','gaps'],
        ['short','otrav'],
        ['float','d1'],
        ['float','f1'],
        ['float','d2'],
        ['float','f2'],
        ['float','ungpow'],
        ['float','unscale'],
        ['int','ntr'],
        ['short','mark'],
        ['short','shortpad'],
        ['h14array','unass']
        ]
    
    # Replace the names with the correct identifiers for pack
    for i in header_format[:-1]:
        i[0] = fmt_id[fmt.index(i[0])]
        
    return header_format

def COH_header_fmt():
    """
    Gives the format for the header of the coherence files. 

    Returns
    -------
    header : list
        List containing a list for every header entry in the SU .su format. 
        In this smaller list, the first entry gives the data format of the entry
        and the second the name of the variable. 

    """
    
    fmt_id = ['i','h','f','q']
    fmt = ['int','short','float']
    
    header_format = [
        ['int','ntr'],
        ['int','nf'],
        ['float','df'],
        ['int','idx0'],
        ['int','idx1']
        ]
    
    # Replace the names with the correct identifiers for pack
    for i in header_format:
        i[0] = fmt_id[fmt.index(i[0])]
        
    return header_format


class Header:
    
    def __init__(self, fmt = "SU"):
        
        # Get the format for the header from the definition function
        if fmt == "SU":
            self.format = SU_header_fmt()
        elif fmt == "COH":
            self.format = COH_header_fmt()
        else: # If no valid formats are provided throw an error
            raise ValueError(f"No valid format is provided, can be 'SU', but is '{fmt}'")
        
        # Go over each entry in the format and fill in zeroes
        for data_type, name in self.format:
            # See if data_type contains a valid format for struct
            try:
                calcsize(data_type[0])
            except Exception:
                raise ValueError(f"Invalid binary format '{data_type[0]}'")
            
            # If the last part is array, set an array
            if data_type[-5:] == "array":
                setattr(self, name, np.zeros(int(data_type[1:-5]), dtype=np.int8))
            else:
                # If it is a single value, initialise with a zero
                setattr(self, name, 0)
    
    def __repr__(self):
        return "Header()"
    
    def __str__(self):
        
        # Build up a string with every entry from the format on a line
        string = "Header with values:\n"
        
        for __, name in self.format:
            value = getattr(self, name)
            
            # Determine how many tabs are needed to align the results
            tabs = int(4 - np.floor(len(name) / 4))
            
            string += f"\t{name}"+tabs*"\t"+f":\t{value}\n"
        
        return string
    
    def __eq__(self, other_header):
        # Check if the other object is a Header
        if not isinstance(other_header, Header):
            raise ValueError(f"Cannot compare Header object with other object of type {type(other_header)}")
        
        # Go over each entry in the format
        for __, name in self.format:
            
            # And see if they are the same in both headers
            if np.all(getattr(self, name) != getattr(other_header, name)):
                return False
        
        # If False is never triggered, return True
        return True
    
    def get_tuple(self):
        """
        Builds up a tuple with every entry from the format in the right order.
        Arrays are extracted so that every array entry is a separate entry. 
        Meant for packing the data in the header into bytes

        Returns
        -------
        tuple
            Tuple with the values according to the format.

        """
        
        # Build up a list with every attribute from the format in the right order
        lst = []
        
        for data_type, name in self.format:
            value = getattr(self,name)
            if data_type in ['i','h','f','q']:
                lst.append(value)
            elif data_type[-5:] == "array":
                # Unpack arrays into separate entries for the packing
                for i in value:
                    lst.append(i)
        
        # Finally convert to a tuple
        return tuple(lst)
    
    def binary_fmt(self, endian = "little"):
        """
        Gets the binary format for the pack function in struct from the format.

        Parameters
        ----------
        endian : str, optional
            DESCRIPTION. The default is "little".

        Returns
        -------
        fmt : str
            Format to pack the data in the header.

        """
        endian = conv_endian(endian)
        
        # Build up the binary format by beginning with the endian
        fmt = endian
        for data_type, __ in self.format:
            if data_type[-5:] == "array":
                # For an array add the specified value at the first position
                # multiplied by the amount specified in the following indices
                fmt += data_type[0]*int(data_type[1:-5])
            else:
                # For a single value simply add the provided value
                fmt += data_type[0]
        
        return fmt
    
    def fit_binary(self):
        """
        To prepare the data in the header for packing into binary, prevent
        values from being too large/small. If the value is too large for its
        format, saturate it by setting it to the maximum allowed value

        Returns
        -------
        self
            Changes the values inside itself.

        """
        # Make sure that each value in the header fits with the binary format
        # that is provided for it. If it is too large or small, give the
        # saturated value
        for data_type, name in self.format:
            value = getattr(self, name)
            
            if data_type in ['i','h','f','q']:
                new_val = check_binary_fit(value,data_type)
                
                setattr(self, name, new_val)
            elif data_type[-5:] == "array":
                for i in range(len(value)):
                    value[i] = check_binary_fit(value[i],data_type[0])
                
                setattr(self, name, value)
        
        return self
    
    def attr_list(self):
        """
        Return a list with the name of every attribute specified in the format

        Returns
        -------
        attributes : list
            A list containing the string names of each attribute in the format.

        """
        # Return a list with all the attribute names from the format
        attributes = []
        for __, name in self.format:
            attributes.append(name)
        return attributes
    
    def get_type(self, attr_name):
        """
        Retrieve the data type that is specified for a certain attribute in the
        format. Fits with a data type in struct, so for example h for a short
        integer.

        Parameters
        ----------
        attr_name : str
            Name of the attribute for which the data type is checked.

        Returns
        -------
        str
            Single character string that could be used by struct to indicate 
            how the attribute should be packed.

        """
        # Get the byte format for a specified attribute
        idx = self.attr_list().index(attr_name)
        
        return self.format[idx][0][0]
    
    def get_dtype(self, attr_name):
        """
        Get the dtype that belongs to a certain attribute

        Parameters
        ----------
        attr_name : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Get the dtype fitting with a certain attribute in the format
        bytetype = self.get_type(attr_name)
        
        return np.dtype(bytetype)
        
        # XXX No way to check array size
        # btypes = ['i','h','f']
        # dtypes = [np.int32, np.int16, np.float32]
        
        # return dtypes[btypes.index(bytetype)]
    
    def force_types(self):
        """
        Convert all format values in the header to fit with the type that it is
        stored in.

        Returns
        -------
        Header
            Returns self.

        """
        for data_type, name in self.format:
            value = getattr(self, name)
            
            new_value = convert_type(value, data_type)
            setattr(self, name, new_value)
        
        return self