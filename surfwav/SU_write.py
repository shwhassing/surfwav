import numpy as np
from struct import pack, unpack, calcsize
from surfwav.header import Header
from surfwav.util import conv_endian

su_data_fmt = "f" # Data format for SU traces, is float (4 bytes)
# su_header_length = 240 # Length of SU header in bytes
COH_data_fmt = "f"

def read_header(byte_header, endian = "little"):
    """
    When given a Seismic Unix header in bytes, read it and convert to a Header
    object.

    Parameters
    ----------
    byte_header : bytes
        The header as loaded in bytes.
    endian : str, optional
        The endian used to read the header. See the input of conv_endian for
        options. The default is "little".

    Returns
    -------
    header : Header
        The header with the entries read from the bytes.

    """
    # Get the byte representation for this format for unpack
    byte_fmt = Header(fmt='SU').binary_fmt(endian=endian)
    
    # Unpack the provided bytes with the format
    info_header = unpack(byte_fmt, byte_header)
    
    # Initialise a header
    header = Header()
    
    # Now go over each entry in the header format and get the corresponding
    # value from the read bytes
    i = 0
    for entry in Header(fmt="SU").format:
        # If an array entry is requested, return an array from the following
        # entries
        if entry[0][-5:] == "array":
            array_len = int(entry[0][1:-5])
            setattr(header, entry[1], np.array(info_header[i:i+array_len]))
            i += array_len
            # Else just get a single value
        else:
            setattr(header, entry[1], info_header[i])
            i += 1
    
    return header

def read_trace(byte_trace, endian="little"):
    """
    Read trace data from bytes. Takes format of the numbers as defined in 
    module, namely 'f' for float

    Parameters
    ----------
    byte_trace : bytes
        Bytes that are converted to number data.
    endian : str, optional
        The endian used to read the header. See the input of conv_endian for
        options. The default is "little".

    Returns
    -------
    np.ndarray
        Array containing the number data of a trace.

    """
    endian = conv_endian(endian)
    
    # Set up the binary format to read the trace for unpack
    fmt_trace = endian + su_data_fmt*int(len(byte_trace)/calcsize(su_data_fmt))
    
    # Read the provided bytes and convert to an array
    return np.array(unpack(fmt_trace,byte_trace))

def bin_to_header(file, endian="little"):
    """
    Reads the following bytes from a file and convert to a Header object

    Parameters
    ----------
    file : _io.TextIOWrapper
        File from which the bytes are read. Assumes the file is given at the
        right location
    endian : str, optional
        The endian used to read the header. See the input of conv_endian for
        options. The default is "little".

    Returns
    -------
    header : Header
        Header read from the bytes.

    """
    # Read the following bytes from the file for the header
    byte_header = file.read(calcsize(Header("SU").binary_fmt()))
    # Convert these bytes to a header object
    header = read_header(byte_header, endian=endian)
    
    return header

def bin_to_trace(file, amt_samples, endian="little"):
    """
    Reads the following bytes from file based on the amount of samples that are
    specified to obtain.

    Parameters
    ----------
    file : _io.TextIOWrapper
        The file from which the bytes are read. Assumes that the file reads 
        from the correct location
    amt_samples : int
        The amount of samples that are available in the trace.
    endian : str, optional
        The endian used to read the header. See the input of conv_endian for
        options. The default is "little".

    Returns
    -------
    data_trace : np.ndarray
        Array containing the data read from the file.

    """
    
    # Read the right amount of bytes from the file, determined by the size
    # of each float and the amount of samples in the trace
    byte_trace = file.read(amt_samples*calcsize(su_data_fmt))
    # Convert the bytes to a trace
    data_trace = read_trace(byte_trace, endian=endian)
    
    return data_trace

def check_trace_size(data):
    """
    Checks if all arrays in the given list of arrays are the same length. 
    Returns a boolean

    Parameters
    ----------
    data : list
        List containing np.ndarray for which the length is checked.

    Returns
    -------
    bool
        Whether or not all arrays in the given list are the same length.

    """
    
    # Initialise array with lengths of each entry of data
    lens = np.zeros(len(data))
    
    # Go over each trace in the data
    for i,line in enumerate(data):
        # and get its length
        lens[i] = len(line)
    
    # If they are all the same return True, otherwise False
    if np.all(lens == lens[0]):
        return True
    else:
        return False

def read_su(filename, endian = "little"):
    """
    Reads a .su file when given a filename and returns a Gather object with
    the data from the file

    Parameters
    ----------
    filename : str
        Path to the file that should be read.
    endian : str, optional
        The endian used to read the header. See the input of conv_endian for
        options. The default is "little".

    Raises
    ------
    NotImplementedError
        If the traces in .su file are not the same size.

    Returns
    -------
    Gather
        Gather containing the data that is in the given file.

    """
    
    # Initialise the headers and data
    headers = []
    data = []    
    
    # Open the right file as a binary file
    with open(filename, mode='rb') as file:
        
        # Read the first header at the start of the file
        headers.append(bin_to_header(file, endian=endian))
        # Determine how many traces are in the file
        amt_traces = headers[0].ntr
        
        # Then get the amount of samples in the first trace
        amt_samples = headers[0].ns
        # Add the first trace to the data
        data.append(bin_to_trace(file, amt_samples, endian=endian))
        
        # Now go over each other trace and repeat
        for i in range(amt_traces-1):
            headers.append(bin_to_header(file))
            
            amt_samples = headers[-1].ns
            data.append(bin_to_trace(file,amt_samples))

    # XXX Add padding
    # Check if all traces are the same size
    if check_trace_size(data):
        # Convert the data to a single array
        data = np.array(data)
    else:
        raise NotImplementedError("Not all traces are the same size")
    
    # Return a gather with the given data
    return data, headers

def write_su(filename, gather, endian = 'little'):
    """
    Writes a binary Seismic Unix .su file when given a data array and the 
    headers. 

    Parameters
    ----------
    filename : str
        Filename of the output .su file.
    data : np.ndarray
        Array containing the data. Must have a shape that fits [amt. of traces,
                                                                amt. of samples]
    headers : list
        List containg the header information for each trace.
    endian : str, optional
        Endian format for the data. Can be the strings 'little' or 'big' or the
        identifiers '<' or '>'. The default is 'little'.

    Returns
    -------
    None.

    """
        
    # Convert the endian from a string to the required specifier
    endian = conv_endian(endian)
    
    # Open a binary file
    with open(filename, 'wb') as file:
        
        # Go over each header
        for data, header in gather:
            # Get the binary format for the header
            bin_header_fmt = header.binary_fmt()
            # Get the data in the header as a tuple
            values = header.force_types().fit_binary().get_tuple()
            # Write this to the file
            file.write(pack(bin_header_fmt,*values))
            
            # Now get the binary format for the data in the trace
            bin_data_fmt = endian + str(gather.ns) + su_data_fmt
            # And write the trace to the file
            file.write(pack(bin_data_fmt,*data))
            
def write_COH(filename, data, stack_count, df, endian="little"):
    endian = conv_endian(endian)
    
    # Get the size of the dataset
    ntr = data.shape[0]
    nf = data.shape[-1]
    
    # Open the new file
    with open(filename, 'wb') as file:
        # Go over each receiver pair
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # If
                if stack_count[i,j] != 0:
                    header = Header("COH")
                    header.ntr = ntr
                    header.nf = nf
                    header.df = df
                    
                    header.idx0 = i
                    header.idx1 = j
                    
                    bin_header_fmt = header.binary_fmt()
                
                    values = header.force_types().fit_binary().get_tuple()
                    
                    file.write(pack(bin_header_fmt, *values))
                    
                    bin_data_fmt = endian + str(nf) + COH_data_fmt
                    
                    file.write(pack(bin_data_fmt, *data[i,j,:]))