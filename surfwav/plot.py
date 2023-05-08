# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:08:06 2023

@author: Sverre Hassing
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
import numpy as np
from surfwav.SU_write import read_su
from surfwav.util import point_to_line

def get_name_dist(dist_size):
    num_dist_list = [1,1000,0.3048,1609.344]
    name_dist_list = ['m','km','ft','mi']
    
    if isinstance(dist_size,float) or isinstance(dist_size,int):
        if dist_size in num_dist_list:
            return name_dist_list[num_dist_list.index(dist_size)], dist_size
        else:
            return f"m/{dist_size}", dist_size
    elif dist_size in name_dist_list:
        dist_size = num_dist_list[name_dist_list.index(dist_size)]
        return name_dist_list[num_dist_list.index(dist_size)], dist_size
    else:
        raise ValueError(f"{dist_size} is not a valid option for dist_size. Can be float, int or {name_dist_list}")

def plot_spread(files, axs, model):
    """
    Plot the receiver and source positions above the top layer of the subsurface
    model when given the model and data files. 

    Parameters
    ----------
    files : list
        A list with the path to each of the files.
    axs : list
        List with plt.axes._subplots.AxesSubplot on which the spread of each
        file is plotted separately.
    model : surfwav.model.SeisModel
        The model that belongs to the data.

    Returns
    -------
    None.

    """
    
    # Initialise the values that give the boundaries of the array
    minx = 0
    maxx = 0
    
    for j,file in enumerate(files):
        # Read in the file that was used for the transform
        gather = read_su(file)
        
        # Get the source and receiver positions
        src_pos = gather.unq_src_pos()
        rec_pos = gather.unq_rec_pos()
        
        # Determine the boundaries of the whole array
        minx = min([minx, np.min(src_pos[:,0]), np.min(rec_pos[:,0])])
        maxx = np.max([maxx, np.max(src_pos[:,0]), np.max(rec_pos[:,0])])
            
        # Plot the source and receiver positions
        axs[j].scatter(src_pos[:,0], src_pos[:,1], marker='*')
        axs[j].scatter(rec_pos[:,0], rec_pos[:,1], marker='v',s=1)
        
        cols = []
        for val in model.vs[:,0]:
            cols.append((val/max(model.vs[:,0]),0,0))
        
        ylim = axs[j].get_ylim()
        axs[j].scatter(model.x, np.zeros(model.shape[0])+ylim[0], color=cols)
        axs[j].set_ylim(ylim)
        
        axs[j].yaxis.set_visible(False)
    
    # Now set the limits for the array examples
    for ax in axs:
        border = 0.1*(maxx-minx)
        ax.set_xlim([minx-border,maxx+border])
        
def disp_power(f, vel, fdbf, figure=None, plot_max=False, normalise='max'):
    """
    Plot a frequency-phase velocity transform when given the data.

    Parameters
    ----------
    f : np.ndarray
        The frequency axis of the data.
    vel : np.ndarray
        The phase-velocity axis of the data.
    fdbf : np.ndarray
        The f-c transform that will be plotted.
    figure : tuple
        Tuple containing the figure and axis on which the plot must be drawn
        as (fig, ax). If not provided, will initialise a new figure.  The 
        default is None.
    normalise : str, optional
        How to normalise the data. The default is 'max'. Can be:
            - 'max'
            By the maximum of each trace
            - None
            Do not normalise the data

    Returns
    -------
    fig : plt.figure.Figure
        Figure object on which the plot is drawn.
    ax : plt.axes._subplots.AxesSubplot
        Axis on which the plot is drawn.

    """
    
    if normalise == 'max':
        fdbf_plot = fdbf/np.max(fdbf,axis=0)[np.newaxis,:]
    elif normalise is None:
        fdbf_plot = fdbf

    if figure is None:
        fig, ax = plt.subplots(dpi=300,figsize=(7,7))
    else:
        fig, ax = figure
        
    im = ax.imshow(abs(fdbf_plot),
              aspect='auto',
              extent=[f[0], f[-1], vel[0], vel[-1]],
              origin='lower',
              cmap='jet')
    if plot_max:
        # print(np.argmax(fdbf_plot, axis=1).shape, f.shape, vel.shape)
        plt.scatter(f, vel[np.argmax(fdbf_plot, axis=0)], marker='x', color='black')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase velocity [m/s]")
    im.set_clim([0.,1.])
    fig.colorbar(im, ax=ax)
    
    return fig, ax

def plot_ft(f,dists, spec, figure=None):
    """
    Plot the Fourier transform when given the data

    Parameters
    ----------
    f : np.ndarray
        The frequency axis of the data.
    dists : np.ndarray
        The space axis of the data.
    spec : np.ndarray
        The Fourier transform that will be plotted.
    figure : tuple
        Tuple containing the figure and axis on which the plot must be drawn
        as (fig, ax). If not provided, will initialise a new figure.  The 
        default is None.

    Returns
    -------
    fig : plt.figure.Figure
        Figure object on which the plot is drawn.
    ax : plt.axes._subplots.AxesSubplot
        Axis on which the plot is drawn.
    """
    
    if figure is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        fig, ax = figure
    
    ax.imshow(abs(spec).T, 
              extent=[dists[0], dists[-1], f[0],f[-1]],
              origin='lower',
              aspect='auto')
    ax.set_xlabel("Distance along line")
    ax.set_ylabel("Frequency [Hz]")
    
    return fig, ax

def plot_fk(f,k,fk_spectrum, figure=None):
    """
    Plot a given f-k spectrum. 

    Parameters
    ----------
    f : np.ndarray
        Frequency axis of the data.
    k : np.ndarray
        Wavenumber axis of the data.
    fk_spectrum : np.ndarray
        The f-k data that will be plotted.
    figure : tuple
        Tuple containing the figure and axis on which the plot must be drawn
        as (fig, ax). If not provided, will initialise a new figure.  The 
        default is None.

    Returns
    -------
    fig : plt.figure.Figure
        Figure object on which the plot is drawn.
    ax : plt.axes._subplots.AxesSubplot
        Axis on which the plot is drawn.

    """
    
    if figure is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        fig, ax = figure
    
    ax.imshow(np.abs(fk_spectrum).T,
              extent=[k[0],k[-1],f[0],f[-1]],
              origin='lower',
              aspect='auto')
    ax.set_xlabel("Wavenumber [1/m]")
    ax.set_ylabel("Frequency [Hz]")
    
    return fig, ax

def plot_section_map(gather,
                     figure=None,
                     figsize=(10,6),
                     dpi=300,
                     recordlength=None,
                     dist_method='offset',
                     plot_source=True,
                     orient=None,
                     intsect=None,
                     intsect_len = 0.1,
                     fs=11,
                     dist_size=1,
                     save=False,
                     out_file=None,
                     cmap='seismic',
                     **kwargs):
    """
    Function that plots a supplied stream as a seismic section with a 
    colourmap. All extra keyword arguments are supplied to the 
    matplotlib.pyplot.imshow function. 

    Parameters
    ----------
    gather : Gather
        Gather for which the traces are plotted.
    figure : tuple
        Tuple containing the figure and axis on which the plot must be drawn
        as (fig, ax)
    figsize : tuple, optional
        Size in inches of the resulting figure. The default is (10,6).
    dpi : float, optional
        Dots per inch of the resulting figure. The default is 300.
    recordlength : float, optional
        [s] Length of the record that is plotted. The default is None.
    orient : list, optional
        [A,B] List containing strings that are plotted at the top left and right
        of the plot to indicate the orientation of the line. String A is plotted
        left and B right. The default is None.
    intsect : float, optional
        [m] Intersection point of seismic lines indicated with red lines at the 
        top and bottom of the plot. A label with 'Intersection' is added at the
        bottom. A negative number can be supplied to count from the end of the
        line. The default is None.
    intsect_len : float, optional
        [inch] Length of the red lines used for the intersection line. Length
        required is based on the size of the plot, so can require some 
        experimentation. The default is 0.1.
    fs : float, optional
        Size of the font. The default is 11.
    dist_size : str or float, optional
        Conversion factor of the distances from metres used for the unit in the
        x-axis label. For example, if kilometres are used, dist_size is 1000. 
        For some values, it will get the name, or when a string is provided it 
        will fill in a number. Options are:
            1 - 'm'
            1000 - 'km'
            0.3048 - 'ft'
            1609.344 - 'mi'
        The default is 1.
    save : bool, optional
        Whether to save the plot. If True, a path for the new file needs to be
        given with out_file. The default is False.
    out_file : str, optional
        Path and filename of the output image if the plot is saved. The default
        is None.
    **kwargs
        Extra keyword arguments are given to the matplotlib.pyplot.imshow 
        function that plots the image.

    Raises
    ------
    ValueError
        If out_file is not defined, while save is True, so if no filename is 
        given while the image is saved an error is raised.

    Returns
    -------
    fig, ax : matplotlib.pyplot.figure.Figure, matplotlib.pyplot.axes._subplots.AxesSubplot
        Returns the Matlotlib figure and axis used to create the plot

    """
    
    # Determine which unit of distance is used
    name_dist,dist_size = get_name_dist(dist_size)
    
    # Enforce the recordlength
    gather = gather.trim(end=recordlength)
    
    # Extract the distance information
    if dist_method == 'dist':
        dists = gather.dists()
        dist_str = 'Distance along line'
    elif dist_method == 'offset':
        dists = np.array(gather.get_item('offset'))
        dist_str = 'Offset'
    else:
        raise ValueError(f"Invalid method ({dist_method}) provided, can be ['dist', 'offset']")
        
    # Normalise and extract data
    raw_data = gather.normalise().data
    
    # Take the maximum and minimum values
    maxima = [raw_data.max(),abs(raw_data.min())]
    
    # t_max = record[0].stats.delta*(record[0].stats.npts-1)
    
    if figure == None:
        # Create the plot
        fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
    else:
        fig,ax = figure
    
    t_max = gather.t[-1]
    
    # And plot the image
    ax.imshow(raw_data.T,
               extent=[dists[0],dists[-1],t_max,gather.t[0]],
               origin='upper',
               aspect='auto',
               cmap=cmap,
               vmin=-np.max(maxima),
               vmax=np.max(maxima),
               **kwargs
               )
    
    # Add the intersection point of the two lines as red lines outside the plot
    if intsect != None:
        # If the supplied distance is negative, take it from the end of the
        # line
        if intsect < 0:
            intsect_point = dists.max() + intsect/dist_size
        else:
            intsect_point = intsect/dist_size
        
        # Set up the two lines
        line0 = Line2D([intsect_point,intsect_point],[0-intsect_len,0],color='r')
        line1 = Line2D([intsect_point,intsect_point],[t_max,t_max+intsect_len],color='r')
        
        # Add the two lines
        line0.set_clip_on(False)
        ax.add_line(line0)
        
        line1.set_clip_on(False)
        ax.add_line(line1)
        
        # Add a descriptor
        ax.text(intsect_point,
                 t_max+1.7*intsect_len, 
                 "Intersection", 
                 color='r', 
                 rotation=0, 
                 horizontalalignment='center',
                 rotation_mode='anchor')
    
    # Add the orientation of the line above the plot, generally wind directions
    if orient != None:
        ax.text(dists.min(), 
                 0,
                 orient[0],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 fontsize=1.6*fs)
        ax.text(dists.max(), 
                 0,
                 orient[1],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 fontsize=1.6*fs)
        
    if gather.type == 'source' and plot_source:
        if dist_method == 'dist':
            # Fit the receiver line
            coef, intercept = gather.fit_line('rec')
            # Project the source position onto this line
            src_pos = point_to_line(coef, intercept, gather.unq_src_pos()[0][:-1]).squeeze()
            # Get the distance from the first receiver to the projected position
            src_dist = np.linalg.norm(gather.rec_pos()[0][:-1] - src_pos, axis=0)
        elif dist_method == 'offset':
            src_dist = 0
        # Plot a star on this location at t=0
        src_point = ax.scatter(src_dist, 0, marker='*', c='r', s=60, zorder=10)
        # Show the whole star
        src_point.set_clip_on(False)
    
    # Set labels
    ax.set_ylabel('Two-way traveltime [s]')
    ax.set_xlabel(f'{dist_str} [{name_dist}]')
    
    # Enforce the same boundaries as for the wiggle plot
    max_dist = dists.max() - dists.min()
    ax.set_xlim([dists.min()-0.05*max_dist, dists.max()+0.05*max_dist])
    
    # Set the ticks for both axes
    # XXX bit too hardcoded now
    ax.xaxis.set_major_locator(MultipleLocator(250/dist_size))
    ax.xaxis.set_minor_locator(MultipleLocator(50/dist_size))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    if save:
        if out_file == None:
            raise ValueError("File is saved without a filename")
        plt.savefig(out_file, transparent=True)
        
    return fig,ax
    
def plot_section_wiggle(gather,
                        figure=None,
                        figsize=(10,6),
                        dpi=300,
                        lc=(0,0,0),
                        la=0.5,
                        lw=0.4,
                        fill_color=(0.5,0.5,0.5),
                        dist_method='offset',
                        normalise='max-trace',
                        plot_source=True,
                        orient=None,
                        intsect=None,
                        intsect_len = 0.1,
                        dist_size=1,
                        recordlength=None,
                        tr_scale = 1.,
                        fs=11,
                        save=False,
                        out_file=None):
    """
    Creates a wiggle plot of a section based on the supplied stream. Positive 
    amplitudes can be filled by a rgb colour supplied with fill_color.

    Parameters
    ----------
    gather : Gather
        Gather containing the seismic data
    figure : tuple
        Tuple containing the figure and axis on which the plot must be drawn
        as (fig, ax)
    figsize : tuple, optional
        Size of the figure in inches. The default is (10,6).
    dpi : float, optional
        Dots per inch of the figure. The default is 300.
    lc : tuple, optional
        RGB colour of the trace lines. The default is (0,0,0).
    la : float, optional
        Transparency (alpha) of the seismic traces. The default is 0.5.
    lw : float, optional
        Width of the trace lines. The default is 0.4.
    fill_color : tuple, optional
        RGB colour filling the space between the zero-point of each trace and
        positive amplitudes. Should be supplied as (r,g,b). The default is None.
    orient : list, optional
        [A,B] List containing strings that are plotted at the top left and right
        of the plot to indicate the orientation of the line. String A is plotted
        left and B right. The default is None.
    intsect : float, optional
        [m] Intersection point of seismic lines indicated with red lines at the 
        top and bottom of the plot. A label with 'Intersection' is added at the
        bottom. A negative number can be supplied to count from the end of the
        line. The default is None.
    intsect_len : float, optional
        [inch] Length of the red lines used for the intersection line. Length
        required is based on the size of the plot, so can require some 
        experimentation. The default is 0.1.
    dist_size : str or float, optional
        Conversion factor of the distances from metres used for the unit in the
        x-axis label. For example, if kilometres are used, dist_size is 1000. 
        For some values, it will get the name, or when a string is provided it 
        will fill in a number. Options are:
            1 - 'm'
            1000 - 'km'
            0.3048 - 'ft'
            1609.344 - 'mi'
        The default is 1.
    recordlength : float, optional
        [s] Length of the record that is plotted. The default is None.
    tr_scale : float, optional
        Amplitude scaling of the traces. The default is 1..
    fs : float, optional
        Font size on the plot. The default is 11.
    save : bool, optional
        Whether to save the plot. If True, a path for the new file needs to be
        given with out_file. The default is False.
    out_file : str, optional
        Path and filename of the output image if the plot is saved. The default
        is None.

    Raises
    ------
    ValueError
        If out_file is not defined, while save is True, so if no filename is 
        given while the image is saved an error is raised.

    Returns
    -------
    fig, ax : matplotlib.pyplot.figure.Figure, matplotlib.pyplot.axes._subplots.AxesSubplot
        Returns the Matlotlib figure and axis used to create the plot

    """
    
    # Determine which unit of distance is used
    name_dist,dist_size = get_name_dist(dist_size)
        
    # Enforce the record length
    gather = gather.trim(end=recordlength)
    
    # Extract the distance information
    if dist_method == 'dist':
        dists = gather.dists()
        dist_str = 'Distance along line'
    elif dist_method == 'offset':
        dists = np.array(gather.get_item('offset'))
        dist_str = 'Offset'
    else:
        raise ValueError(f"Invalid method ({dist_method}) provided, can be ['dist', 'offset']")
                
    # Array with time information
    times = gather.t
    
    # Normalise and extract data
    raw_data = gather.normalise(normalise).data
    
    # Determine how much room is available for each trace, scaled by input parameter
    # tr_scale
    plot_ampl = (dists.max()-dists.min())/(1.5*len(dists))*tr_scale
    
    if figure == None:
        # Create the plot
        fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
    else:
        fig,ax = figure

    
    if intsect != None or orient != None:
        max_dist = dists.max()
    
    # Go over each trace
    for i,data in enumerate(raw_data):

        # Get the location of this trace
        centre = dists[i]/dist_size
        
        # Normalise the data for the plot around the location of the trace
        plot_normalised_data = data*plot_ampl + centre
        # Plot the result
        ax.plot(plot_normalised_data,
                 times,
                 color=lc,
                 lw=1.,
                 alpha=la)
        
        # Fill in the wiggles with the right colour
        if fill_color != None:
            ax.fill_betweenx(times,
                              centre,
                              plot_normalised_data,
                              where=plot_normalised_data>centre,
                              facecolor=fill_color)
        
    # Add the intersection point as red lines with text
    if intsect != None:
        # If the supplied distance is negative, subtract it from the end of the
        # line
        if intsect < 0:
            intsect_point = max_dist + intsect/dist_size
        else:
            intsect_point = intsect/dist_size
        
        # Set up the lines above and below the figure
        line0 = Line2D([intsect_point,intsect_point],[times[0]-intsect_len,times[0]],color='r')
        line1 = Line2D([intsect_point,intsect_point],[times[-1],times[-1]+intsect_len],color='r')
        
        # Add the two lines
        line0.set_clip_on(False)
        ax.add_line(line0)
        
        line1.set_clip_on(False)
        ax.add_line(line1)
        
        # Add the text
        ax.text(intsect_point,
                 times[-1]+1.7*intsect_len, 
                 "Intersection", 
                 color='r', 
                 rotation=0, 
                 horizontalalignment='center',
                 rotation_mode='anchor')
    
    if gather.type == 'source' and plot_source:
        if dist_method == 'dist':
            # Fit the receiver line
            coef, intercept = gather.fit_line('rec')
            # Project the source position onto this line
            src_pos = point_to_line(coef, intercept, gather.unq_src_pos()[0][:-1]).squeeze()
            # Get the distance from the first receiver to the projected position
            src_dist = np.linalg.norm(gather.rec_pos()[0][:-1] - src_pos, axis=0)
        elif dist_method == 'offset':
            src_dist = 0
        # Plot a star on this location at t=0
        src_point = ax.scatter(src_dist, 0, marker='*', c='r', s=60, zorder=10)
        # Show the whole star
        src_point.set_clip_on(False)
    
    # Add the orientation of the line as wind directions on top of the plot
    if orient != None:
        ax.text(0, 
                 times[0],
                 orient[0],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 fontsize=1.6*fs)
        ax.text(max_dist, 
                 times[0],
                 orient[1],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 fontsize=1.6*fs)
    
    # Set the limits of the time axis
    ax.set_ylim([times[-1],times[0]])
    # Set labels
    ax.set_xlabel(f'{dist_str} [{name_dist}]')
    ax.set_ylabel('Two-way traveltime [s]')
    
    # Set some space around the traces to be the same as the map plot
    max_dist = dists.max() - dists.min()
    ax.set_xlim([dists.min()-0.05*max_dist, dists.max()+0.05*max_dist])
    
    # Set up the ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # ax.xaxis.set_major_locator(MultipleLocator(250/dist_size))
    # ax.xaxis.set_minor_locator(MultipleLocator(50/dist_size))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    # Save the figure if requested
    if save:
        if out_file == None:
            raise ValueError("File is saved without a filename")
        plt.savefig(out_file, transparent=True)
        
    # Return the resulting figure
    return fig,ax