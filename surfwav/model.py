# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:52:19 2023

@author: Sverre Hassing
"""
import numpy as np
import matplotlib.pyplot as plt
from surfwav.wavelets import ricker, sweep
from surfwav.make_files import make_su, make_su_model
# from surfwav.SU_write import read_su
from surfwav.gather import Gather
from surfwav.util import freq_ft, get_idx

class SeisModel:
    
    def __init__(self, *args, filename=None, affix=None, which_model=None):
        if len(args) == 0 and filename == None:
            raise ValueError("Model needs axes or filename defined")
        elif len(args) >= 4:
            raise ValueError(f"Model has too many axes ({len(args)}) defined, can be 3 at maximum")
        
        # Add which models are included and whether certain components are added
        self.models = ['vp', 'vs', 'rho']
        self._got_source    = False
        self._got_vp        = False
        self._got_vs        = False
        self._got_rho       = False
        
        # If no filename is given, take the data from the arguments
        if filename == None:
            # The arguments describe the axes in the model
            self.ndim = len(args)
            
            # Get the names of each axis
            axis_names = self._axis_names()
            
            # Set each of the arguments to one of the axes
            for i,arg in enumerate(args):
                setattr(self, axis_names[i], args[i])
                # And step size of the data
                setattr(self, 'd'+axis_names[i], (args[i][-1]-args[i][0])/(len(args[i])-1))
            
            # Set the shape to the length of each axis
            self.shape = tuple([len(x) for x in args])
            # Initialise arrays for each axis
            self.vp = np.empty(self.shape)
            self.vs = np.empty(self.shape)
            self.rho = np.empty(self.shape)
            
        else:
            # If a filename is given, use that
            # The shape is set during the loading of a file
            self.shape = None
            
            # If the path points to the file for a specific model
            if which_model is not None:
                # Read in the data for the right model
                self.read_model(filename, which_model)
                # Set every other model to zeroes
                for other_model in self.models.pop(self.models.index(which_model)):
                    setattr(self, other_model, np.empty(self.shape))
            # If a general affix is provided, the files are assumed to fit the 
            # outpupt format, so that all files can be loaded with filename and
            # affix
            elif affix is not None:
                # Go over the models
                for which_model in self.models:
                    self.read_model(filename+f"//{which_model}_model{affix}.su", which_model)
                # Add the source wavelet
                self.read_model(filename+f"//source_wavelet{affix}.su", 'source')
            else:
                raise ValueError("If 'filename' is provided, either 'which_model' or 'affix' must be defined")
    
    def __eq__(self, obj, tol=1e-4):
        if not isinstance(obj, SeisModel):
            raise ValueError("Can only compare SeisModel with other SeisModel, not {type(obj)}")
        
        # First check the general shape and data content
        checks = [self.ndim != obj.ndim,
                  self.shape != obj.shape,
                  not any([getattr(self, f"d{x}") - getattr(obj, f"d{x}") <= tol for x in self._axis_names()]),
                  not any([getattr(self, f'{x}')[0] - getattr(obj, f'{x}')[0] <= tol for x in self._axis_names()]),
                  self.models != obj.models]
        
        # If they are not all the same, return False
        if any(checks):
            return False
                
        # Then go over the data in the medium properties
        for which_model in self.models:
            # Check if they both have the same models
            if getattr(self, f"_got_{which_model}") != getattr(obj, f"_got_{which_model}"):
                return False
            # For the models that both have, check if the models are the same
            if np.all(getattr(self, which_model) - getattr(obj, which_model) > tol) and getattr(self, f"_got_{which_model}"):
                return False
                            
        # Check if both have a source wavelet
        if self._got_source != obj._got_source:
            return False
        # If there is a source wavelet, check if it is the same
        if self._got_source:
            source_checks = [self.dt != obj.dt,
                             np.all(self.source_ampl - obj.source_ampl > tol),
                             np.all(self.source_t - obj.source_t > tol)]
            if any(source_checks):
                return False
        
        # If all checks fail, return True
        return True
    
    def _check_complete(self):
        """
        Checks if all necessary data is set to export the model.

        Returns
        -------
        bool
            Whether all necessary data is provided.

        """
        return all([self._got_source,
                    self._got_vp,
                    self._got_vs,
                    self._got_rho])
    
    def _axis_names(self):
        """
        Based on the amount of dimensions sets the names of all axes. 

        Returns
        -------
        list
            List with the names of all axes.

        """
        if self.ndim == 1:
            return ['z']
        elif self.ndim == 2:
            return ['x','z']
        elif self.ndim == 3:
            return ['x', 'y', 'z']
    
    def _get_label(self, which_model):
        """
        Get the label of the data for the specified model.

        Parameters
        ----------
        which_model : str
            Which model to get the label for.

        Returns
        -------
        str
            String describing the data for this model.

        """
        labels = ['P-velocity [m/s]','S-velocity [m/s]','Density [kg/m^3]']
        return labels[self.models.index(which_model)]
    
    def _check_dispersion(self):
        """
        Check if numerical dispersion will be a factor when using the current
        settings of the model with fdelmod.

        Raises
        ------
        ValueError
            If not all models are set.

        Returns
        -------
        bool
            Whether or not dispersion will be a significant factor.

        """
        if not self._check_complete():
            raise ValueError("Not all models are added yet")
        
        max_dx = 0
        for d in self._axis_names():
            min_dx = max(max_dx, getattr(self, 'd'+d))
        min_v = min(np.min(self.vp), np.min(self.vs))
        
        fmax = self.get_fmax()
        
        if min_dx >= min_v/(5*fmax):
            print("WARNING: Modelling is likely to be dispersive with these settings")
            print(f"Options are:\n\t-\tDecrease dz to below {min_v/(5*fmax)} m")
            print(f"\t-\tDecrease fmax below {min_v/(5*min_dx)} Hz")
            print(f"\t-\tIncrease the minimum velocity to {fmax*min_dx*5} m/s")
            
            return True
        return False
    
    def _check_cfl(self):
        """
        Whether or not the CFL-criterion is met for modelling with fdelmod with
        the current settings of the model. 

        Raises
        ------
        ValueError
            If not all models are set.

        Returns
        -------
        bool
            Whether the CFL criterion is met.

        """
        
        if not self._check_complete():
            raise ValueError("Not all models are added yet")
        
        min_dx = np.inf
        for d in self._axis_names():
            min_dx = min(min_dx, getattr(self, 'd'+d))
        max_v = max(np.max(self.vp), np.max(self.vs))
        
        if self.dt >= 0.606*min_dx/max_v:
            print("WARNING: The CFL criterion for the 4th order spatial derivatives is not met")
            print(f"Options are:\n\t-\tDecrease dt below {0.606*min_dx/max_v} s")
            print(f"\t-\tIncrease smallest space step to {self.dt*max_v/0.606} m")
            print(f"\t-\tDecrease the maximum velocity to {0.606*min_dx/self.dt} m/s\n")
            
            return True
        return False
    
    def _check_model_shape(self, shape, header):
        """
        Check if the size of the model loaded with header is the same as this
        model

        Parameters
        ----------
        shape : tuple
            The size of the new model.
        header : surfwav.Header
            The header loaded from the data.

        Raises
        ------
        IndexError
            If the size of the new model is not the same as this model.
        ValueError
            If the origin or step size is not the same as in the data.

        Returns
        -------
        bool
            True if everythiing is the same.

        """
        
        x_origin = header.f2
        model_dx = header.d2
        z_origin = header.f1
        model_dz = header.d1
        
        if self.shape == None:
            return None
        else:
            
            if self.shape != shape:
                raise IndexError(f"Provided data ({shape}) has different shape than self ({self.shape})")
            elif z_origin != min(self.z):
                raise ValueError(f"Origin of z-axis in provided model ({z_origin}) is different than own origin ({min(self.z)})")
            elif model_dz != self.dz:
                raise ValueError(f"Vertical step dz in provided model ({model_dz}) is different than own dz ({self.dz})")
            if self.ndim == 2:
                if x_origin != min(self.x):
                    raise ValueError(f"Origin of x-axis in provided model ({x_origin}) is different than own origin({min(self.x)})")
                if model_dx != self.dx:
                    raise ValueError(f"Horizontal step dx in provided model ({model_dx}) is different than own dx ({self.dx})")
            
            return True
    
    def _check_export(self):
        """
        Check if it is possible to export the current file

        Raises
        ------
        ValueError
            If various of the checks fail.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        
        if not self._check_complete():
            raise ValueError("The source wavelet and all values must be defined before exporting")
        elif self._check_cfl():
            raise ValueError("CFL criterion is not met")
        elif self._check_dispersion():
            raise ValueError("Model will be dispersive")
            
        return True
    
    def get_src_wvl(self, method, *args, **kwargs):
        """
        Calculate a source wavelet with functions surfwav.wavelets.ricker or
        surfwav.wavelets.sweep. See those functions for (keyword) arguments.

        Parameters
        ----------
        method : str
            Which method to use to calculate a source wavelet. Can be 'ricker'
            or 'sweep'
        *args : tuple
            The arguments for the source wavelet function.
        **kwargs : dict
            The keyword arguments for the source wavelet function.

        Raises
        ------
        ValueError
            If no valid value for method is provided.

        Returns
        -------
        None.

        """
        methods = ['ricker', 'sweep']
        
        if method == 'ricker':
            t,u = ricker(*args, **kwargs)
        elif method == 'sweep':
            t,u = sweep(*args, **kwargs)
        else:
            raise ValueError(f"'{method}' is not a valid method, can be {methods}")
        
        self.dt = args[0]
        self.rec_delay = abs(t[0])
        self.source_t = t - t[0]
        self.source_ampl = u
        self._got_source = True
    
    def source_wavelet(self, t, u, rec_delay = 0.):
        """
        Define a source wavelet by giving the time axis and displacement. 

        Parameters
        ----------
        t : np.ndarray
            The time axis of the source wavelet.
        u : np.ndarray
            The displacement of the source wavelet.
        rec_delay : float, optional
            The receiver delay of the source wavelt. The default is 0.

        Returns
        -------
        None.

        """
        
        self.source_t = t - t[0]
        self.dt = self.source_t[1]
        self.source_ampl = u
        self.rec_delay = rec_delay
        self._got_source = True
        
    def get_fmax(self):
        """
        Calculate the maximum frequency of the source wavelet according to the
        calculation used in fdelmod

        Raises
        ------
        ValueError
            If no source wavelet is defined yet.

        Returns
        -------
        float
            The maximum frequency of the source wavelet.

        """
        
        if not self._got_source:
            raise ValueError("Source wavelet is not yet defined")
        
        f, spec = freq_ft(self.source_ampl, self.dt, 100*len(self.source_ampl), give='abs')

        idx_fpeak = np.argmax(spec)
        idx_fmax = np.argwhere(spec[idx_fpeak:] <= 0.0025 * spec[idx_fpeak])[0][0] + idx_fpeak
        return f[idx_fmax]
    
    def read_model(self, filename, which_model):
        """
        Read in a model from a seismic unix .su file. Which model is provided
        is given by which_model. 

        Parameters
        ----------
        filename : str
            Path to the file that is read in.
        which_model : str
            Which model can be read from this file.

        Returns
        -------
        None.

        """
        # Read in the su file to a Gather
        model_gather = Gather(filename=filename)
        # model_gather = read_su(filename)
        # Take the first header for some information
        header = model_gather.headers[0]
        
        # If which_model is one of the models
        if which_model in self.models:
            # Get the shape of the data
            data_shape = model_gather.data.shape
            # And some information about the axes
            x_origin = header.f2
            model_dx = header.d2
            z_origin = header.f1
            model_dz = header.d1
                        
            # If no shape is defined yet, take it from the file
            if self.shape is None:
                self.shape = model_gather.data.squeeze().shape
                
                self.dz = model_dz
                self.z = np.arange(z_origin, z_origin+(self.shape[1])*self.dz, self.dz)
                
                if data_shape[0] != 1:
                    self.ndim = 2
                    self.dx = model_dx
                    self.x = np.arange(x_origin, x_origin+(self.shape[0])*self.dx, self.dx)
                else:
                    self.ndim = 1
            
            # Check if the data has the right shape
            self._check_model_shape(model_gather.shape, header)
            
            # Read the data
            setattr(self, f'_got_{which_model}', True)
            setattr(self, which_model, model_gather.data)
            
        elif which_model == 'source':
            # If a source wavelet is supplied, read the data
            self.dt = header.dt/1e6
            self.source_ampl = model_gather.data.squeeze()
            self.source_t = np.arange(0, len(self.source_ampl)*self.dt, self.dt)
            self._got_source = True
            
    def layer_model(self, which_model, layer_tops, vals):
        """
        Create a simple horizontally layered model. It is defined by the top
        of each layer and the value for that layer. 

        Parameters
        ----------
        which_model : str
            Which model to apply the new data to.
        layer_tops : np.ndarray
            The top of each layer.
        vals : np.ndarray
            The value belonging to that layer.

        Raises
        ------
        ValueError
            If the amount of layer_tops and values is not the same.

        Returns
        -------
        None.

        """
        
        if len(layer_tops) != len(vals):
            raise ValueError(f"Length of the layer tops ({len(layer_tops)}) and the new values ({len(vals)}) are not the same")
        
        # Get the model and set all values to the bottom layer
        model = getattr(self, which_model)
        model = model*0 + vals[-1]
        
        # Now go over all the other layers and set the right value
        for i in np.arange(0,len(vals)-1):
            idx0 = get_idx(layer_tops[i], min(self.z), self.dz)
            idx1 = get_idx(layer_tops[i+1], min(self.z), self.dz)
                
            model[...,idx0:idx1] = vals[i]
        
        # Set this model
        setattr(self, which_model, model)
        setattr(self, f'_got_{which_model}', True)
    
    def vs_to_vp(self, method, *args):
        """
        Convert a vs model to a vp model. Currently only multiply is implemented.

        Parameters
        ----------
        method : str
            The conversion method. Can be:
                - multiply
                If a single argument is provided, all data is multiplied with
                this one. If there are two, it is assumed that the first
                argument contains multipliers for each layer, while the second
                one contains the top of each layer. 
        *args : tuple
            The arguments for the method.

        Raises
        ------
        ValueError
            If the value of method is not a valid one.

        Returns
        -------
        None.

        """
        
        if method == 'multiply':
            try:
                len(args[1])
            except TypeError:
                layers = [args[1]]
            else:
                layers = args[1]
            
            # If there are two arguments
            if len(args) == 2:
                layer_tops = np.zeros(len(layers)+1)
                layer_tops[:-1] = layers
                layer_tops[-1] = self.shape[-1]
            else:
                layer_tops = [0,self.shape[-1]]
            
            # Extract the multipliers from args, if there is a single one, put
            # it into a list
            multipliers = args[0]
            try:
                iter(multipliers)
            except TypeError:
                multipliers = [multipliers]
            
            # Apply the layered multiplication
            idcs = get_idx(layer_tops, min(self.z), self.dz)
            for i, mult in enumerate(multipliers):
                self.vp[:,idcs[i]:idcs[i+1]] = self.vs[:,idcs[i]:idcs[i+1]] * mult
                
            self._got_vp = True
        else:
            raise ValueError(f"{method} is not a valid method, can be 'multiply'")
    
    def vp_to_vs(self, method, *args):
        """
        Convert a vp model to a vs model. Currently only multiply is implemented.

        Parameters
        ----------
        method : str
            The conversion method. Can be:
                - multiply
                If a single argument is provided, all data is multiplied with
                this one. If there are two, it is assumed that the first
                argument contains multipliers for each layer, while the second
                one contains the top of each layer. 
        *args : tuple
            The arguments for the method.

        Raises
        ------
        ValueError
            If the value of method is not a valid one.

        Returns
        -------
        None.

        """
        
        if method == 'multiply':
            try:
                len(args[1])
            except TypeError:
                layers = [args[1]]
            else:
                layers = args[1]
            
            # If there are two arguments
            if len(args) == 2:
                layer_tops = np.zeros(len(layers)+1)
                layer_tops[:-1] = layers
                layer_tops[-1] = self.shape[-1]
            else:
                layer_tops = [0,self.shape[-1]]
            
            # Extract the multipliers from args, if there is a single one, put
            # it into a list
            multipliers = args[0]
            try:
                iter(multipliers)
            except TypeError:
                multipliers = [multipliers]
            
            # Apply the layered multiplication
            idcs = get_idx(layer_tops, min(self.z), self.dz)
            for i, mult in enumerate(multipliers):
                self.vs[:,idcs[i]:idcs[i+1]] = self.vp[:,idcs[i]:idcs[i+1]] * mult
                
            self._got_vs = True
        else:
            raise ValueError(f"{method} is not a valid method, can be 'multiply'")
    
    def vp_to_rho(self, method):
        """
        Convert vp to rho. Currently only Gardner's relation is implemented. 

        Parameters
        ----------
        method : str
            String describing what method to use for the conversion.

        Raises
        ------
        ValueError
            If no valid value for method is provided.

        Returns
        -------
        None.

        """
        
        if method == 'Gardner':
            self.rho = 0.31*((self.vp)**0.25)*1000
            self._got_rho = True
        else:
            raise ValueError(f"{method} is not a valid method, can be 'Gardner'")
            
    def plot(self, which_model, idx = 0, figure=None, **plot_kwargs):
        """
        Plot one of the models in the object. Figure adapts for the amount of 
        dimensions. For a 3D model, a slice of the data is used. 

        Parameters
        ----------
        which_model : str
            Which model should be plotted.
        idx : int, optional
            Which slice of the y-axis is plotted for a 3D model. The default is 0.
        figure : tuple, optional
            Tuple containing (fig, ax) to plot the data on. The default is None.
        **plot_kwargs : dict
            Keyword arguments for the plot.

        Returns
        -------
        None.

        """
        
        # If no figure is given, initialise a new one
        if figure == None:
            fig, ax = plt.subplots(1,1,**plot_kwargs)
        else:
            fig, ax = figure
        
        model = getattr(self, which_model)
        if self.ndim == 1:
            ax.plot(model,self.z)
            ax.set_ylim([max(self.z), min(self.z)])
            
            ax.set_xlabel(self._get_label(which_model))
            ax.set_ylabel("Depth [m]")
            ax.grid()
            
        elif self.ndim == 2:
            im = ax.imshow(model.T,
                      aspect = 'auto',
                      extent = [self.x[0], self.x[-1], self.z[-1], self.z[0]])
            ax.set_xlabel("Surface distance [m]")
            ax.set_ylabel("Depth [m]")
            ax.xaxis.set_label_position('top') 
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            fig.colorbar(im, ax=ax,
                                label=self._get_label(which_model))
            
            
        elif self.ndim == 3:
            im = ax.imshow(model[:,idx,:].T,
                           aspect = 'auto',
                           extent = [min(self.x), max(self.x), max(self.z), min(self.z)])
            ax.set_xlabel("Surface distance [m]")
            ax.set_ylabel("Depth [m]")
            fig.colorbar(im, ax=ax,
                                label=self._get_label(which_model))
        
    def plot_all(self, **plot_kwargs):
        """
        Plot all models in the object

        Parameters
        ----------
        **plot_kwargs : dict
            Keyword arguments for the figure.

        Returns
        -------
        None.

        """
        
        amt_models = len(self.models)
        
        fig, axs = plt.subplots(amt_models, 1, figsize=(6,12), **plot_kwargs)
        
        for i, which_model in enumerate(self.models):
            self.plot(which_model, figure=(fig,axs[i]))
            
    def plot_source(self):
        """
        Plot the source wavelet

        Raises
        ------
        ValueError
            If no source wavelet has been defined.

        Returns
        -------
        None.

        """
        
        if not self._got_source:
            raise ValueError("This model has not source wavelet defined")
            
        plt.figure(dpi=300)
        plt.plot(self.source_t,self.source_ampl)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Source wavelet")
        plt.show()
    
    def plot_freq(self):
        """
        Plot the frequency spectrum of the source wavelet
        
        Raises
        ------
        ValueError
            If no source wavelet is defined.

        Returns
        -------
        None.
        """
        
        if not self._got_source:
            raise ValueError("This model has not source wavelet defined")
        
        f, spec = freq_ft(self.source_ampl, self.dt, 100*len(self.source_ampl), give='abs')

        fp = f[np.argmax(spec)]
        fmax = self.get_fmax()

        plt.figure(dpi=300)
        plt.plot(f,spec)
        plt.xlim([0,1.1*fmax])
        ylim = plt.gca().get_ylim()
        plt.ylim([0,ylim[1]])
        plt.xlabel("Frequency [Hz]")
        plt.vlines(fp, 0, ylim[1], color='r', ls='--', zorder=1)
        plt.text(1.05*fp, 
                  0.04*ylim[1], 
                  f'Peak frequency\n{round(fp)} Hz', 
                  c='r')

        plt.vlines(fmax, 0, ylim[1], color='g', ls='--', zorder=1)
        plt.text(0.98*fmax, 
                  0.95*ylim[1], 
                  f'Max. frequency\n{round(fmax)} Hz', 
                  c='g',
                  horizontalalignment='right',
                  verticalalignment='top')
        plt.show()
            
    def write_su(self, folder_output, affix = '', endian = 'little', force=False):
        """
        Write the different models to a su file. It creates four files:
            vp_model{affix}.su
            vs_model{affix}.su
            rho_model{affix}.su
            source_wavelet{affix}.su
        These are also added to the output list. The affix can be used to 
        differentiate with different models.

        Parameters
        ----------
        folder_output : str
            Path to the output folder.
        affix : str, optional
            An affix for the filenames. The default is ''.
        endian : str, optional
            What endian to use for the file. The default is 'little'.
        force : bool, optional
            Whether to force the export of the files. The default is False.

        Returns
        -------
        fnames : list
            A list with all paths of the resulting files.

        """
        
        # Check the data for export
        if not force:
            self._check_export()
        
        # Write all the models
        fnames = []
        for model in self.models:
            filename = folder_output + f"//{model}_model" + affix + ".su"
            fnames.append(filename)
            make_su_model(filename, getattr(self, model), self.dx, self.dz, origin=(min(self.x),min(self.z)), endian=endian)
        
        # Write the source wavelet
        filename = folder_output + f"//source_wavelet{affix}.su"
        fnames.append(filename)
        make_su(filename, self.source_ampl, self.dt, 0, method_offset='simple')
        
        return fnames