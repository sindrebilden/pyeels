import hyperspy.api as hs
from pyeels.cpyeels import calculate_spectrum, calculate_momentum_squared
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import signal as sign
import time
import logging
_logger = logging.getLogger(__name__)

class EELS:
    
    temperature = 0
    fermienergy = 0
    
    def __init__(self, crystal, name=None):
        self.crystal = crystal
        

    def set_diffractionzone(self, zone=None, bins=None):
        """ Define the resolution of the diffraction space, similar to the CCD in TEM
        
        :type  zone: ndarray
        :param zone: The range of the diffraction space, a value containing the full brillouin zone is calculated if no value given
        
        :type  bins: ndarray
        :param bins: The number of bins in diffraction space, a value corresponding to the resolution of the brillouin zone is calculated if no value given. Will allways round up to an odd number
        """
        if not zone:
            self.zone = np.max(np.abs(self.crystal.brillouinzone.lattice),axis=0)
        else:
            self.zone = zone
            
        if not bins:
            self.bins = np.round(self.crystal.brillouinzone.mesh).astype(int) #/1.3 is a bad temporarly solution
        else:
            self.bins = bins
        
        for i in range(len(self.bins)):
            if (self.bins[i]%2==0):
                self.bins[i] += 1
        

    def set_meta(self, name, authors, title=None, notes=None):
        """ Set the user defined metadata
        
        :type  name: str
        :param name: the name of the simulation experiment
        
        :type  authors: str, list
        :param authors: The name of authors contributing to the specrum

        :type  title: str
        :param title: The title in the spectrum, should describe the crystal system

        :type  notes: string
        :param notes: Additional notes that is convenient for a user
        """
        
        self.name = name
        self.authors = authors
        
        if not title:
            self.title = self.name
        else:
            self.title = title
        
        if not notes:
            self.notes = "No notes provided."
        else:
            self.notes = notes
        
    def _create_meta(self):
        """ Generate and organize info into a matadata dictionary recognized by hyperspy
        :returns: nested dictionary of information """


        metadata = {}
        metadata['General'] = {}

        metadata['General']['name'] = self.name
        metadata['General']['title'] = self.title
        metadata['General']['authors'] = self.authors
        metadata['General']['notes'] = self.notes

        metadata['Signal'] = {}
        metadata['Signal']['binned'] = True
        metadata['Signal']['signal_type'] = None

        metadata['Sample'] = {}
        metadata['Sample']['elements'] = self.crystal.get_atom_numbers()


        metadata['Sample']['system'] = {}
        metadata['Sample']['system']['cell'] = {}
        axes = ['a','b','c']
        for i in range(len(self.crystal.lattice)):
            metadata['Sample']['system']['cell'][axes[i]] = self.crystal.lattice[i]

        metadata['Sample']['system']['fermienergy'] = self.fermienergy
        metadata['Sample']['system']['temperature'] = self.temperature

        metadata['Sample']['system']['model'] = self.crystal.brillouinzone.band_model
        metadata['Sample']['system']['bands'] = {}
        metadata['Sample']['system']['bands']['count'] = len(self.crystal.brillouinzone.bands)
        for i, band in enumerate(self.crystal.brillouinzone.bands):
            metadata['Sample']['system']['bands']["band {}".format(i)] = self.crystal.brillouinzone.bands[i]
                
        metadata['Sample']['description'] = None

        return metadata


    def _create_signal(self, data, eBin):
        """Organize and convert data and axes into a hyperspy signal 
        :param data: the resulting array from simulation
        :param eBin: the energy binning used in simulation
        :returns: hyperspy signal 
        """

        metadata = self._create_meta()

        s = hs.signals.BaseSignal(data, metadata=metadata)


        names = ["Energy", "q_x", "q_y", "q_z"]
        for i in range(len(data.shape)-1):
            name = names[i+1]
            s.axes_manager[2-i].name = name
            s.axes_manager[name].scale = 1.0/(self.crystal.brillouinzone.mesh[i]-1) #self.crystal.brillouinzone.lattice[i]
            s.axes_manager[name].units = "AA-1"
            s.axes_manager[name].offset = -0.5#-self.crystal.brillouinzone.lattice[i]/2
        i += 1
        name = names[0]
        s.axes_manager[i].name = name
        s.axes_manager[name].scale = eBin[1]-eBin[0]
        s.axes_manager[name].units = "eV"
        s.axes_manager[name].offset = eBin[0]
        #s.metadata.Signal.binned = True


        p = s.as_signal1D(-1)
        return p

    
    def calculate_eels(self, energyBins, fermienergy=None, temperature=None):
        """ Calculate the momentum dependent scattering cross section of the system,
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 
        
        :type  fermienergy: float
        :param fermienergy: a spesific fermienergy in eV, if not defined the standard fermienergy is used
        
        :type  temperature: float
        :param temperature: a spesific temperature in Kelvin, if not defined the standard temperature is used
        """

        if temperature:
            self.temperature = temperature
        if fermienergy:
            self.fermienergy = fermienergy

        

        energyBands = []
        waveStates = []
        for band in self.bands:
            energyBands.append(band.energies)
            waveStates.append(band.waves)

        data = calculate_spectrum(
                self.crystal.brillouinzone.mesh,
                self.crystal.brillouinzone.lattice, 
                initial_band.k_grid,
                np.stack(energyBands, axis=1),  
                np.stack(waveStates, axis=1).real, 
                np.stack(waveStates, axis=1).imag,
                self.energyBins, 
                self.fermienergy, 
                self.temperature
            )       

        return self._create_signal(data, energyBins)
        
    def compress_signals(self, signals):
        """ Takes a list of spectra and adds them togheter to one spectrum
        
        :type  signals: list
        :param signals: List of hyperspy signal or ndarray

        :returns: The singals added into one signal
        """

        signal_total = signals[0]
        for signal in signals[1:]:
            signal_total = np.add(signal_total,signal)

        return signal_total



    def calculate_eels_multiproc(self, energyBins, bands=(None,None), fermienergy=None, temperature=None, max_cpu=None, compact=True):
        """ Calculate the momentum dependent scattering cross section of the system, using multiple processes
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 
        
        :type  bands: tuple
        :param bands: Tuple of start index and end index of the bands included from the band structure

        :type  fermienergy: float
        :param fermienergy: a spesific fermienergy in eV, if not defined the standard fermienergy is used
        
        :type  temperature: float
        :param temperature: a spesific temperature in Kelvin, if not defined the standard temperature is used

        :type  max_cpu: int
        :param max_cpu: The user defined maximum allowed number of CPU-cores, if not, the hardware limit is used. 

        :type  compact: bool
        :param compact: If set True, all transitions are added to one spectrum. If False a list of spectra is returned with transitions from and to individual bands. 
        
        :returns: An individual hyperspy spectrum or list of spectra, see :param: compact for info
        """

        if not compact:
            raise NotImplementedError("EELS for individual bands are not implemented.")

        dielectrics = self.calculate_dielectric_multiproc(energyBins, bands, fermienergy, temperature, max_cpu, compact=True)

        q_squared = calculate_momentum_squared(
                self.crystal.brillouinzone.mesh,
                self.crystal.brillouinzone.lattice,
                self.energyBins
                )

        q_squared[q_squared[:] == 0] = np.nan

        if (dielectrics.shape == q_squared.shape):
            if compact:
                signal_total = np.nan_to_num(dielectrics/q_squared)
                return self._create_signal(signal_total, energyBins)
        else:
            raise ValueError("The shapes of dielectric function and q squared mismatch, try restart kernel.")
        """
        else:
            original_title = self.title
            for i, dielectric in enumerate(dielectrics):
                self.title = original_title+" from band {} to {}".format(transitions[i][0],transitions[i][1])
                dielectrics[i] = self._create_signal(np.nan_to_num(dielectric/q_squared), energyBins)
            self.title = original_title

            return signals
            """

    def calculate_dielectric_multiproc(self, energyBins, bands=(None,None), fermienergy=None, temperature=None, max_cpu=None, compact=True):
        """ Calculate the momentum dependent dielectric function of the system, using multiple processes
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 
        
        :type  bands: tuple
        :param bands: Tuple of start index and end index of the bands included from the band structure

        :type  fermienergy: float
        :param fermienergy: a spesific fermienergy in eV, if not defined the standard fermienergy is used
        
        :type  temperature: float
        :param temperature: a spesific temperature in Kelvin, if not defined the standard temperature is used

        :type  max_cpu: int
        :param max_cpu: The user defined maximum allowed number of CPU-cores, if not, the hardware limit is used. 

        :type  compact: bool
        :param compact: If set True, all transitions are added to one array. If False a list of arrays is returned with transitions from and to individual bands. 
        
        :returns: An individual numpy ndarray or list of arrays, see :param: compact for info
        """


        polarizations = self.calculate_polarization_multiproc(energyBins, bands, fermienergy, temperature, max_cpu, compact)

        q_squared = calculate_momentum_squared(
                self.crystal.brillouinzone.mesh,
                self.crystal.brillouinzone.lattice,
                self.energyBins
                )

        q_squared[q_squared[:] == 0] = np.nan

        if compact:
            dielectric = np.nan_to_num(polarizations/q_squared) #(polarizations + polarizations**2/q_squared)/q_squared  ###### SHOULD I TREAT IMAGINARY PARTS?
            return dielectric
        else:
            raise NotImplementedError("Approximative dielectric function for indidual bands is not implemented.")
            """
            polarization_compact = polarizations[0]
            for polarization in polarizations[1:]:
                polarization_compact += polarization

            original_title = self.title
            for i, polarization in enumerate(polarizations):
                self.title = original_title+" from band {} to {}".format(transitions[i][0],transitions[i][1])
                signals[i] = self._create_signal(np.nan_to_num(signal/q_squared), energyBins)
            self.title = original_title

            return signals
            """
    def calculate_polarization_multiproc(self, energyBins, bands=(None,None), fermienergy=None, temperature=None, max_cpu=None, compact=True):
        """ Calculate the momentum dependent polarization matrix of the system, using multiple processes
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 
        
        :type  bands: tuple
        :param bands: Tuple of start index and end index of the bands included from the band structure

        :type  fermienergy: float
        :param fermienergy: a spesific fermienergy in eV, if not defined the standard fermienergy is used
        
        :type  temperature: float
        :param temperature: a spesific temperature in Kelvin, if not defined the standard temperature is used

        :type  max_cpu: int
        :param max_cpu: The user defined maximum allowed number of CPU-cores, if not, the hardware limit is used. 

        :type  compact: bool
        :param compact: If set True, all transitions are added to one array. If False a list of arrays is returned with transitions from and to individual bands. 
        
        :returns: An individual numpy ndarray or list of arrays, see :param: compact for info
        """

        if temperature:
            self.temperature = temperature
        if fermienergy:
            self.fermienergy = fermienergy

        self.energyBins = energyBins

        energybands = self.crystal.brillouinzone.bands[bands[0]:bands[1]]

        transitions = []
        for i, initial in enumerate(energybands):
            for f, final in enumerate(energybands[(i+1):]):
                f += i+1

                # Check if bands lay below or above fermi distribution interval. 
                # Interval is estimated to temperature/500, this corresponds to approx. 10th digit accuracy
                if temperature != 0:
                    if initial.energy_min() < fermienergy-temperature/500 and final.energy_max() > fermienergy+temperature/500:
                        transitions.append((i,f, initial, final))
                else:
                    if initial.energy_min() < fermienergy and final.energy_max() > fermienergy:
                        transitions.append((i,f, initial, final))

        if not max_cpu:
            max_cpu = cpu_count()

        if max_cpu > cpu_count():
            max_cpu = cpu_count()

        p = Pool(min(len(transitions),max_cpu), self._init_worker)
        r = p.map_async(self._calculate, transitions)

        # Passing keyboard interuption to the c-extended processes pool
        try:
            r.wait()
        except KeyboardInterrupt:
            _logger.warning("Terminating process pool..")
            p.terminate()
            p.join()
            return None
        else:
            p.close()
            p.join()
            signals = r.get()

            if compact:
                signal_total = signals[0]
                for signal in signals[1:]:
                    signal_total += signal
                return signal_total
            else:
                return signals


                
    def _calculate(self, transition):
            i, f, initial_band, final_band = transition
            
            energyBands = [initial_band.energies, final_band.energies]
            waveStates =  [initial_band.waves, final_band.waves]

            return calculate_spectrum(
                self.crystal.brillouinzone.mesh,
                self.crystal.brillouinzone.lattice, 
                initial_band.k_grid,
                np.stack(energyBands, axis=1),  
                np.stack(waveStates, axis=1).real, 
                np.stack(waveStates, axis=1).imag,
                self.energyBins, 
                self.fermienergy, 
                self.temperature
            )       

    def _init_worker(self):
        sign.signal(sign.SIGINT, sign.SIG_IGN)
        

    # Signal handeling

    def gauss(self, sigma, eRange):
        """ Creates a gauss to smear data
        :type  sigma: float
        :param sigma: the sigmal value of the gauss

        :type  eRange: ndarray
        :param eRange: an array of energy values 

        :returns: an array with a gaussian in energy space
        """
        dE = eRange[1]-eRange[0]
        gx = np.arange(-3*sigma,3*sigma, dE)
        gaussian = np.exp(-0.5*(gx/sigma)**2)
        gaussian = gaussian/np.sum(gaussian)
        
        gauss =np.zeros((1,1,1,len(gaussian)))
        gauss[0,0,0,:] = gaussian
        return gauss

    def smear(self, s, sigma):
        """ Smear the signal with a gaussian smearing
        :type  s: hyperspy signal
        :param s: the signal to be smeared

        :type  sigma: float  
        :param sigma: the sigma value of the gauss
        
        :returns: the smeared signal
        """
        hist = s.data
        scale = s.axes_manager['Energy'].scale
        offset = s.axes_manager['Energy'].offset
        size = s.axes_manager['Energy'].size

        eRange = np.linspace(offset, offset+(size-1)*scale, size)

        gaussian = gauss(sigma, eRange)
        
        crop_front = len(gaussian[0,0,0,:])//2
        if len(gaussian[0,0,0,:])%2 == 1:
            crop_end = crop_front
        else:
            crop_end = crop_front-1
            
        hist = convolve(hist, gaussian)
        
        s_smooth = copy.deepcopy(s)
        
        s_smooth.data = hist[:,:,:,crop_front:-crop_end]
        s_smooth.metadata['General']['title']  = s.metadata['General']['title'] + " smoothed"
        s_smooth.metadata['General']['name']  = s.metadata['General']['name'] + " smoothed"
        return s_smooth



    def plotSignals(self, signals, colors=None, linestyles=None, plotstyle=None, fill=False, linewidth=None):
        
        s = signals[0]
        x_label = "{} [{}]".format(s.axes_manager.signal_axes[0].name,s.axes_manager.signal_axes[0].units)
        
        fig, ax = plt.subplots()
        ax.set_xlabel(x_label)
        ax.set_ylabel("Intensity [arb.]")
        
        if not linestyles:
            linestyles = []
            for i in range(len(signals)):
                linestyles.append('-')
        elif isinstance(linestyles,str):
            linestyles = [linestyles]

        if (len(linestyles) < len(signals)):
            for i in range(len(signals)-len(linestyles)):
                linestyles.append(linestyles[i])

        # REWRITE TO COLORS
        if not colors:
            standard_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
            colors = []
            for i in range(len(signals)):
                colors.append(standard_colors[2*i])
        elif isinstance(colors,str):
            colors = [colors]

        if (len(colors) < len(colors)):
            for i in range(len(signals)-len(colors)):
                colors.append(colors[i])

        if not linewidth:
            linewidth = 2;

        for i,s in enumerate(signals):
            x = s.axes_manager.signal_axes[0].axis
            y = s.sum().data
            label = s.metadata['General']['title']
            ax.plot(x,y,linestyles[i],color=colors[i], linewidth=linewidth, label=label)
            if fill:
                ax.fill_between(x,0,y,facecolor=colors[i],alpha=0.4)

                
        


    def __repr__(self):
        string = "EELS Signal Calculator:\n\nSignal name:\n\t{}\nAuthors:\n\t{}\nTitle:\n\t{}\nNotes:\n\t{}\n\n".format(self.name, self.authors, self.title, self.notes)
        string += "Temperature: {} K\tFermiEnergy: {} eV\n".format(self.temperature, self.fermienergy)
        return string

    