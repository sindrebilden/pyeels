import hyperspy.api as hs
from pyeels.cpyeels import calculate_spectrum, calculate_momentum_squared
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.signal import convolve, kaiser
from scipy.fftpack import hilbert

import copy
import os
import signal as sign
import time
import logging
_logger = logging.getLogger(__name__)

class EELS:
    _MC2 = 0.511e6 #eV
    _HBARC = 1973 #eVÅ
    _E_SQUARED = 0.09170123689*_HBARC

    temperature = 0
    fermienergy = 0
    
    def __init__(self, crystal, name=None):
        self.crystal = crystal
        self.max_cpu = 1

        self.set_incident_energy(60e3)
        

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
            self.bins = np.round(self.crystal.brillouinzone.mesh).astype(int)
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


    def set_incident_energy(self, incident_energy):
        self.incident_energy = incident_energy
        self.incident_k = self.incident_momentum()
        
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
            metadata['Sample']['system']['bands']["band {}".format(i)] = self.crystal.brillouinzone.bands[i].__repr__()
                
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


        names = ["Energy", "q_a", "q_b", "q_c"]
        units = ["eV", "a-1", "b-1", "c-1"]
        for i in range(len(data.shape)-1):
            name = names[i + 1]
            s.axes_manager[2 - i].name  = name
            s.axes_manager[name].scale  = 1.0 / (self.crystal.brillouinzone.mesh[i] - 1) #self.crystal.brillouinzone.lattice[i]
            s.axes_manager[name].units  = units[i + 1]
            s.axes_manager[name].offset = -0.5#-self.crystal.brillouinzone.lattice[i]/2
        i += 1
        name = names[0]
        s.axes_manager[i].name = name
        s.axes_manager[name].scale  = eBin[1] - eBin[0]
        s.axes_manager[name].units  = names[0]
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

    def joint_density_of_states(self, bands=(None,None), fermienergy=None, compact=True):
        """ Calculate the joint density of states for direct transitions
            
        :type  bands: tuple
        :param bands: Tuple of start index and end index of the bands included from the band structure

        :type  fermienergy: float
        :param fermienergy: a spesific fermienergy in eV, if not defined the standard fermienergy is used
        """

        raise NotImplementedError

        """
        DOS = np.zeros(self.crystal.brillouinzone.bands[0].energies.shape)

        for initial_band in self.crystal.brillouinzone.bands:
            for final_band in self.crystal.brillouinzone.bands:
                
                if initial_band != final_band:
                    DOS += final_band.energies-initial_band.energies
                    
        return DOS
        """

    def calculate_eels_multiproc(self, energyBins, incident_energy=None, bands=(None,None), fermienergy=None, temperature=None, background_permittivity=0, max_cpu=None, compact=True):
        """ Calculate the momentum dependent scattering cross section of the system, using multiple processes
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 

        :type  incident_energy: float
        :param incident_energy: The energy of the incident electrons
        
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
        
        :returns: An individual hyperspy spectrum or list of spectra, see :param: compact for info, returns None if terminated
        """



        if incident_energy:
            self.set_incident_energy(incident_energy)
        else:
            if not self.incident_energy:
                _logger.warning("No acceleration energy found, use set_incident_energy() for this. Using 60keV.")
                self.set_incident_energy(60e3)

        if not (isinstance(bands[0],type(None)) and isinstance(bands[1],type(None))):
            self.bands = bands

        if temperature:
            self.temperature = temperature

        if fermienergy:
            self.fermienergy = fermienergy

        if max_cpu:
            self.max_cpu = max_cpu

        if background_permittivity:
            self.background_permittivity = background_permittivity


        dielectric_imag = self.calculate_dielectric_imag_multiproc(energyBins=energyBins, compact=compact)

        energy_loss = []
        if not isinstance(dielectric_imag, type(None)):            

            if compact:
                energy_loss = self.calculate_energy_loss(dielectric_imag, energyBins)

            else:
                for i, sub_diel_imag in enumerate(dielectric_imag):
                    loss_function = self.calculate_energy_loss(sub_diel_imag, energyBins)
                    energy_loss.append(loss_function)  
        else:
            return None                          

        if not isinstance(energy_loss, type(None)):            
            weights = self.signal_weights()*(self._E_SQUARED*self._MC2)/(np.pi**2*self._HBARC**2*self.incident_k**3)

            if compact:
                if (energy_loss.shape == weights.shape):
                    signal_total = np.nan_to_num(energy_loss*weights)
                    return self._create_signal(signal_total, energyBins)
                else:
                    raise ValueError("The shapes of dielectric function and weights mismatch, try restart kernel.")
            else:
                    original_title = self.title
                    signals = []
                    for i, sub_energy_loss in enumerate(energy_loss):
                        if (sub_energy_loss.shape == weights.shape):
                            signal = np.nan_to_num(sub_energy_loss*weights)
                            self.title = original_title+" from band {}: {} to {}: {}".format(self._transitions[i][0], self._transitions[i][2], self._transitions[i][1], self._transitions[i][3])
                            signals.append(self._create_signal(signal, energyBins))                            
                        else:
                            raise ValueError("The shapes of dielectric function and weights mismatch, try restart kernel.")

                    self.title = original_title
                    return signals


        else:
            return None

    def incident_momentum(self):
        """ Calculates the relativistic incident momentum from an energy
        :type  incident_energy: float
        :param incident_energy: the incident energy
        :returns: the incident momentum
        """
        momentum = np.sqrt((self.incident_energy+self._MC2)**2-self._MC2**2)/self._HBARC

        return momentum
    def signal_weights(self):
        """ Calculates the signal weights (theta^2+theta_E^2)^-1 rising from the formulation of stopping power, by R. H. Ritchie (1957).
        
        :returns: singal weights in energy and momentum space 
        """

        # Calculate theta^2
        q_squared = calculate_momentum_squared(
        self.crystal.brillouinzone.mesh,
        self.crystal.brillouinzone.lattice,
        self.energyBins
        )/self.incident_k**2

        # Calculate theta_e^2
        e = (self.energyBins/(2*self.incident_energy))**2

        # Calculate (theta^2+theta_e^2)
        for i in range(0,q_squared.shape[1]):
            for j in range(0,q_squared.shape[2]):
                for k in range(0,q_squared.shape[3]):
                    q_squared[:,i,j,k] += e


        # Calculate (theta^2+theta_e^2)^-1
        q_squared[q_squared[:] == 0] = np.nan
        weights = q_squared**-1
        weights[np.isnan(weights)] = 0

        return weights

    def calculate_energy_loss(self, dielectric_imag, energyBins, background_permittivity=0):
        """ Calculate energy loss function for a spectrum image """

        axes_shape = dielectric_imag.shape[1:]
        energy_shape = dielectric_imag.shape[0]

        flat_axes = np.zeros(axes_shape).flatten().shape

        dielectric_imag = dielectric_imag.reshape((energy_shape,)+flat_axes)        
        energy_loss = np.zeros((energy_shape,)+flat_axes)

        for i, diel in enumerate(dielectric_imag.T):
            energy_loss[:,i] = self.energy_loss_function(diel, energy=energyBins, background_permittivity=background_permittivity)

        energy_loss = energy_loss.reshape((energy_shape,)+axes_shape)

        return energy_loss

    def calculate_dielectric_real(self, dielectric_imag, energyBins, beta=14, sigma=1, background_permittivity=0):
        """ Calculate real dielectric function for a spectrum image """

        axes_shape = dielectric_imag.shape[1:]
        energy_shape = dielectric_imag.shape[0]

        flat_axes = np.zeros(axes_shape).flatten().shape

        dielectric_imag = dielectric_imag.reshape((energy_shape,)+flat_axes)        
        diel_real = np.zeros((energy_shape,)+flat_axes)

        for i, diel_imag in enumerate(dielectric_imag.T):
            diel_real[:,i] = self.kk_imag_to_real(diel_imag, energy=energyBins, beta=beta, sigma=sigma, background_permittivity=background_permittivity)

        diel_real = diel_real.reshape((energy_shape,)+axes_shape)

        return diel_real


    def calculate_dielectric_imag_multiproc(self, energyBins, bands=(None,None), incident_energy=None, fermienergy=None, temperature=None, max_cpu=None, compact=True):
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



        if incident_energy:
            self.set_incident_energy(incident_energy)
        else:
            if not self.incident_energy:
                _logger.warning("No acceleration energy found, use set_incident_energy() for this. Using 60keV.")
                self.set_incident_energy(60e3)

        if not (isinstance(bands[0],type(None)) and isinstance(bands[1],type(None))):
            self.bands = bands
            
        if temperature:
            self.temperature = temperature

        if fermienergy:
            self.fermienergy = fermienergy

        if max_cpu:
            self.max_cpu = max_cpu


        polarizations = self.calculate_polarization_multiproc(energyBins=energyBins, compact=compact)

        if not isinstance(polarizations, type(None)):           
            weights = self.signal_weights()*(4*np.pi*self._E_SQUARED)/self.incident_k**2

            # Correct the weighting in the optical limit (Stephen L. Adler 1962)
            center = ((self.bins-1)/2).astype(int)
            weights[:,center[0], center[1], center[2]] = (4*np.pi*self._E_SQUARED*self._HBARC**4)/(self._MC2**2*self.energyBins**2)

            if compact:
                if (polarizations.shape == weights.shape):
                    signal_total = np.nan_to_num(polarizations*weights)
                    return signal_total
                else:
                    raise ValueError("The shapes of polarization and weights mismatch, try restart kernel.")
            else:
                    original_title = self.title
                    signals = []
                    for sub_polarizations in polarizations:
                        if (sub_polarizations.shape == weights.shape):
                            signals.append(np.nan_to_num(sub_polarizations*weights))
                        else:
                            raise ValueError("The shapes of polarization and weights mismatch, try restart kernel.")

                    return signals

            
        else:
            return None


    def calculate_polarization_multiproc(self, energyBins, bands=(None,None), incident_energy=None, fermienergy=None, temperature=None, max_cpu=None, compact=True):
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

        if incident_energy:
            self.set_incident_energy(incident_energy)
        else:
            if not self.incident_energy:
                _logger.warning("No acceleration energy found, use set_incident_energy() for this. Using 60keV.")
                self.set_incident_energy(60e3)

        if not (isinstance(bands[0],type(None)) and isinstance(bands[1],type(None))):
            self.bands = bands
            
        if temperature:
            self.temperature = temperature

        if fermienergy:
            self.fermienergy = fermienergy

        if max_cpu:
            self.max_cpu = max_cpu

        self.energyBins = energyBins


        print(self.bands)

        energybands = self.crystal.brillouinzone.bands[self.bands[0]:self.bands[1]]
        print(len(energybands))

        self._transitions = []
        for i, initial in enumerate(energybands):
            for f, final in enumerate(energybands[(i+1):]):
                f += i+1

                # Check if bands lay below or above fermi distribution interval. 
                # Interval is estimated to temperature/500, this corresponds to approx. 10th digit accuracy
                if self.temperature != 0:
                    print("T neq 0")
                    if initial.energy_min() < self.fermienergy-self.temperature/500 and final.energy_max() > self.fermienergy+self.temperature/500:
                        self._transitions.append((i, f, initial, final))
                else:
                    if initial.energy_min() < self.fermienergy and final.energy_max() > self.fermienergy:
                        self._transitions.append((i, f, initial, final))
        print(len(self._transitions))
        if not max_cpu:
            max_cpu = cpu_count()

        if max_cpu > cpu_count():
            max_cpu = cpu_count()

        p = Pool(min(len(self._transitions),max_cpu), self._init_worker)
        r = p.map_async(self._calculate, self._transitions)

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
        """ Calculates the irreducible polarization matrix by embeded c-code. 
        The result is weighted by the number of k-points in the Brillouin Zone.

        :type  transition: tuple
        :param transition: containing index of initial and final band and the respective band objects

        :returns: a weighted irreducible polarization matrix as a numpy ndarray
        """
        i, f, initial_band, final_band = transition
        
        energyBands = [initial_band.energies, final_band.energies]
        waveStates =  [initial_band.waves, final_band.waves]

        k_weights = self.crystal.brillouinzone.mesh[0]*self.crystal.brillouinzone.mesh[1]*self.crystal.brillouinzone.mesh[2];      

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
        )/k_weights

    def _init_worker(self):
        sign.signal(sign.SIGINT, sign.SIG_IGN)
        

    def kk_imag_to_real(self, imag, energy, beta=14, sigma=1, background_permittivity=0):
        """ Kramers Kronig transformation of the imaginary dielectric function to a real dielectric function
        :type  imag: numpy array
        :param imag: the imaginary dielectric function along an energy axis
        
        :type  energy: numpy array
        :param energy: the energy axis of the imaginary dielectric function
        
        :type  beta: int, float
        :param beta: a parameter of the attenuation window, 14 is standard suggested by the scipy package
        
        :type  sigma: float
        :param sigma: a parameter of the center attenuation (an inverse gauss), 1 is standard 
        
        :type  background_permittivity: float
        :param background_permittivity: a correction to the calculation of the real dielectric function, usually for ferroelectric materials
        
        :returns: the real dielectric function along the energy axis
        """
        
        # Extend energy axis to zero if needed
        dE = (energy[1]-energy[0])
        n = int(energy[0]/dE)

        low_energy = np.linspace(energy[0]-dE*n,energy[0]-dE,n)
        energy = np.hstack([low_energy,energy])
        imag = np.hstack([np.zeros((n,)),imag])

        half_index = imag.shape[0]

        
        imag = np.hstack([-np.array(list(reversed(imag))),imag])
        energy = np.hstack([-np.array(list(reversed(energy))),energy])
        
        inverse_gauss = 1-np.exp(-energy**2/sigma)
        
        window = kaiser(imag.shape[0],beta)
        
        return hilbert(imag*window*inverse_gauss)[half_index+n:]+background_permittivity

    def energy_loss_function(self, imag, real=None, energy=None, background_permittivity=0):
        """ Calculated the full energy loss function Im[-eps**-1] from the imaginary dielectric funtion
        :type  imag: numpy array
        :param imag: the imaginary dielectric function along an energy axis
        
        :type  real: numpy array
        :param real: the real dielectric function along an energy axis, if None it will be computed

        :type  energy: numpy array
        :param energy: the energy axis of the imaginary dielectric function

        :type  background_permittivity: float
        :param background_permittivity: the background permittivity for real dielectrid function
        
        :returns: the energy loss function along the enery axis
        """

        if isinstance(real, type(None)):
            if not isinstance(energy, type(None)):        
                real = self.kk_imag_to_real(imag, energy, background_permittivity=background_permittivity)
            else:
                _logger.warning("Neither real or energy are given, cannot compute full energy loss function.")


        return imag/(real**2+imag**2)


    # Signal handeling
    @classmethod
    def _gauss(cls, sigma, eRange):
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

    @classmethod
    def _thermal(cls, sigma, eRange):
        """ Creates a gauss to smear data
        :type  sigma: float
        :param sigma: the sigmal value of the gauss

        :type  eRange: ndarray
        :param eRange: an array of energy values 

        :returns: an array with a gaussian in energy space
        """
        dE = eRange[1]-eRange[0]
        tx = np.arange(-31*sigma,31*sigma, dE) #Limit for 1e-3 treshold
        thermal = np.imag(1/(tx-1j*sigma))
        thermal = thermal/np.sum(thermal)
        
        therm =np.zeros((1,1,1,len(thermal)))
        therm[0,0,0,:] = thermal
        return therm

    @classmethod
    def gaussian_smear(cls, s, sigma):
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

        gaussian = cls._gauss(sigma, eRange)
        
        crop_front = len(gaussian[0,0,0,:])//2
        if len(gaussian[0,0,0,:])%2 == 1:
            crop_end = crop_front
        else:
            crop_end = crop_front-1
            
        if len(hist.shape) == 1:
            hist = convolve(hist, gaussian[0,0,0,:])
        elif len(hist.shape) == 2:
            hist = convolve(hist, gaussian[0,0,:,:])
        elif len(hist.shape) == 3:
            hist = convolve(hist, gaussian[0,:,:,:])
        else:
            hist = convolve(hist, gaussian)
    

        s_smooth = copy.deepcopy(s)
        
        s_smooth.data = hist[...,crop_front:-crop_end]

        s_smooth.metadata['General']['title']  = s.metadata['General']['title'] + " gaussian smearing s={}".format(sigma)
        s_smooth.metadata['General']['name']  = s.metadata['General']['name'] + " gaussian smearing s={}".format(sigma)
        return s_smooth

    @classmethod
    def thermal_smear(cls, s, sigma):
        """ Smear the signal with a thermal smearing Im[1/(dE-i*sigma)]
        :type  s: hyperspy signal
        :param s: the signal to be smeared

        :type  sigma: float  
        :param sigma: the sigma value of the imaginary energy
        
        :returns: the smeared signal
        """
        hist = s.data
        scale = s.axes_manager['Energy'].scale
        offset = s.axes_manager['Energy'].offset
        size = s.axes_manager['Energy'].size

        eRange = np.linspace(offset, offset+(size-1)*scale, size)

        thermal = cls._thermal(sigma, eRange)
        
        crop_front = len(thermal[0,0,0,:])//2

        if len(thermal[0,0,0,:])%2 == 1:
            crop_end = crop_front
        else:
            crop_end = crop_front-1
        
        if len(hist.shape) == 1:
            hist = convolve(hist, thermal[0,0,0,:])
        elif len(hist.shape) == 2:
            hist = convolve(hist, thermal[0,0,:,:])
        elif len(hist.shape) == 3:
            hist = convolve(hist, thermal[0,:,:,:])
        else:
            hist = convolve(hist, thermal)
    
    

        s_smooth = copy.deepcopy(s)
        
        s_smooth.data = hist[...,crop_front:-crop_end]

        s_smooth.metadata['General']['title']  = s.metadata['General']['title'] + " thermal smearing s={}".format(sigma)
        s_smooth.metadata['General']['name']  = s.metadata['General']['name'] + " thermal smearing s={}".format(sigma)
        return s_smooth

    @classmethod
    def set_ROI(cls, s, shape="circle", interactive=False):
        """ Selects an interactive region of interst (ROI) to the signal

        :type  s: hyperspy signal
        :param s: the signal of interest

        :type  shape: string
        :param shape: the description of the ROI; circle, ring, rectangle

        :type  interactive: boolean
        :param interactive: interactive if True, False if left blank
        
        :returns: hyperspy roi, hyperspy signal
        """
        import hyperspy.api as hs
     
        if s.axes_manager.navigation_dimension < 2:
            axes = "sig"
            x_axis = s.axes_manager[s.axes_manager.signal_indices_in_array[1]]
            y_axis = s.axes_manager[s.axes_manager.signal_indices_in_array[0]]
        else:
            axes = "nav"
            x_axis = s.axes_manager[s.axes_manager.navigation_indices_in_array[1]]
            y_axis = s.axes_manager[s.axes_manager.navigation_indices_in_array[0]]


        if shape == "circle":
            x = x_axis.axis[round(x_axis.size/2)]
            y = y_axis.axis[round(y_axis.size/2)]

            r_outer = x_axis.axis[round(3*x_axis.size/4)]
        
            sroi = hs.roi.CircleROI(x, y, r=r_outer)
            """
            s.plot()
            sroi= sroi.interactive(s) 
            ss = hs.interactive(f=sroi.sum, event=sroi.events.data_changed)
            """
        elif shape == "ring":
            x = x_axis.axis[round(x_axis.size/2)]
            y = y_axis.axis[round(y_axis.size/2)]

            r_outer = x_axis.axis[round(4*x_axis.size/5)]
            r_inner = x_axis.axis[round(3*x_axis.size/4)]
        
            sroi = hs.roi.CircleROI(x, y, r=r_outer, r_inner=r_inner)
            """
            s.plot()
            sroi= sroi.interactive(s) 
            ss = hs.interactive(f=sroi.sum, event=sroi.events.data_changed)
            """
        else:
            if not shape == "rectangle":
                print("Did not recognize shape, using rectangle")
            x1 = x_axis.axis[1]
            x2 = x_axis.axis[round(x_axis.size/10)]
            y1 = y_axis.axis[1]
            y2 = y_axis.axis[round(y_axis.size/10)]

            sroi = hs.roi.RectangularROI(x1, y1, x2, y2)
            
        if interactive:
            s.plot()
            roi_signal = sroi.interactive(s)
            ss = hs.interactive(f=roi_signal.sum, event=roi_signal.events.data_changed) 
        else:
            roi_signal = sroi(s)
            ss = roi_signal.sum()
            
        return sroi, ss



    def __repr__(self):
        string = "EELS Signal Calculator:\n\nSignal name:\n\t{}\nAuthors:\n\t{}\nTitle:\n\t{}\nNotes:\n\t{}\n\n".format(self.name, self.authors, self.title, self.notes)
        string += "Temperature: {} K\tFermiEnergy: {} eV\n".format(self.temperature, self.fermienergy)
        return string

    