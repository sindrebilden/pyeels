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
    _E_SQUARED = 0.09170123689*_HBARC #eVÅ

    temperature = 0
    fermienergy = 0
    
    def __init__(self, crystal, name=None):
        self.crystal = crystal
        self.max_cpu = 1
        self.bands = (None, None)


        # A default smearing (eta) is set to 0.1 of the energy spacing
        self.smearing = 0.1 
        self.refractive_index = None

        self.set_incident_energy(60e3)
        
        self.polarization = None
        self.dielectric = None

        self.operator = np.eye(self.crystal.brillouinzone.bands[0].waves.shape[1], dtype=np.complex)

        self.valence_electrons = 1


    def set_operator(self, operator):

        if operator.shape != self.operator.shape:
            raise ValueError("Shape of operator must match the wave components, i.e. the shape {}".format(self.operator.shape))

        self.operator = operator.astype(np.complex)

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

    def calculate_eels_multiproc(self, energyBins, incident_energy=None, bands=(None,None), fermienergy=None, temperature=None, max_cpu=None, smearing=None, refractive_index=None, compact=True):
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

        :type  smearing: float
        :param smearing: a smearing factor in creating the dielectric function

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

        if smearing:
            self.smearing = smearing

        if refractive_index:
            self.refractive_index = refractive_index

        if max_cpu:
            self.max_cpu = max_cpu


        if isinstance(self.dielectric, type(None)):
            dielectrics = self.calculate_dielectric_multiproc(energyBins=energyBins, compact=compact)
        else:
            dielectrics = self.dielectric
            _logger.warning("Found stored dielectric, using this")

            if energyBins.shape[0] == dielectrics.shape[0]:
                self.energyBins = energyBins
            else:
                raise ValueError("The energy bins does not match the precalculated dielectric.")



        energy_loss = []
        if not isinstance(dielectrics, type(None)):            

            if compact:
                energy_loss = EELS.calculate_energy_loss_function(dielectrics)

            else:
                for i, dielectric in enumerate(dielectrics):
                    energy_loss.append(self.calculate_energy_loss_function(dielectric))

        else:
            return None                          

        if not isinstance(energy_loss, type(None)):            
            weights = self.signal_weights()*(self._E_SQUARED*self._MC2)/(np.pi**2*self._HBARC**2*self.incident_k**3)

            # Correct the weighting in the optical limit (Stephen L. Adler 1962)
            center = ((self.bins-1)/2).astype(int)
            weights[:,center[0], center[1], center[2]] = np.zeros(weights[:,center[0], center[1], center[2]].shape)

            if compact:
                if (energy_loss.shape == weights.shape):
                    signal_total = np.nan_to_num(energy_loss*weights)

                    signal_total = self._create_signal(signal_total, self.energyBins)

                    return signal_total
                else:
                    raise ValueError("The shapes of dielectric function and weights mismatch, try restart kernel.")
            else:
                    original_title = self.title
                    signals = []
                    for i, sub_energy_loss in enumerate(energy_loss):
                        if (sub_energy_loss.shape == weights.shape):
                            signal = np.nan_to_num(sub_energy_loss*weights)
                            self.title = original_title+" from band {}: {} to {}: {}".format(self._transitions[i][0], self._transitions[i][2], self._transitions[i][1], self._transitions[i][3])
                            signals.append(self._create_signal(signal, self.energyBins))                            
                        else:
                            raise ValueError("The shapes of dielectric function and weights mismatch, try restart kernel.")

                    self.title = original_title

                    return signals


        else:
            return None

    def calculate_dielectric_multiproc(self, energyBins, bands=(None,None), incident_energy=None, fermienergy=None, temperature=None, max_cpu=None, smearing=None, refractive_index=None, plasmon_energy=None, compact=True):
        """ Calculate the complex momentum dependent dielectric function of the system, using multiple processes
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 
        
        :type  bands: tuple
        :param bands: Tuple of start index and end index of the bands included from the band structure

        :type  fermienergy: float
        :param fermienergy: a spesific fermienergy in eV, if not defined the standard fermienergy is used
        
        :type  temperature: float
        :param temperature: a spesific temperature in Kelvin, if not defined the standard temperature is used

        :type  smearing: float
        :param smearing: a smearing factor in creating the dielectric function

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

        if smearing:
            self.smearing = smearing

        if refractive_index:
            self.refractive_index = refractive_index

        if max_cpu:
            self.max_cpu = max_cpu






        if isinstance(self.polarization, type(None)):
            polarizations = self.calculate_polarization_multiproc(energyBins=energyBins, compact=compact)
        else:
            polarizations = self.polarization

            _logger.warning("Found stored polarization, using this")


            if energyBins.shape[0] == polarizations.shape[0]:
                self.energyBins = energyBins
            else:
                raise ValueError("The energy bins does not match the precalculated polarization.")



        if not isinstance(polarizations, type(None)): 

            weights = self.signal_weights()*(4*np.pi*self._E_SQUARED)/(self.incident_k**2*self.crystal.volume)

            # Correct the weighting in the optical limit (Stephen L. Adler 1962)
            center = ((self.bins-1)/2).astype(int)
            
            temp_energy = copy.deepcopy(self.energyBins)
            temp_energy[temp_energy[:] == 0 ] = np.nan

            weights[:,center[0], center[1], center[2]] = (4*np.pi*self._E_SQUARED*self._HBARC**4)/(self._MC2**2*temp_energy**2*self.crystal.volume)
            weights = np.nan_to_num(weights)

            del(temp_energy)

            if compact:
                dielectric_real = EELS.smear_data(data=polarizations, energy=self.energyBins, sigma=self.smearing, type='Real')
                dielectric_real = self._trim_edges(dielectric_real)

                dielectric_imag = EELS.smear_data(data=polarizations, energy=self.energyBins, sigma=self.smearing, type='Imaginary')
                dielectric_imag = self._trim_edges(dielectric_imag)

                del(polarizations)


                # Pack data
                dielectric = ( (dielectric_real + 1j * dielectric_imag) )
                del(dielectric_imag, dielectric_real)

                if (dielectric.shape == weights.shape):
                    dielectric = np.nan_to_num(dielectric*weights)
                else:
                    raise ValueError("The shapes of polarization and weights mismatch, try to restart the kernel.")

                dielectric = 1 + dielectric.real + 1j * dielectric.imag
                if self.refractive_index:
                    dielectric = self.normalize_dielectric_by_refractive_index(dielectric=dielectric, refractive_index=self.refractive_index)
                elif plasmon_energy:
                    dielectric = self.normalize_dielectric_by_valence(dielectric=dielectric, valence_electrons=self.valence_electrons, plasmon_energy=plasmon_energy)
                self.dielectric = dielectric
                return dielectric

            else:
                dielectrics = []

                for i, polarization in enumerate(polarizations):
                    dielectric_real = EELS.smear_data(data=polarization, energy=self.energyBins, sigma=self.smearing, type='Real')
                    dielectric_real = self._trim_edges(dielectric_real)

                    dielectric_imag = EELS.smear_data(data=polarization, energy=self.energyBins, sigma=self.smearing, type='Imaginary')+1
                    dielectric_imag = self._trim_edges(dielectric_imag)

                    del(polarization)

                    dielectric = ( (dielectric_real + 1j * dielectric_imag) )

                    del(dielectric_imag, dielectric_real)
                    if (dielectric.shape == weights.shape):
                        dielectrics.append(np.nan_to_num(dielectric*weights))

                        del(dielectric)

                    else:
                        raise ValueError("The shapes of polarization and weights mismatch, try restart kernel.")

                    dielectric = ( (1 + dielectric.real + 1j * dielectric.imag) )
                    if self.refractive_index:
                        dielectric = self.normalize_dielectric_by_refractive_index(dielectric=dielectric, refractive_index=self.refractive_index)
                    elif plasmon_energy:
                        dielectric = self.normalize_dielectric_by_valence(dielectric=dielectric, valence_electrons=self.valence_electrons, plasmon_energy=plasmon_energy)
                _logger.warning("Calculation of true dielectric function for individual bands is not correctly implemented. This is a temporary solution.")

                self.dielectric = dielectrics
                return dielectrics

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

        energybands = self.crystal.brillouinzone.bands[self.bands[0]:self.bands[1]]

        self._transitions = []



        self.valence_electrons = 0
        for i, initial in enumerate(energybands):
            for f, final in enumerate(energybands[(i+1):]):
                f += i+1

                # Check if bands lay below or above fermi distribution interval. 
                # Interval is estimated to temperature/500, this corresponds to approx. 10th digit accuracy
                if self.temperature != 0:
                    if initial.energy_min() < self.fermienergy-self.temperature/500 and final.energy_max() > self.fermienergy+self.temperature/500:
                        self._transitions.append((i, f, initial, final))
                else:
                    if initial.energy_min() < self.fermienergy and final.energy_max() > self.fermienergy:
                        self._transitions.append((i, f, initial, final))

            #Sum up valence electrons for normalization purpuse in energy loss function
            if self.temperature != 0:
                if initial.energy_min() < self.fermienergy-self.temperature/500:
                    self.valence_electrons +=2
            else:
                if initial.energy_min() < self.fermienergy:
                    self.valence_electrons +=2
                        
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

                self.polarization = signal_total
                return signal_total
            else:
                self.polarization = signal
                return signals

    def _trim_edges(self, data):
        """ Set the edges of the spatial data structure to np.nan 

        :type  data: np.ndarray
        :param data: the data to be trimmed

        :returns: data with np.nan along every spatial edges
        """
        
        transposed = False
        if data.shape[0] > data.shape[-1]:
            transposed = True
            data = data.T

        data = np.pad(data[1:-1,1:-1,1:-1],(1), mode='constant', constant_values=np.nan)[:,:,:,1:-1]

        if transposed:
            return data.T
        else:
            return data

                
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
            self.operator.real, 
            self.operator.imag,
            self.energyBins, 
            self.fermienergy, 
            self.temperature
        )/k_weights

    def _init_worker(self):
        sign.signal(sign.SIGINT, sign.SIG_IGN)
    
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
        
        return np.nan_to_num(weights)


    def normalize_dielectric_by_refractive_index(self, dielectric, refractive_index=1):
        """ Normalize the dielectric function eps(q,w) to a known refractive index 

        :type  dielectric: np.ndarray
        :param dielectric: the raw dielectric function  eps(q,w)

        :type  refractive_index: 
        :param refractive_index: 

        :returns: a normalized dielectric function where eps(0,0) = refractive_index^2
        """

        transposed = False
        if not dielectric.shape[-1] == self.energyBins.shape[0]:
            dielectric = dielectric.T
            transposed = True

        if dielectric.shape[-1] == self.energyBins.shape[0]:
            dielectric_imag = dielectric.imag
            dielectric_real = dielectric.real-1
            dielectric = dielectric_real + 1j*dielectric_imag

            center_index = np.hstack([(np.asarray(dielectric_real.shape[:-1]).astype(int)-1)/2,np.zeros(1)]).astype(int)
            center_index[0] += 1
            middle = dielectric_real.item(tuple(center_index))

            if middle < 0:
                _logger.warning("Static dielectric function is below 1, this unphysical")
                return None

            scale = 1
            if middle != 0 and refractive_index > 1:
                scale = (refractive_index**2-1)/(middle)
            elif middle == 0 and refractive_index == 1:
                scale = 1
            elif refractive_index < 1:
                raise ValueError("Cannot handle refractive index below 1")
            elif middle == 0:
                _logger.warning('Real static polarizability is zero which makes it impossible to rescale. Returning input.')
            else:
                raise NotImplementedError("Cannot handle this case of refractive index.")

            print(scale)

            dielectric = dielectric*scale
            dielectric = 1+dielectric.real + 1j*dielectric.imag

            if transposed:
                return dielectric.T
            else:
                return dielectric
        else:
            raise ValueError("The dielectric matrix does not match the energy dimension")

    def normalize_dielectric_by_valence(self, dielectric, valence_electrons=None, plasmon_energy=None):
        """ Normalize the dielectric function to the valence electron density 

        :type  dielectric: np.ndarray
        :param dielectric: the loss function to be normalized

        :type  valence_electrons: float
        :param valence_electrons: the number of valence electrons

        :type  plasmon_energy: float
        :param plasmon_energy: the plasmon energy if known

        :returns: the normalized loss function
        """

        if not valence_electrons:
            valence_electrons = self.valence_electrons
        
        transposed = False
        if not dielectric.shape[-1] == self.energyBins.shape[0]:
            dielectric = dielectric.T
            transposed = True

        if dielectric.shape[-1] == self.energyBins.shape[0]:
#            energy_step = self.energyBins[1]-self.energyBins[0]

            dielectric_imag = dielectric.imag
            dielectric_real = dielectric.real

            # rewrite to integral from scipy?
            integrated = np.trapz(y=dielectric_imag*self.energyBins, x=self.energyBins, axis=-1)

            print("integral is {}".format(integrated))

            integrated[integrated[:]==0] = np.nan


            if plasmon_energy:
                plasmon_energy_squared = plasmon_energy**2
            else:
                plasmon_energy_squared = (4*np.pi*self._E_SQUARED*self._HBARC**2*valence_electrons)/(self._MC2*self.crystal.volume)

            scale = np.pi*plasmon_energy/(2*integrated)

            scale = np.nan_to_num(scale)

            print("scale is {}".format(scale))

            dielectric = (((dielectric_real.T-1)*scale)+1 + 1j*dielectric_imag.T*scale).T

            if transposed:
                return dielectric.T
            else:
                return dielectric
        else:
            raise ValueError("The shape of the dielectric function does not match self.energyBins")

    def map_onset(self, data):
        """ Create an onset map from the given n-dimensional data, the last axis must represent energy

        :type data: np.ndarray
        :data data: the data to map the onsets from

        :returns: an onset map of given the data
        """

        energy = self.energyBins

        free_onsets = np.ones(data.shape[:-1])
        
        onsets = np.zeros(data.shape[:-1])
        
        
        for i in range(data.shape[-1]):
            above = (data[...,i]>0)
            onsets += above*energy[i]*free_onsets
            
            free_onsets *= 1-above
        
        onsets[onsets[:]==0] = np.nan
        
        return onsets

    def mask_data_to_polarization_onset(self, data):
        """ Apply a onset mask setting all values below the onset of the Polarization 

        :type  data: np.ndarray
        :param data: the data to be masked, must have same shape as self.polarization

        :returns: masked data
        """

        if (data.shape != self.polarization.shape) and (data.T.shape != self.polarization.shape):
            raise ValueError("The data must have same shape as self.polarization.")      

        transposed = False
        if data.shape[0] == self.energyBins.shape[0]:
            transposed = True
            data = data.T

        if data.shape[-1] == self.energyBins.shape[0]:

            onset = self.map_onset(self.polarization.T)

            energy = np.zeros(data.shape)
            energy[...,:] = self.energyBins
            mask = (energy[:,...].T>onset).T

            data = data*mask

            if transposed:
                return data.T
            else:
                return data
        else:
            raise ValueError("The shape of data does not match self.energyBins but matches self.polarization, have you changed the bins?")



    

    @classmethod
    def calculate_energy_loss_function(cls, dielectric):
        """ Calculate energy loss function for a complex spectrum image 
        
        :type  dielectric: np.ndarray
        :param dielectric: the complex dielectric matrix 

        :returns: the energy loss matrix
        """

        return dielectric.imag/(dielectric.real**2+dielectric.imag**2)




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

        gx = np.arange(-eRange[-1],eRange[-1], dE)
        gaussian = np.exp(-0.5*(gx/sigma)**2)
        gaussian = gaussian/gaussian.sum()
        
        gauss =np.zeros((1,1,1,len(gaussian)))
        gauss[0,0,0,:] = gaussian
        return gauss

    @classmethod
    def _imaginary(cls, sigma, eRange):
        """ Creates an weight function to the imaginary part of dielectric function to smear data
        :type  sigma: float
        :param sigma: the sigmal value of the weight function

        :type  eRange: ndarray
        :param eRange: an array of energy values 

        :returns: an array with an imaginary part weight function in energy space
        """
        dE = eRange[1]-eRange[0]

        tx = np.arange(-eRange[-1],eRange[-1], dE) #-50*sigma,50*sigma
        weights = 1/(-tx-1j*sigma)
        imaginary = weights.imag/(np.abs(weights.imag).sum()*dE)
        
        imag =np.zeros((1,1,1,len(imaginary)))
        imag[0,0,0,:] = imaginary
        return imag

    @classmethod
    def _real(cls, sigma, eRange):
        """ Creates an weight function to the real part of dielectric function to smear data
        :type  sigma: float
        :param sigma: the sigmal value of the weight function

        :type  eRange: ndarray
        :param eRange: an array of energy values 

        :returns: an array with a real part weight function in energy space
        """
        dE = eRange[1]-eRange[0]
    
        tx = np.arange(-eRange[-1],eRange[-1], dE) #-50*sigma,50*sigma
        weights = 1/(-tx-1j*sigma)
        real_temp = weights.real/(np.abs(weights.imag).sum()*dE)
        
        real =np.zeros((1,1,1,len(real_temp)))
        real[0,0,0,:] = real_temp
        return real

    @classmethod
    def smear_data(cls, data, energy, sigma, type='Gaussian'):
        """ Smear the signal with a smearing of chosen type
        :type  data: np.ndarray
        :param data: the data to be smeared with the energy axis as the first axis

        :type  energy: np.ndarray
        :param energy: the energy axis of the data
        
        :type  sigma: float  
        :param sigma: the sigma value of the gauss

        :type  type: string
        :param type: Keyword for type of smeraing, Gaussian, Imaginary, or Real
        
        :returns: the smeared signal
        """
        transposed = False
        if not data.shape[-1] == energy.shape[0]:
            transposed = True
            data = data.T

        if not data.shape[-1] == energy.shape[0]:
            _logger.warning("Last or first axis must match the energy axis")
            return None
            
        #Determine the type of smearing
        if (type == 'Gaussian') or (type == 'G') or (type == 'Gauss')  or (type == 0):
            smearing = cls._gauss(sigma, energy)
            
        elif (type == 'Imaginary') or (type == 'I') or (type == 'Imag') or (type == 'Im')  or (type == 1):
            smearing = cls._imaginary(sigma, energy)

        elif (type == 'Real') or (type == 'R') or (type == 'Re')  or (type == 2):
            smearing = cls._real(sigma, energy)

        else:
            raise ValueError("Type not known")

        crop_front = len(smearing[0,0,0,:])//2

        if crop_front == 0:
            raise ValueError("Sigma is too small")

        if len(smearing[0,0,0,:])%2 == 1:
            crop_end = crop_front
        else:
            crop_end = crop_front-1

        #Extend the dataset by a length of (2*crop_data) with constant values to shift the convolution distortion out of the region of interest
        if len(data.shape) == 1:
            data = np.hstack([np.ones(data.shape[:-1]+(2*crop_front,)).T*data[...,0].T,data.T,np.ones(data.shape[:-1]+(2*crop_end,)).T*data[...,-1].T]).T
        else:
            data = np.vstack([np.ones(data.shape[:-1]+(2*crop_front,)).T*data[...,0].T,data.T,np.ones(data.shape[:-1]+(2*crop_end,)).T*data[...,-1].T]).T


        if len(data.shape) == 1:
            data = convolve(data, smearing[0,0,0,:])
        elif len(data.shape) == 2:
            data = convolve(data, smearing[0,0,:,:])
        elif len(data.shape) == 3:
            data = convolve(data, smearing[0,:,:,:])
        else:
            data = convolve(data, smearing)
    
        if transposed:
            #Trim the convolution contribution and the extended constant values
            return data[...,3*crop_front:-crop_end*3].T        
        else:
            return data[...,3*crop_front:-crop_end*3]


    @classmethod
    def gaussian_smear(cls, data, energy, sigma):
        """ Smear the signal with a Gaussian smearing
        :type  data: np.ndarray
        :param data: the data to be smeared with the energy axis as last axis

        :type  sigma: float  
        :param sigma: the sigma value of the gauss
        
        :returns: the smeared signal
        """    
        return smear_data(data=data, energy=energy, sigma=sigma, type='Gaussian')

    @classmethod
    def gaussian_smear_signal(cls, s, sigma):
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

        s_smooth = copy.deepcopy(s)
        
        s_smooth.data = EELS.smear_data(data=hist, energy=eRange, sigma=sigma, type='Gaussian')
        s_smooth.metadata['General']['title']  = s.metadata['General']['title'] + " gaussian smearing s={}".format(sigma)
        s_smooth.metadata['General']['name']  = s.metadata['General']['name'] + " gaussian smearing s={}".format(sigma)
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

    