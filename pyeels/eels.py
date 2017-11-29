import hyperspy.api as hs
from pyeels.cpyeels import calculate_spectrum

import numpy as np

from multiprocessing import Pool, cpu_count



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
            s.axes_manager[name].scale = self.zone[i]/(self.bins[i]-1)
            s.axes_manager[name].units = "AA-1"
            s.axes_manager[name].offset = -self.zone[i]/2
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
                self.zone, 
                self.bins, 
                self.crystal.brillouinzone.lattice, 
                initial_band.k_list,
                np.stack(energyBands, axis=1),  
                np.stack(waveStates, axis=1), 
                self.energyBins, 
                self.fermienergy, 
                self.temperature
            )       

        return self._create_signal(data, energyBins)
        
    def calculate_eels_multiproc(self, energyBins, bands=(None,None), fermienergy=None, temperature=None, max_cpu=None):
        """ Calculate the momentum dependent scattering cross section of the system, using multiple processes
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 
        
        :type  fermienergy: float
        :param fermienergy: a spesific fermienergy in eV, if not defined the standard fermienergy is used
        
        :type  temperature: float
        :param temperature: a spesific temperature in Kelvin, if not defined the standard temperature is used

        :type  max_cpu: int
        :param max_cpu: The user defined maximum allowed number of CPU-cores, if not, the hardware limit is used. 
        """


        if temperature:
            self.temperature = temperature
        if fermienergy:
            self.fermienergy = fermienergy

        self.energyBins = energyBins
        

        bands = self.crystal.brillouinzone.bands[bands[0]:bands[1]]

        transitions = []
        for i, initial in enumerate(bands):
            for f, final in enumerate(bands[(i+1):]):
                f += i+1
                transitions.append((i,f, initial, final))

        if not max_cpu:
            max_cpu = cpu_count()

        if max_cpu > cpu_count():
            max_cpu = cpu_count()

        p = Pool(processes=min(len(transitions),max_cpu))
        signals = p.map(self._calculate, transitions)
        p.close()

        signal_total = signals[0]
        for signal in signals[1:]:
            signal_total += signal
        
        return self._create_signal(signal_total, energyBins)
                
    def _calculate(self, transition):
            i, f, initial_band, final_band = transition
            
            energyBands = [initial_band.energies, final_band.energies]
            waveStates =  [initial_band.waves, final_band.waves]
            
            return calculate_spectrum(
                self.zone, 
                self.bins, 
                self.crystal.brillouinzone.lattice, 
                initial_band.k_list,
                np.stack(energyBands, axis=1),  
                np.stack(waveStates, axis=1), 
                self.energyBins, 
                self.fermienergy, 
                self.temperature
            )       

        
    def __repr__(self):
        string = "EELS Signal Calculator:\n\nSignal name:\n\t{}\nAuthors:\n\t{}\nTitle:\n\t{}\nNotes:\n\t{}\n\n".format(self.name, self.authors, self.title, self.notes)
        string += "Temperature: {} K\tFermiEnergy: {} eV\n".format(self.temperature, self.fermienergy)
        return string