import hyperspy.api as hs
import _spectrum
import numpy as np

from multiprocessing import Pool



class EELS:
    
    temperature = 0
    fermiEnergy = 0
    
    def __init__(self, crystal, name=None):
        self.crystal = crystal
        

    def setDiffractionZone(self, zone=None, bins=None):
        """ Define the resolution of the diffraction space, similar to the CCD in TEM
        
        :type  zone: ndarray
        :param zone: The range of the diffraction space, a value containing the full brillouin zone is calculated if no value given
        
        :type  bins: ndarray
        :param bins: The number of bins in diffraction space, a value corresponding to the resolution of the brillouin zone is calculated if no value given. Will allways round up to an odd number
        """
        if not zone:
            self.zone = np.max(np.abs(self.crystal.brillouinZone.lattice),axis=0)
        else:
            self.zone = zone
            
        if not bins:
            self.bins = np.round(self.crystal.brillouinZone.mesh).astype(int) #/1.3 is a bad temporarly solution
        else:
            self.bins = bins
        
        for i in range(len(self.bins)):
            if (self.bins[i]%2==0):
                self.bins[i] += 1
        

    def setMeta(self, name, authors, title=None, notes=None):
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
        
    def _createMeta(self):
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
        metadata['Sample']['elements'] = self.crystal.getAtomNumbers()


        metadata['Sample']['system'] = {}
        metadata['Sample']['system']['cell'] = {}
        axes = ['a','b','c']
        for i in range(len(self.crystal.lattice)):
            metadata['Sample']['system']['cell'][axes[i]] = self.crystal.lattice[i]

        metadata['Sample']['system']['fermiEnergy'] = self.fermiEnergy
        metadata['Sample']['system']['temperature'] = self.temperature

        metadata['Sample']['system']['model'] = self.crystal.brillouinZone.band_model
        metadata['Sample']['system']['bands'] = {}
        metadata['Sample']['system']['bands']['count'] = len(self.crystal.brillouinZone.bands)
        for i, band in enumerate(self.crystal.brillouinZone.bands):
            metadata['Sample']['system']['bands']["band {}".format(i)] = self.crystal.brillouinZone.bands[i]
                
        metadata['Sample']['description'] = None

        return metadata


    def createSignal(self, data, eBin):
        """Organize and convert data and axes into a hyperspy signal 
        :param data: the resulting array from simulation
        :param eBin: the energy binning used in simulation
        :returns: hyperspy signal 
        """

        metadata = self._createMeta()

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

    
    def calculateScatteringCrossSection(self, energyBins, fermiEnergy=None, temperature=None):
        """ Calculate the momentum dependent scattering cross section of the system,
        
        :type  energyBins: ndarray
        :param energyBins: The binning range 
        
        :type  fermiEnergy: float
        :param fermiEnergy: a spesific fermiEnergy in eV, if not defined the standard fermiEnergy is used
        
        :type  temperature: float
        :param temperature: a spesific temperature in Kelvin, if not defined the standard temperature is used
        """

        if temperature:
            self.temperature = temperature
        if fermiEnergy:
            self.fermiEnergy = fermiEnergy

        

        energyBands = []
        waveStates = []
        for band in self.bands:
            energyBands.append(band.energies)
            waveStates.append(band.waves)

        #data = _spectrum.calculate_spectrum(diffractionZone, diffractionBins, self.cell.brillouinZone, self.reciprocalGrid()[1], np.stack(energyBands, axis=1),  np.stack(waveStates, axis=1), energyBins, self.fermiEnergy, self.temperature)

        return self.createSignal(data, energyBins)
        
    def multiCalculateScatteringCrossSection(self, energyBins, bands=(None,None), fermiEnergy=None, temperature=None):
        if temperature:
            self.temperature = temperature
        if fermiEnergy:
            self.fermiEnergy = fermiEnergy

        self.energyBins = energyBins
        

        bands = self.crystal.brillouinZone.bands[bands[0]:bands[1]]

        transitions = []
        for i, initial in enumerate(bands):
            for f, final in enumerate(bands[(i+1):]):
                f += i+1
                transitions.append((i,f, initial, final))
        
        p = Pool(processes=min(len(transitions),4))
        signals = p.map(self.calculate, transitions)
        p.close()

        signal_total = signals[0]
        for signal in signals[1:]:
            signal_total += signal
        
        return self.createSignal(signal_total, energyBins)
                
    def calculate(self, transition):
            i, f, initial_band, final_band = transition
            
            energyBands = [initial_band.energies, final_band.energies]
            waveStates =  [initial_band.waves, final_band.waves]
            
            return _spectrum.calculate_spectrum(
                self.zone, 
                self.bins, 
                self.crystal.brillouinZone.lattice, 
                initial_band.k_list,
                np.stack(energyBands, axis=1),  
                np.stack(waveStates, axis=1), 
                self.energyBins, 
                self.fermiEnergy, 
                self.temperature
            )       

        
    def __repr__(self):
        string = "EELS Signal Calculator:\n\nSignal name:\n\t{}\nAuthors:\n\t{}\nTitle:\n\t{}\nNotes:\n\t{}\n\n".format(self.name, self.authors, self.title, self.notes)
        string += "Temperature: {} K\tFermiEnergy: {} eV\n".format(self.temperature, self.fermiEnergy)
        return string



class thread_object:
    def __init__(self, initial_band, final_band, signal):
        """ Create instance of a thread object, helpes sorting data
        
        :type  signal: ndarray
        :param signal: the signal-array from code
        """
        self.initial_band = initial_band
        self.final_band = final_band
        self.signal = signal
    def __repr__(self):
        return "From {} to {}\n".format(self.initial_band, self.final_band)