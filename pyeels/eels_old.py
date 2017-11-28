import hyperspy.api as hs
import _spectrum
import matplotlib.pyplot as plt
from ase import Atoms
import spglib as spg
import numpy as np

class Atom(object):
	"""Atom so define in unit Cell"""
	def __init__(self, number, position):
		self.number = number
		self.position = position


class Cell(object):
	"""Cell object """
	def __init__(self, lattice=np.eye(3), a=None, b=None, c=None, atoms=None):
		self._atoms = []
		self.lattice = lattice
		self.a = lattice[0]
		self.b = lattice[1]
		self.c = lattice[2]
		
		if isinstance(a,np.ndarray):
			try:
				self.a = a
			except:
				raise Warning("Vector a is not matching a three dimensional lattice")

		if isinstance(b,np.ndarray):
			try:
				self.b = b
			except:
				raise Warning("Vector b is not matching a three dimensional lattice")

		if isinstance(c,np.ndarray):
			try:
				self.c = c
			except:
				raise Warning("Vector c is not matching a three dimensional lattice")

		self._setVolume()

		self._setReciprocalSpace()

		if isinstance(atoms, list):
			for atom in atoms:
				self.addAtom(atom)

	def _setVolume(self):
		self.volume = np.dot(self.a, np.cross(self.b, self.c))

	def _setReciprocalSpace(self):
		self.a_resiprocal = np.cross(self.b, self.c) / self.volume * 2*np.pi
		self.b_resiprocal = np.cross(self.c, self.a) / self.volume * 2*np.pi
		self.c_resiprocal = np.cross(self.a, self.b) / self.volume * 2*np.pi

		self.brillouinZone = np.vstack([self.a_resiprocal, self.b_resiprocal, self.c_resiprocal])


	def addAtom(self, atom):
		self._atoms.append(atom)

	def getAtomNumbers(self):
		numbers = []
		for atom in self._atoms:
			numbers.append(atom.number)
		return numbers

	def getAtomPositons(self):
		positions = []
		for atom in self._atoms:
			positions.append(atom.position)
		return positions

	def __repr__(self):
		return "Lattice parameters:\n{}\nRelatice atom positions:\n {}".format(self.lattice, np.round(np.asarray(self.getAtomPositons()),2))

class Band(object):
	HBAR_C = 1973 #eV AA
	M_E = .511e6 #eV/c^2

	def __init__(self, k_grid, energies=None, waves=None):
		self.k_grid = k_grid
		if isinstance(energies, np.ndarray):
			self.energies = energies
		if isinstance(waves, np.ndarray):
			self.waves = waves

	def setParabolic(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,))):
		self.energies = energy_offset+(self.HBAR_C**2/(2*self.M_E))*((self.k_grid[:,0]-k_center[0])**2/effective_mass[0]\
					+(self.k_grid[:,1]-k_center[1])**2/effective_mass[1]+(self.k_grid[:,2]-k_center[2])**2/effective_mass[2])
	
		#self.energies = energy_offset+(self.HBAR_C**2/(2*self.M_E))*(k[:,0]**2+k[:,1]**2+k[:,2]**2)	


		self.waves = np.stack([np.zeros(self.energies.shape),np.ones(self.energies.shape)], axis=1)

	"""def __repr__(self):
					try:
						plt.plot(self.k_grid[:,0],self.energies) #does this work?
					except:
						plt.plot(self.k_grid,self.k_grid*0)
					return "Energyband: plt.show() to plot" """

class ModelSystem():
	def __init__(self, cell=None, atoms=None, fermiEnergy=0, temperature=0):
		self.fermiEnergy = fermiEnergy
		self.temperature = temperature
		self.bands = []

		if isinstance(cell, type(Cell())):
			self.cell = cell
		else:
			if isinstance(atoms, list):
				self.cell = Cell(atoms=atoms)
			else:
				self.cell = Cell()

	def spaceGroup(self):
		return spg.get_spacegroup((self.cell.lattice,
								   self.cell.getAtomPositons(),
								   self.cell.getAtomNumbers()), symprec=1e-5)

	def setFermiEnergy(self, fermiEnergy):
		self.fermiEnergy = fermiEnergy

	def setTemperature(self, temperature):
		self.temperature = temperature

	def addBand(self, band):
		self.bands.append(band)

	def addParabolicBand(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,))):
		k = np.dot(self.reciprocalGrid()[0],self.cell.brillouinZone)
		effective_mass = np.dot(effective_mass,self.cell.brillouinZone)
		k_center = np.dot(k_center,self.cell.brillouinZone)

		band = Band(k)
		band.setParabolic(energy_offset, effective_mass, k_center)
		self.addBand(band)
		
	def meshgrid(self, pointDensity):
		""" Defines a mesgrid of k-points in reciprocal space, with a required density of k-points per
		reciprocal aangstrom
		"""
		if isinstance(pointDensity, (float, int)):
			pointDensity = np.ones((3,))*pointDensity

		if isinstance(pointDensity, (list, tuple)):
			pointDensity = np.asarray(pointDensity)

		if isinstance(pointDensity, np.ndarray):
			self._pointDensity = pointDensity

	def diffractionGrid(self):
		diffractionZone = np.max(np.abs(self.cell.brillouinZone),axis=0)
		diffractionBins = np.round(diffractionZone*self._pointDensity/1.3+2).astype(int) #/1.3 is a bad temporarly solution
		for i in range(len(diffractionBins)):
			if (diffractionBins[i]%2==0):
				diffractionBins[i] += 1				
		return diffractionZone, diffractionBins

	def reciprocalGrid(self):
		if isinstance(self._pointDensity,type(None)):
			raise TypeError("No meshgrid is defined, use meshgrid(pointDensity) to set meshgrid.")
		else:
			gridZone = np.max(np.abs(self.cell.brillouinZone),axis=0)
			gridPoints = np.round(gridZone*self._pointDensity).astype(int)
			for i in range(len(gridPoints)):
				if (gridPoints[i]%2==0):
					gridPoints[i] += 1	

			mapping, grid = spg.get_ir_reciprocal_mesh(gridPoints, (self.cell.lattice, self.cell.getAtomPositons(), self.cell.getAtomNumbers()), is_shift=[0, 0, 0])
			k_grid = grid[np.unique(mapping)]/(gridPoints-1)
			
			k_list = []
			for i, map_id in enumerate(mapping[np.unique(mapping)]):
			    k_list.append((grid[mapping==map_id]/(self._pointDensity-1)).tolist()) #np.dot(,self.cell.brillouinZone)
			return k_grid, k_list

	def createMeta(self, name=None, title=None, authors=None, notes=None, signal_type=None, elements=None, model=None):
		""" Generate and organize info into a matadata dictionary recognized by hyperspy
		:returns: nested dictionary of information """

		if not name:
			name = "Unnamed_simulation"
		if not title:
			title = name
		if elements:
			symbol = Atoms.get_chemical_symbols()
			for i in range(elements):
				elements[i] = symbol[i]

		description = "Simulated material system\n See 'system' for further information."


		description += "Fermi energy:{}".format(self.fermiEnergy)

		metadata = {}
		metadata['General'] = {}

		metadata['General']['name'] = name
		metadata['General']['title'] = title
		metadata['General']['authors'] = authors
		metadata['General']['notes'] = notes

		metadata['Signal'] = {}
		metadata['Signal']['binned'] = True
		metadata['Signal']['signal_type'] = signal_type

		metadata['Sample'] = {}
		metadata['Sample']['elements'] = self.cell.getAtomNumbers()


		metadata['Sample']['system'] = {}
		metadata['Sample']['system']['cell'] = {}
		axes = ['a','b','c']
		for i in range(len(self.cell.lattice)):
			metadata['Sample']['system']['cell'][axes[i]] = self.cell.lattice[i]
		    
		metadata['Sample']['system']['fermiEnergy'] = self.fermiEnergy
		metadata['Sample']['system']['temperature'] = self.temperature


		metadata['Sample']['system']['bands'] = self.bands
		metadata['Sample']['description'] = description

		return metadata		


	def createSignal(self, data, eBin):
		"""Organize and convert data and axes into a hyperspy signal 
		:param data: the resulting array from simulation
		:param eBin: the energy binning used in simulation
		:returns: hyperspy signal 
		"""
		
		diffractionZone, diffractionBins = self.diffractionGrid()
		
		metadata = self.createMeta()

		s = hs.signals.BaseSignal(data, metadata=metadata)


		names = ["Energy", "q_x", "q_y", "q_z"]
		for i in range(len(data.shape)-1):
			name = names[i+1]
			s.axes_manager[2-i].name = name
			s.axes_manager[name].scale = diffractionZone[i]/(diffractionBins[i]-1)
			s.axes_manager[name].units = "AA-1"
			s.axes_manager[name].offset = -diffractionZone[i]/2
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
		:param fermiEnergy: a spesific fermiEnergy, if not defined the standard fermiEnergy is used
		:param temperature: a spesific temperature, if not defined the standard temperature is used
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

		diffractionZone, diffractionBins = self.diffractionGrid()
		print("CCD:")
		print(diffractionZone, diffractionBins)

		data = _spectrum.calculate_spectrum(diffractionZone, diffractionBins, self.cell.brillouinZone, self.reciprocalGrid()[1], np.stack(energyBands, axis=1),  np.stack(waveStates, axis=1), energyBins, self.fermiEnergy, self.temperature)

		return self.createSignal(data, energyBins)

