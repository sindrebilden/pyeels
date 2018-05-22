PyEELS - Python EELS simulation package
=======================================

PyEELS is a python package intended for simulating Electron Energy Loss Spectroscopy from model band structures. 


The use of the package can be seen as threefold:
1. Constructing a real space model system
2. Create/generate model band structures in reciprocal space
3. Simulate EELS on the model band structure


The creation of model band structures is mainly based on [PythTB](http://physics.rutgers.edu/pythtb/), an additional model with parabolic bands is also provided.

The simulation of EELS is implemented in an C extension and can be multiprocessed for faster calculation.

[hyperspy](http://hyperspy.org/) has been chosen as a framework for an interactive visualization of EELS-spectra, in this process Jupyter Notebook is a natural platform for scripting purpose.

A minimal working example is presented below:

```python
# Generate real space crystal model
from pyeels import Crystal

myCrystal = Crystal(lattice=np.eye(3))
myAtom = Atom(number=0,position=[0,0,0])
myAtom.add_orbital(Orbital(label="s", onsite=0))
myCrystal.add_atom(atom=myAtom)

# Create parabolic bands in reciprocal space
from pyeels import ParabolicBand
reci = ParabolicBand(mySystem)

reci.set_grid(mesh=31) # Number of k-points in each dimension

# Parabolic valence band
reci.set_parabolic(effective_mass=[-0.5, -0.5, -0.5], 
                   energy_offset=0, 
                   k_center=[0,0,0], 
                   wave=np.array([0,0.02])
                  )
		 
# Parabolic conduction band
reci.set_parabolic(effective_mass=[ 0.5,  0.5,  0.5], 
                   energy_offset=1, 
                   k_center=[0,0,0],
                   wave=np.array([0,1])
                  )

# Calculate EELS on the parabolic bands
from pyeels import EELS

mySystem = EELS(myCrystal)
mySystem.temperature = 0    # Absolute zero
mySystem.fermienergy = 0.5  # Placing the fermi level at center of the band gap

mySystem.set_meta(
	name="My test sample", 
	author=["Author1", "Author2"], 
	title="myCrystal", 
	notes="This model is just an example." 
	)

# The q-resolution of the scattering cross section
# no argument correspond to the density of the k-grid in Brillouin Zone
mySystem.set_diffraction_zone()

mySignal = mySystem.calculate_eels_multiproc(energyBins=np.linspace(0,4,200),
                                     	     smearing=0.05
                                     	     max_cpu=4
					    )
					    
#HyperSpy visuailzation
mySignal.plot()
```
