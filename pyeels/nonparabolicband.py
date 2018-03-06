from pyeels.crystal import Crystal
from pyeels.band import Band
import matplotlib.pyplot as plt
import spglib as spg
import pythtb as tb
import numpy as np
import logging
_logger = logging.getLogger(__name__)

class NonParabolicBand:
    """Non Parabolic band class constructed for simple simulations """
    _HBAR_C = 1973 #eV AA
    _M_E = .511e6 #eV/c^2
    
    def __init__(self, crystal):
        """ Create instance of the Parabolic Band model 
        
        :type  crystal: crystal object
        :param crystal: a crystal object containing atoms
        """
        
        self._crystal = crystal
        self._crystal.brillouinzone.band_model = "Non Parabolic"
        self._spg = (crystal.lattice, crystal.get_atom_positons(), crystal.get_atom_numbers())
        
        self._orbital_positons = []
        for atom in crystal.atoms:
            for orbital in atom.orbitals:
                self._orbital_positons.append(atom.position)
        
        self.set_grid()
        
        self._model = tb.tb_model(3,3,self._crystal.lattice, self._orbital_positons)

    def _calculate_non_parabolic(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,)), correction_factor=None):
        """ Calculate energy of non parabolic band in k-space suggested by T. Pisarkiewicz 
        
        :type  energy_offset: float
        :param energy_offset: the energy min/max of the band
        
        :type  effective_mass: ndarray
        :param effective_mass: the effective mass along each direction of the cell [m_a, m_b, m_c]
        
        :type  k_center: ndarray
        :param k_center: the center of the band in reciprocal space (within the brillouin zone) [k0_a, k0_b, k0_c]

        :type  correction_factor: float
        :param correction_factor: A correction factor determining the slope of the linear band further out in k-space, default=1/energy_offset

        :returns: energies at each k-grid point and corresponding wavefunctions
        """
        
        if isinstance(effective_mass, float):
            effective_mass = effective*np.ones((3,))

        if not correction_factor:
            if energy_offset != 0:
                correction_factor = 1/energy_offset
            else:
                correction_factor = 0

        if correction_factor != 0:
            energies = energy_offset+(-1+np.sqrt(1+4*correction_factor*(self._HBAR_C**2/(2*self._M_E))*((self._k_grid[:,0]-k_center[0])**2/effective_mass[0]\
                    +(self._k_grid[:,1]-k_center[1])**2/effective_mass[1]+(self._k_grid[:,2]-k_center[2])**2/effective_mass[2])))/(2*correction_factor)
        else: 
            # The limit correction_factor=0 goes to parabolic model
            energies = energy_offset+(self._HBAR_C**2/(2*self._M_E))*((self._k_grid[:,0]-k_center[0])**2/effective_mass[0]\
                    +(self._k_grid[:,1]-k_center[1])**2/effective_mass[1]+(self._k_grid[:,2]-k_center[2])**2/effective_mass[2])

        return energies
        
    def _calculate_non_parabolic_periodic(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,)), correction_factor=None):
        """ Calculate energy of non parabolic band in k-space suggested by T. Pisarkiewicz, with periodic boundary conditions in the Brillouin Zone
        
        :type  energy_offset: float
        :param energy_offset: the energy min/max of the band
        
        :type  effective_mass: ndarray
        :param effective_mass: the effective mass along each direction of the cell [m_a, m_b, m_c]
        
        :type  k_center: ndarray
        :param k_center: the center of the band in reciprocal space (within the brillouin zone) [k0_a, k0_b, k0_c]
        """

        energies = self._calculate_non_parabolic(energy_offset=energy_offset, effective_mass=effective_mass, k_center=k_center)


        if np.any(k_center != 0):
            k_shifts = np.eye(3)
            k_center_initial = k_center

            for dim in range(3):
                if k_center_initial[dim] > 0:
                    k_center = k_center_initial-k_shifts[dim]
                elif k_center_initial[dim] < 0:
                    k_center = k_center_initial+k_shifts[dim]
                energies = np.minimum(energies,self._calculate_non_parabolic(energy_offset=energy_offset, effective_mass=effective_mass, k_center=k_center))
        
        waves = np.stack([np.zeros(energies.shape),np.ones(energies.shape)], axis=1)
        
        return energies, waves
    
    def set_non_parabolic(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,)), correction_factor=None):
        """ Calculate bands and place as a band object in crystal 
        
        :type  energy_offset: float
        :param energy_offset: the energy min/max of the band
        
        :type  effective_mass: ndarray
        :param effective_mass: the effective mass along each direction of the cell [m_a, m_b, m_c]
        
        :type  k_center: ndarray
        :param k_center: the center of the band in reciprocal space (within the brillouin zone) [k0_a, k0_b, k0_c]
        """
        energies, waves = self._calculate_non_parabolic_periodic(energy_offset=energy_offset, effective_mass=effective_mass, k_center=k_center, correction_factor=correction_factor)
        self._crystal.brillouinzone.add_band(Band(k_grid=self._k_grid, energies=energies, waves=waves))
        

        
    def set_grid(self, mesh=3):
        """ Define the resolution of the reciprocal space
        
        :type  mesh: ndarray, list, int
        :param mesh: The number of point along a reciprocal latticevector
        """
        if isinstance(mesh, (float, int)):
            mesh = np.ones((3,),dtype=np.int)*int(mesh)

        if isinstance(mesh, (list, tuple)):
            mesh = np.asarray(mesh)

        if isinstance(mesh, np.ndarray):
            self._crystal.brillouinzone.mesh = mesh
            mapping, grid = spg.get_ir_reciprocal_mesh(mesh, self._spg, is_shift=[0, 0, 0])
            
            if np.any(mesh==np.array([1, 1, 1])):
                mesh+= (mesh==np.array([1, 1, 1]))*1

            self._k_grid = grid/(mesh-1)
        else:
            _logger.warning("Unknown type {} for mesh, try ndarray.".format(type(mesh)))
        
    
    def bandstructure(self, ylim=(None,None), color=None, ax=None):
        """ Plot a representation of the band structure
        
        :type  ylim: tuple, list
        :param ylim: lower and upper limit of y-values (ymin,ymax)
        """

        raise NotImplementedError

    def __repr__(self):
        return "Paraboliv band model for: \n \n {} \n".format(self._crystal)