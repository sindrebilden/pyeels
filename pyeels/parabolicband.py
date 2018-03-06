from pyeels.crystal import Crystal
from pyeels.band import Band
import matplotlib.pyplot as plt
import spglib as spg
import pythtb as tb
import numpy as np
import logging
_logger = logging.getLogger(__name__)

class ParabolicBand:
    """ Parabolic band class constructed for simple simulations """
    _HBAR_C = 1973 #eV AA
    _M_E = .511e6 #eV/c^2
    
    def __init__(self, crystal):
        """ Create instance of the Parabolic Band model 
        
        :type  crystal: crystal object
        :param crystal: a crystal object containing atoms
        """
        
        self._crystal = crystal
        self._crystal.brillouinzone.band_model = "Parabolic"
        self._spg = (crystal.lattice, crystal.get_atom_positons(), crystal.get_atom_numbers())
        
        self._orbital_positons = []
        for atom in crystal.atoms:
            for orbital in atom.orbitals:
                self._orbital_positons.append(atom.position)
        
        self.set_grid()
        
        self._model = tb.tb_model(3,3,self._crystal.lattice, self._orbital_positons)
    
    def _calculate_parabolic(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,))):
        """ Calculate energy of parabolic band in k-space without any boundary conditions
        
        :type  energy_offset: float
        :param energy_offset: the energy min/max of the band
        
        :type  effective_mass: ndarray
        :param effective_mass: the effective mass along each direction of the cell [m_a, m_b, m_c]
        
        :type  k_center: ndarray
        :param k_center: the center of the band in reciprocal space (within the brillouin zone) [k0_a, k0_b, k0_c]
    
        :returns: energy band as an array
        """


        return energy_offset+(self._HBAR_C**2/(2*self._M_E))*((self._k_grid[:,0]-k_center[0])**2/effective_mass[0]\
                            +(self._k_grid[:,1]-k_center[1])**2/effective_mass[1]+(self._k_grid[:,2]-k_center[2])**2/effective_mass[2])

    def _calculate_parabolic_periodic(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,))):
        """ Calculate energy of parabolic band in k-space with periodic boundary conditions in the Brillouin Zone
        
        :type  energy_offset: float
        :param energy_offset: the energy min/max of the band
        
        :type  effective_mass: ndarray
        :param effective_mass: the effective mass along each direction of the cell [m_a, m_b, m_c]
        
        :type  k_center: ndarray
        :param k_center: the center of the band in reciprocal space (within the brillouin zone) [k0_a, k0_b, k0_c]
        """

        energies = self._calculate_parabolic(energy_offset=energy_offset, effective_mass=effective_mass, k_center=k_center)

        

        if np.any(k_center != 0):
            k_shifts = np.eye(3)
            k_center_initial = k_center

            for dim in range(3):
                if k_center_initial[dim] > 0:
                    k_center = k_center_initial-k_shifts[dim]
                elif k_center_initial[dim] < 0:
                    k_center = k_center_initial+k_shifts[dim]
                energies = np.minimum(energies,self._calculate_parabolic(energy_offset=energy_offset, effective_mass=effective_mass, k_center=k_center))
        
        waves = np.stack([np.zeros(energies.shape),np.ones(energies.shape)], axis=1)
        
        return energies, waves
    
    def set_parabolic(self, energy_offset=0, effective_mass=np.ones((3,)), k_center=np.zeros((3,))):
        """ Calculate bands and place as a band object in crystal 
        
        :type  energy_offset: float
        :param energy_offset: the energy min/max of the band
        
        :type  effective_mass: ndarray
        :param effective_mass: the effective mass along each direction of the cell [m_a, m_b, m_c]
        
        :type  k_center: ndarray
        :param k_center: the center of the band in reciprocal space (within the brillouin zone) [k0_a, k0_b, k0_c]
        """
        energies, waves = self._calculate_parabolic_periodic(energy_offset=energy_offset, effective_mass=effective_mass, k_center=k_center)
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

        """ seekpath automatic lines"""
        #path = sp.get_explicit_k_path((lattice, positions, numbers), False, recipe="hpkot", threshold=1e-5,reference_distance=1)
        #expath = path['explicit_kpoints_abs'][:5]
        #labels = path['explicit_kpoints_labels'][:5]

        """ manual lines"""

        raise NotImplementedError

        # To be implemented
        """
        path=[[0.0,0.0,0.5],[0.5,0.0,0.5],[0.5,0,0.0],[0.0,0.0,0.0],[0,0,0.5],[2./3.,1./3.,0.5],[2./3.,1./3.,0.0],[0,0,0]]
        label=(r'$A $',      r'$L$',       r'$M$',   r'$\Gamma$', r'$A $', r'$H$',  r'$K$',r'$\Gamma $')

        
        # call function k_path to construct the actual path
        (k_vec,k_dist,k_node)=self._model.k_path(path,301,report=False)

        evals = self._calculateParabolic(k_vec)[:1]

        fig = None
        if not ax:
            fig, ax = plt.subplots(figsize=(8,6))
            fig.tight_layout()

            ax.set_title("Bandstructure for Zno based on Kobayashi")
            ax.set_ylabel("Band energy")

            # specify horizontal axis details
            ax.set_xlim([0,k_node[-1]])
            # put tickmarks and labels at node positions
            ax.set_xticks(k_node)
            ax.set_xticklabels(label)
            # add vertical lines at node positions

            for n in range(len(k_node)):
                if label[n] == r'$\Gamma$':
                    ax.axvline(x=k_node[n],linewidth=1, color='k')
                else:
                    ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    
        for band in evals:
            ax.plot(k_dist, band, color=color)

        if not fig:
            return ax
        else:
            ax.set_ylim(ylim)
            return ax, fig
        """
    def __repr__(self):
        return "Paraboliv band model for: \n \n {} \n".format(self._crystal)