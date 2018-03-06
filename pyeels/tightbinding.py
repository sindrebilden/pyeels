from pyeels.crystal import Crystal
from pyeels.band import Band
import matplotlib.pyplot as plt
import spglib as spg
import pythtb as tb
import numpy as np
import logging
_logger = logging.getLogger(__name__)

class TightBinding:
    """ Tight binding class constructed around the 'PythTB <http://physics.rutgers.edu/pythtb/>' package """
    def __init__(self, crystal):
        """ Create instance of the Tight binding model 
        
        :type  crystal: crystal object
        :param crystal: a crystal object containing atoms
        """
        
        self._crystal = crystal
        self._crystal.brillouinzone.band_model = "Tight Binding"
        self._spg = (crystal.lattice, crystal.get_atom_positons(), crystal.get_atom_numbers())
        
        self._orbital_positons = []
        for atom in crystal.atoms:
            for orbital in atom.orbitals:
                self._orbital_positons.append(atom.position)        
        
        self.setGrid()
        self.model = tb.tb_model(3,3,self._crystal.lattice, self._orbital_positons)
        
    def setGrid(self, mesh=3):
        """ Define the resolution of the reciprocal space
        
        :type  mesh: ndarray, list, int
        :param mesh: The number of point along a reciprocal latticevector
        """
        if isinstance(mesh, (float, int)):
            mesh = np.ones((3,),dtype=np.int)*int(mesh)

        if isinstance(mesh, (list, tuple)):
            mesh = np.asarray(mesh)

        if isinstance(mesh, np.ndarray):
            for i in range(len(mesh)):
                if (mesh[i]%2==0):
                    mesh[i] += 1

            self._crystal.brillouinzone.mesh = mesh
            mapping, grid = spg.get_ir_reciprocal_mesh(mesh, self._spg, is_shift=[0, 0, 0])
            
            if np.any(mesh==np.array([1, 1, 1])):
                mesh+= (mesh==np.array([1, 1, 1]))*1

            self._k_grid = grid/(mesh-1)
        else:
            _logger.warning("Unknown type {} for mesh, try ndarray.".format(type(mesh)))
        
        
    def display_pythtb(self):
        """ Displat the info from pythTB """
        self.model.display()
        
    def calculate(self, eig_vectors = False):
        """ Calculate band energies for the given k-points and place them as band objects in the crystal, 
        NB! replaces the full band structure
        
        :type  eig_vectors: bool
        :param eig_vectors: if eigen vectors are returned
        """
        self._crystal.brillouinzone.bands = []
        
        if eig_vectors:
            energies, waves = self.model.solve_all(self._k_grid,eig_vectors=eig_vectors)
        else:
            energies = self.model.solve_all(self._k_grid,eig_vectors=eig_vectors)
            waves = np.stack([np.zeros(energies.shape,dtype=np.complex128),np.ones(energies.shape,dtype=np.complex128)], axis=2)
        
        for i, band in enumerate(energies):
            self._crystal.brillouinzone.add_band(Band(k_grid=self._k_grid, energies=band, waves=waves[i]))


    
    def bandstructure(self, ylim=(None,None),  bands=(None,None), color=None, linestyle=None, marker=None, ax=None):
        """ Plot a representation of the band structure
        
        :type  ylim: tuple, list
        :param ylim: lower and upper limit of y-values (ymin,ymax)
        """

        """ seekpath automatic lines"""
        #path = sp.get_explicit_k_path((lattice, positions, numbers), False, recipe="hpkot", threshold=1e-5,reference_distance=1)
        #expath = path['explicit_kpoints_abs'][:5]
        #labels = path['explicit_kpoints_labels'][:5]

        """ manual lines"""
        path=[[0.0,0.0,0.5],[0.5,0.0,0.5],[0.5,0,0.0],[0.0,0.0,0.0],[0,0,0.5],[2./3.,1./3.,0.5],[2./3.,1./3.,0.0],[0,0,0]]
        label=(r'$A $',      r'$L$',       r'$M$',   r'$\Gamma$', r'$A $', r'$H$',  r'$K$',r'$\Gamma $')

        
        # call function k_path to construct the actual path
        (k_vec,k_dist,k_node)=self.model.k_path(path,301,report=False)

        evals = self.model.solve_all(k_vec)
        
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
    
        for band in evals[bands[0]:bands[1]]:
            ax.plot(k_dist, band, color=color, linestyle=linestyle, marker=marker)

        if not fig:
            return ax
        else:
            ax.set_ylim(ylim)
            return ax, fig

    def density_of_states(self, energybins):
        """ calculates the density of states (DOS) in the material
        :type  energybins: ndarray
        :param energybins: numpy array of the energy bins for DOS
        """

        DOS = np.zeros(energybins.shape)

        if len(self._crystal.brillouinzone.bands) > 0:
            for band in self._crystal.brillouinzone.bands:
                DOS += band.density_of_states(energybins)
            return DOS
        else:
            raise ValueError("No bands found in crystal, run calculate() to calculate bands.")

    def __repr__(self):
        return "Parabolic band model for: \n \n {} \n".format(self._crystal)





class WurtziteSP3(TightBinding):
    """ A wurtzite Tight Binding model that includes s,px,py,pz orbitals at each site.
    Model designed by 'Kobayashi et Al. 1983 <https://link.aps.org/doi/10.1103/PhysRevB.28.935>'"""
    def __init__(self, crystal):
        TightBinding.__init__(self, crystal)
        
    def get_hopping_parameters(self):
        """ Get a list of all hopping parameters in the model
        :returns: Vss, Vxx, Vxy, Vsapc, Vpasc. See :func:'set_hopping_parameters' for info."""
        return [self._Vss, self._Vxx, self._Vxy, self._Vsapc, self._Vpasc ]
        
        
    def set_hopping_parameters(self, Vss, Vxx, Vxy, Vsapc, Vpasc):
        """ Set all hopping parameters in the model
        
        :type  Vss: float
        :param Vss: The hopping parameter from s to s
        
        :type  Vxx: float
        :param Vxx: The hopping parameter from px of the anion to px of the cation
        
        :type  Vxy: float
        :param Vxy: The hopping parameter from px of the anion to py of the cation
        
        :type  Vsapc: float
        :param Vsapc: The hopping parameter from s of the anion to p of the cation
        
        :type  Vpasc: float
        :param Vpasc: The hopping parameter from p of the anion to s of the cation
        """
        self._Vss = Vss
        self._Vxx = Vxx
        self._Vxy = Vxy
        self._Vsapc = Vsapc
        self._Vpasc = Vpasc
        
        """###############     Vertical bonding system    ##################"""      
        self._UVss = 0.25*Vss
        self._UVzz = 0.25*(Vxx+2*Vxy)
        self._UVxx = 0.25*(Vxx-Vxy)
        self._UVsz = -0.25*np.sqrt(3)*Vsapc
        self._UVzs =  0.25*np.sqrt(3)*Vpasc
        
        
        """###############    Horizontal bonding system   ##################"""
        self._UHss = self._UVss
        
        self._UHyy = self._UVxx
        self._UHzz = 1/9 * (8*self._UVxx +   self._UVzz)
        self._UHxx = 1/9 * (  self._UVxx + 8*self._UVzz)
        
        self._UHsz = -1/3 * self._UVsz
        self._UHzs = -1/3 * self._UVzs
        
        self._UHsx = -2*np.sqrt(2)/3 * self._UVsz
        self._UHxs = -2*np.sqrt(2)/3 * self._UVzs
        
        self._UHzx =  2*np.sqrt(2)/9 * (self._UVzz-self._UVxx)
        self._UHxz =  2*np.sqrt(2)/9 * (self._UVzz-self._UVxx)

        """##############################################################"""

        """  ONSITE  """

        for i, initial_atom in enumerate(self._crystal.atoms):
            for io, initial_orbital in enumerate(initial_atom.orbitals):
                self.model.set_onsite(initial_orbital.onsite, ind_i=(i*len(self._crystal.atoms)+io), mode='reset')
        


        """  HOPPING  """
                                                 
        self._M03(0,3)
        self._M12(1,2)
        
        self._M13(1,3)
        self._M02(0,2)
                
        
    def _M03(self, i, f):
        """ The M14 transition matrix designed by Kobayashi et Al."""
        self.model.set_hop(self._UVss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, -1], mode='reset')            
        
    def _M12(self, i, f):
        """ The M14 transition matrix designed by Kobayashi et Al."""
        self.model.set_hop(self._UVss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')  
              
    def _M02(self, i, f):
        """ The M13 transition matrix designed by Kobayashi et Al."""
        #s-s
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #s-z
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #s-x
        self.model.set_hop(     self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        #s-y
        self.model.set_hop( np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """##########################################"""
        #z-s
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #z-z
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #z-x
        self.model.set_hop(     self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        #z-y
        self.model.set_hop( np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """#########################################"""
        #x-s
        self.model.set_hop(     self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #x-z
        self.model.set_hop(     self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #x-x
        self.model.set_hop(     self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='add')        
        #x-y
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """###########################################"""
        #y-s
        self.model.set_hop( np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #y-z
        self.model.set_hop( np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #y-x
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')  
        #y-y
        self.model.set_hop(     self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='add')    

    def _M13(self, i, f):
        """ The M24 transition matrix designed by Kobayashi et Al."""
        #s-s
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #s-z
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #s-x
        self.model.set_hop(   -self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        #s-y
        self.model.set_hop(-np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """##########################################"""
        #z-s
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #z-z
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #z-x
        self.model.set_hop(   -self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        #z-y
        self.model.set_hop(-np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """#########################################"""
        #x-s
        self.model.set_hop(   -self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #x-z
        self.model.set_hop(   -self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #x-x
        self.model.set_hop(     self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 1, 0], mode='add')        
        #x-y
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """###########################################"""
        #y-s
        self.model.set_hop(-np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #y-z
        self.model.set_hop(-np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #y-x
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')  
        #y-y
        self.model.set_hop(     self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 1, 0], mode='add')        
    
    
    def __repr__(self):
        return "Wursite SP3 Tight binding model for: \n \n {} \n".format(self._crystal)